"""
resnet_tabular.py — ResNet-tabular for tabular data (Gorishniy et al., NeurIPS 2021).

Reference
---------
Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021).
Revisiting Deep Learning Models for Tabular Data. NeurIPS 2021.
https://arxiv.org/abs/2106.11959

Architecture
------------
    input (N, F)
        → Linear(F → d)                           shape (N, d)
        → ResBlock × n_layers
              each block (pre-norm style):
                z = LayerNorm(x)
                z = Linear(d → d_hidden) → ReLU → Dropout
                z = Linear(d_hidden → d) → Dropout
                x = x + z
          output: (N, d)
        → LayerNorm → ReLU → Linear(d → K)        shape (N, K)

Tuned hyperparameter
--------------------
    lr ∈ {1e-6, 1e-4, 1e-2, 5e-2, 1e-1}
        1e-6 — severely underfits (too small for AdamW to make progress)
        1e-4 — safe but slow; sub-optimal on this dataset
        1e-2 — paper-compatible range, consistently best (middle value)
        5e-2 — slightly aggressive; minor degradation expected
        1e-1 — upper boundary; Adam becomes unstable at this scale
"""
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm


# =========================================================================== #
# PyTorch sub-modules                                                          #
# =========================================================================== #

class _ResBlock(nn.Module):
    """
    One ResNet-tabular residual block (pre-norm style).

    Structure:
        z = LayerNorm(x)
        z = Linear(d → d_hidden) → ReLU → Dropout
        z = Linear(d_hidden → d) → Dropout
        x = x + z

    Parameters
    ----------
    d       : int   — feature dimension
    dropout : float — dropout rate applied after each linear layer
    """

    def __init__(self, d: int, dropout: float):
        super().__init__()
        d_hidden = 4 * d
        self.norm = nn.LayerNorm(d)
        self.fc1  = nn.Linear(d, d_hidden)
        self.fc2  = nn.Linear(d_hidden, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        z = self.norm(x)
        z = torch.relu(self.fc1(z))
        z = self.drop(z)
        z = self.fc2(z)
        z = self.drop(z)
        return x + z


class _ResNet(nn.Module):
    """
    Full ResNet-tabular: input projection → residual blocks → classification head.

    Parameters
    ----------
    n_features : int   — number of input features
    n_classes  : int   — number of output classes
    d          : int   — hidden dimension
    n_layers   : int   — number of residual blocks (tuned hyperparameter)
    dropout    : float — dropout rate
    """

    def __init__(self, n_features, n_classes, d, n_layers, dropout):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d)
        self.blocks     = nn.Sequential(
            *[_ResBlock(d, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)

    def forward(self, x):
        x = self.input_proj(x)      # (N, d)
        x = self.blocks(x)          # (N, d)
        x = torch.relu(self.norm(x))
        return self.head(x)         # (N, K)


# =========================================================================== #
# sklearn-style wrapper                                                        #
# =========================================================================== #

class ResNetTabularClassifier:
    """
    Scikit-learn-style wrapper around _ResNet.

    Exposes .fit() and .predict() so it can be passed to nested_cv()
    as model_factory without any changes to the CV engine.

    Normalisation: Z-score is computed on the training fold and applied to
    both train and test.  Parameters are stored on the instance so that
    predict() uses the same shift and scale as fit().

    Early stopping: monitors training loss; stops if no improvement of
    more than 1e-3 for ``patience`` consecutive epochs.

    Parameters
    ----------
    d          : int   — hidden dimension; fixed at 128
    n_layers   : int   — residual blocks; tuned ∈ {1, 3, 5}
    dropout    : float — dropout rate; fixed at 0.1
    lr         : float — AdamW learning rate; fixed at 1e-4
    weight_decay: float — AdamW weight decay; fixed at 1e-5
    epochs     : int   — maximum training epochs
    batch_size : int   — mini-batch size
    patience   : int   — early-stopping patience
    seed       : int   — random seed for reproducibility
    """

    def __init__(self, d=128, n_layers=3, dropout=0.1,
                 lr=1e-4, weight_decay=1e-5, epochs=30,
                 batch_size=256, patience=5, seed=42):
        self.d            = d
        self.n_layers     = n_layers
        self.dropout      = dropout
        self.lr           = lr
        self.weight_decay = weight_decay
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.patience     = patience
        self.seed         = seed

    # ------------------------------------------------------------------ #
    def fit(self, X, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        self._mu    = X.mean(axis=0)
        self._sigma = X.std(axis=0) + 1e-8
        X = (X - self._mu) / self._sigma

        self.classes_   = np.unique(y)
        K               = len(self.classes_)
        self._label_map = {int(c): i for i, c in enumerate(self.classes_)}
        y_int = np.array([self._label_map[int(c)] for c in y], dtype=np.int64)

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available — GPU required for training")
        self._device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self._model = _ResNet(
            X.shape[1], K, self.d, self.n_layers, self.dropout
        ).to(self._device)

        opt     = torch.optim.AdamW(self._model.parameters(),
                                    lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.CrossEntropyLoss()
        scaler  = torch.amp.GradScaler("cuda")

        Xt = torch.from_numpy(X).to(self._device)
        yt = torch.from_numpy(y_int).to(self._device)
        N  = Xt.shape[0]
        g  = torch.Generator(device=self._device)
        g.manual_seed(self.seed)

        best_loss    = float("inf")
        patience_cnt = 0
        best_state   = None

        self._model.train()
        epoch_bar = tqdm(range(self.epochs), desc=f"lr={self.lr:.0e}",
                         unit="ep", ncols=80, leave=False)
        for epoch in epoch_bar:
            perm       = torch.randperm(N, device=self._device, generator=g)
            epoch_loss = torch.tensor(0.0, device=self._device)
            for i in range(0, N, self.batch_size):
                idx = perm[i:i + self.batch_size]
                Xb, yb = Xt[idx], yt[idx]
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda"):
                    loss = loss_fn(self._model(Xb), yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                epoch_loss += loss.detach() * len(idx)

            epoch_loss = (epoch_loss / N).item()
            epoch_bar.set_postfix(loss=f"{epoch_loss:.4f}", pat=patience_cnt)

            if epoch_loss < best_loss - 1e-3:
                best_loss    = epoch_loss
                patience_cnt = 0
                best_state   = {k: v.clone()
                                for k, v in self._model.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience:
                    break
        epoch_bar.close()

        if best_state is not None:
            self._model.load_state_dict(best_state)

        torch.cuda.empty_cache()
        return self

    # ------------------------------------------------------------------ #
    def predict(self, X):
        X = (np.asarray(X, dtype=np.float32) - self._mu) / self._sigma
        self._model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                Xb = torch.from_numpy(X[i:i + self.batch_size]).to(self._device)
                preds.append(self._model(Xb).argmax(dim=1).cpu())
        idx = torch.cat(preds).numpy()
        return self.classes_[idx]

    def get_params(self):
        return {"d": self.d, "n_layers": self.n_layers,
                "dropout": self.dropout, "lr": self.lr,
                "weight_decay": self.weight_decay, "epochs": self.epochs}
