"""
ResNet for tabular data — Gorishniy et al., NeurIPS 2021.

Architecture:
    input → Linear(F→d) → BN → ReLU
          → [Linear(d→d) → BN → ReLU → Dropout → Linear(d→d) → BN → skip + ReLU] × n_blocks
          → Linear(d→K)

Key design decisions:
  * Internal Z-score normalisation (fit on training fold only).
  * Early stopping based on training-loss plateau (no validation leakage).
  * PyTorch used as an execution framework — forward/backward handled by autograd.
  * Tuned hyperparameter: d ∈ {64, 128, 256}.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data


# =========================================================================== #
# PyTorch architecture                                                         #
# =========================================================================== #

class _ResBlock(nn.Module):
    """One residual block: two linear layers with BN, ReLU, Dropout + skip."""

    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d, bias=False),
            nn.BatchNorm1d(d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, d, bias=False),
            nn.BatchNorm1d(d),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class _ResNet(nn.Module):
    def __init__(self, in_features: int, n_classes: int,
                 d: int, n_blocks: int, dropout: float):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(in_features, d, bias=False),
            nn.BatchNorm1d(d),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *[_ResBlock(d, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Linear(d, n_classes)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


# =========================================================================== #
# sklearn-style wrapper                                                        #
# =========================================================================== #

class ResNetTabularClassifier:
    """
    Parameters
    ----------
    d          : int    — hidden dimension (tuned: 64, 128, 256)
    n_blocks   : int    — number of residual blocks (fixed at 3)
    dropout    : float  — dropout probability inside each block
    lr         : float  — Adam learning rate
    epochs     : int    — maximum training epochs
    batch_size : int    — mini-batch size
    patience   : int    — early-stopping patience (based on train loss plateau)
    seed       : int    — random seed
    """

    def __init__(self, d=128, n_blocks=3, dropout=0.1,
                 lr=1e-3, epochs=50, batch_size=256, patience=10, seed=42):
        self.d          = d
        self.n_blocks   = n_blocks
        self.dropout    = dropout
        self.lr         = lr
        self.epochs     = epochs
        self.batch_size = batch_size
        self.patience   = patience
        self.seed       = seed

    # ------------------------------------------------------------------ #
    def fit(self, X, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        # --- Internal normalisation (fit on X_train only) -------------
        self._mu    = X.mean(axis=0)
        self._sigma = X.std(axis=0) + 1e-8
        X = (X - self._mu) / self._sigma

        self.classes_ = np.unique(y)
        K = len(self.classes_)
        self._label_map = {int(c): i for i, c in enumerate(self.classes_)}
        y_int = np.array([self._label_map[int(c)] for c in y], dtype=np.int64)

        # --- Build model ----------------------------------------------
        self._model = _ResNet(X.shape[1], K, self.d, self.n_blocks, self.dropout)
        opt     = torch.optim.Adam(self._model.parameters(),
                                   lr=self.lr, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()

        Xt = torch.from_numpy(X)
        yt = torch.from_numpy(y_int)
        ds = torch.utils.data.TensorDataset(Xt, yt)
        g  = torch.Generator()
        g.manual_seed(self.seed)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size, shuffle=True, generator=g
        )

        # --- Training loop with early stopping on train loss ----------
        best_loss    = float("inf")
        patience_cnt = 0
        best_state   = None

        self._model.train()
        for _ in range(self.epochs):
            epoch_loss = 0.0
            for Xb, yb in dl:
                opt.zero_grad()
                loss = loss_fn(self._model(Xb), yb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * len(yb)

            epoch_loss /= len(y_int)

            if epoch_loss < best_loss - 1e-4:
                best_loss    = epoch_loss
                patience_cnt = 0
                best_state   = {k: v.clone()
                                for k, v in self._model.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)

        return self

    # ------------------------------------------------------------------ #
    def predict(self, X):
        X = (np.asarray(X, dtype=np.float32) - self._mu) / self._sigma
        self._model.eval()
        with torch.no_grad():
            logits = self._model(torch.from_numpy(X))
            idx    = logits.argmax(dim=1).numpy()
        return self.classes_[idx]

    def get_params(self):
        return {"d": self.d, "n_blocks": self.n_blocks,
                "epochs": self.epochs, "lr": self.lr}
