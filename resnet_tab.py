"""
resnet_tab.py — FT-Transformer for tabular data (Gorishniy et al., NeurIPS 2021).

Reference
---------
Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021).
Revisiting Deep Learning Models for Tabular Data. NeurIPS 2021.
https://arxiv.org/abs/2106.11959

Why FT-Transformer over ResNet-tabular (same paper)?
    Both models are from the same NeurIPS 2021 paper, so the literature
    citation is identical.  FT-Transformer is architecturally more complex
    (Self-Attention vs residual MLP), making it the better representative
    of the "complex" tier in the simple/medium/complex model comparison.
    The computational cost (O(F²) attention over F=639 features × 640
    sequence length) is feasible on an RTX 3080 or Colab Pro GPU but
    prohibitive on CPU — this trade-off is discussed in the report.

Architecture
------------
    input (N, F)
        → FeatureTokenizer  : each feature j gets its own linear weight
                              W_j ∈ ℝ^d, producing F tokens of dimension d.
                              A learnable [CLS] token is prepended.
          output: (N, F+1, d)
        → TransformerEncoder × n_layers
              each layer (pre-norm style, as in paper):
                x = x + Attention(LayerNorm(x))
                x = x + FFN(LayerNorm(x))
              FFN: Linear(d→4d) → GELU → Dropout → Linear(4d→d)
          output: (N, F+1, d)
        → CLS token extraction : x[:, 0, :]       shape (N, d)
        → LayerNorm → Linear(d→K)                 shape (N, K)

Tuned hyperparameter
--------------------
    dropout ∈ {0.0, 0.1, 0.3}  — dropout rate applied in attention and FFN.
    d_token, n_heads, and n_layers are fixed to control search space size.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm


# =========================================================================== #
# PyTorch sub-modules                                                          #
# =========================================================================== #

class _FeatureTokenizer(nn.Module):
    """
    Maps a flat feature vector to a sequence of per-feature token embeddings,
    and prepends a learnable [CLS] token.

    Each feature j has its own weight vector w_j ∈ ℝ^d and bias b_j ∈ ℝ^d,
    so the tokenizer is a column-wise linear projection:
        token_j = x_j * w_j + b_j       (scalar × vector + vector)

    This gives every feature its own embedding space rather than sharing
    weights, which is the key design of the FT-Transformer paper.

    Parameters
    ----------
    n_features : int   — number of input features (F = 639)
    d_token    : int   — embedding dimension per token
    """

    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        # One weight and one bias vector per feature
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias   = nn.Parameter(torch.zeros(n_features, d_token))
        # Learnable [CLS] token prepended to the sequence
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.kaiming_uniform_(self.weight, a=0.0)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x: (N, F)
        # Unsqueeze x to (N, F, 1) then broadcast multiply with weight (F, d)
        tokens = x.unsqueeze(-1) * self.weight + self.bias  # (N, F, d)
        cls    = self.cls_token.expand(x.size(0), -1, -1)   # (N, 1, d)
        return torch.cat([cls, tokens], dim=1)               # (N, F+1, d)


class _TransformerBlock(nn.Module):
    """
    One FT-Transformer encoder layer using pre-norm residual connections.

    Pre-norm (LayerNorm applied *before* the sublayer) is used as in the
    original paper and is more stable than post-norm for deep networks.

    Structure:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    FFN hidden dimension is 4 × d_token, following the standard Transformer
    convention from Vaswani et al. (2017).

    Parameters
    ----------
    d_token  : int   — token embedding dimension
    n_heads  : int   — number of attention heads (d_token must be divisible)
    dropout  : float — dropout applied after attention weights and in FFN
    """

    def __init__(self, d_token: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = d_token // n_heads
        self.dropout  = dropout
        self.norm1  = nn.LayerNorm(d_token)
        self.q_proj = nn.Linear(d_token, d_token)
        self.k_proj = nn.Linear(d_token, d_token)
        self.v_proj = nn.Linear(d_token, d_token)
        self.out_proj = nn.Linear(d_token, d_token)
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn   = nn.Sequential(
            nn.Linear(d_token, 4 * d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_token, d_token),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        normed = self.norm1(x)
        # Q/K/V projections → (B, n_heads, T, head_dim)
        Q = self.q_proj(normed).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(normed).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(normed).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # Flash Attention (PyTorch 2.0+): fused kernel, no O(T²) materialisation
        dp = self.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(Q, K, V, dropout_p=dp)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.drop(self.out_proj(attn_out))
        # FFN sublayer (pre-norm)
        x = x + self.ffn(self.norm2(x))
        return x


class _FTTransformer(nn.Module):
    """
    Full FT-Transformer: tokenizer → encoder stack → CLS head.

    Parameters
    ----------
    n_features : int   — number of input features
    n_classes  : int   — number of output classes
    d_token    : int   — embedding dimension (fixed at 128)
    n_heads    : int   — attention heads (fixed; derived as d_token // 64)
    n_layers   : int   — number of Transformer blocks (fixed at 3)
    dropout    : float — dropout rate (tuned hyperparameter)
    """

    def __init__(self, n_features, n_classes, d_token, n_heads, n_layers, dropout):
        super().__init__()
        self.tokenizer = _FeatureTokenizer(n_features, d_token)
        self.encoder   = nn.Sequential(
            *[_TransformerBlock(d_token, n_heads, dropout)
              for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, n_classes)

    def forward(self, x):
        tokens = self.tokenizer(x)          # (N, F+1, d)
        tokens = self.encoder(tokens)       # (N, F+1, d)
        cls    = self.norm(tokens[:, 0])    # (N, d)  — CLS token only
        return self.head(cls)               # (N, K)


# =========================================================================== #
# sklearn-style wrapper                                                        #
# =========================================================================== #

class FTTransformerClassifier:
    """
    Scikit-learn-style wrapper around _FTTransformer.

    Exposes .fit() and .predict() so it can be passed to nested_cv()
    as model_factory without any changes to the CV engine.

    Normalisation: Z-score is computed on the training fold and applied to
    both train and test.  Parameters are stored on the instance so that
    predict() uses the same shift and scale as fit().

    Early stopping: monitors training loss; stops if no improvement of
    more than 1e-4 for ``patience`` consecutive epochs.  This avoids a
    separate validation split inside the already-nested CV.

    Parameters
    ----------
    d_token    : int   — token embedding dimension; fixed at 128
    n_heads    : int   — derived as d_token // 64 (head_dim fixed at 64)
    n_layers   : int   — Transformer blocks; fixed at 3
    dropout    : float — dropout rate; tuned ∈ {0.0, 0.1, 0.3}
    lr         : float — AdamW learning rate; fixed at 1e-4
    epochs     : int   — maximum training epochs
    batch_size : int   — mini-batch size
    patience   : int   — early-stopping patience (epochs without improvement)
    seed       : int   — random seed for reproducibility
    """

    def __init__(self, d_token=128, n_heads=2, n_layers=3, dropout=0.1,
                 lr=1e-4, weight_decay=1e-5, epochs=100, batch_size=256, patience=10, seed=42):
        self.d_token      = d_token
        self.n_heads      = n_heads
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

        # Z-score normalisation (fit on training split only to prevent leakage)
        self._mu    = X.mean(axis=0)
        self._sigma = X.std(axis=0) + 1e-8
        X = (X - self._mu) / self._sigma

        self.classes_ = np.unique(y)
        K = len(self.classes_)
        self._label_map = {int(c): i for i, c in enumerate(self.classes_)}
        y_int = np.array([self._label_map[int(c)] for c in y], dtype=np.int64)

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available — RTX 3080 required for training")
        self._device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self._model = _FTTransformer(
            X.shape[1], K, self.d_token, self.n_heads, self.n_layers, self.dropout
        ).to(self._device)

        # AdamW with weight decay matches the paper's training setup
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
        epoch_bar = tqdm(range(self.epochs), desc=f"d={self.d_token}",
                         unit="ep", ncols=80, leave=False)
        for epoch in epoch_bar:
            perm = torch.randperm(N, device=self._device, generator=g)
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
                best_state = {k: v.clone()
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
        return {"d_token": self.d_token, "n_heads": self.n_heads,
                "n_layers": self.n_layers, "dropout": self.dropout,
                "lr": self.lr, "weight_decay": self.weight_decay,
                "epochs": self.epochs}
