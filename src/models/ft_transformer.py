"""
ft_transformer.py — FT-Transformer for tabular data (Gorishniy et al., NeurIPS 2021).

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
    d_token ∈ {64, 128, 256}   — embedding dimension for every token.
    n_heads and n_layers are fixed to control search space size.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data


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
        self.norm1 = nn.LayerNorm(d_token)
        self.attn  = nn.MultiheadAttention(
            d_token, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn   = nn.Sequential(
            nn.Linear(d_token, 4 * d_token),
            nn.GELU(),                          # paper uses GELU, not ReLU
            nn.Dropout(dropout),
            nn.Linear(4 * d_token, d_token),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention sublayer (pre-norm)
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.drop(attn_out)
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
    d_token    : int   — embedding dimension (tuned hyperparameter)
    n_heads    : int   — attention heads (fixed across experiments)
    n_layers   : int   — number of Transformer blocks (fixed)
    dropout    : float — dropout rate
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
    d_token    : int   — token embedding dimension; tuned ∈ {64, 128, 256}
    n_heads    : int   — attention heads; fixed at 8
    n_layers   : int   — Transformer blocks; fixed at 3
    dropout    : float — dropout rate; fixed at 0.1
    lr         : float — AdamW learning rate; fixed at 1e-4
    epochs     : int   — maximum training epochs
    batch_size : int   — mini-batch size
    patience   : int   — early-stopping patience (epochs without improvement)
    seed       : int   — random seed for reproducibility
    """

    def __init__(self, d_token=128, n_heads=8, n_layers=3, dropout=0.1,
                 lr=1e-4, epochs=100, batch_size=256, patience=10, seed=42):
        self.d_token    = d_token
        self.n_heads    = n_heads
        self.n_layers   = n_layers
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

        # Z-score normalisation (fit on training split only to prevent leakage)
        self._mu    = X.mean(axis=0)
        self._sigma = X.std(axis=0) + 1e-8
        X = (X - self._mu) / self._sigma

        self.classes_ = np.unique(y)
        K = len(self.classes_)
        self._label_map = {int(c): i for i, c in enumerate(self.classes_)}
        y_int = np.array([self._label_map[int(c)] for c in y], dtype=np.int64)

        # Use GPU if available (RTX 3080 / Colab), otherwise fall back to CPU
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model = _FTTransformer(
            X.shape[1], K, self.d_token, self.n_heads, self.n_layers, self.dropout
        ).to(self._device)

        # AdamW with weight decay matches the paper's training setup
        opt     = torch.optim.AdamW(self._model.parameters(),
                                    lr=self.lr, weight_decay=1e-5)
        loss_fn = nn.CrossEntropyLoss()

        Xt = torch.from_numpy(X)
        yt = torch.from_numpy(y_int)
        ds = torch.utils.data.TensorDataset(Xt, yt)
        g  = torch.Generator()
        g.manual_seed(self.seed)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size, shuffle=True, generator=g
        )

        best_loss    = float("inf")
        patience_cnt = 0
        best_state   = None

        self._model.train()
        for _ in range(self.epochs):
            epoch_loss = 0.0
            for Xb, yb in dl:
                Xb, yb = Xb.to(self._device), yb.to(self._device)
                opt.zero_grad()
                loss = loss_fn(self._model(Xb), yb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * len(yb)

            epoch_loss /= len(y_int)

            if epoch_loss < best_loss - 1e-4:
                best_loss    = epoch_loss
                patience_cnt = 0
                # Save best weights so we can restore after patience runs out
                best_state = {k: v.clone()
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
            Xt     = torch.from_numpy(X).to(self._device)
            logits = self._model(Xt)
            idx    = logits.argmax(dim=1).cpu().numpy()
        return self.classes_[idx]

    def get_params(self):
        return {"d_token": self.d_token, "n_heads": self.n_heads,
                "n_layers": self.n_layers, "epochs": self.epochs, "lr": self.lr}
