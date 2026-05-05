"""
Multinomial Logistic Regression trained with mini-batch SGD — from scratch.

Key design decisions:
  * Internal Z-score normalisation (fit on training fold only, applied to test).
  * Numerically stable softmax (shift by row-max before exp).
  * L2 regularisation controlled by C (same convention as sklearn: λ = 1/C).
  * Tuned hyperparameter: C ∈ {0.01, 0.1, 1.0}.
"""
import numpy as np


class LogisticRegressionSGD:
    """
    Parameters
    ----------
    C          : float  — inverse L2 regularisation strength  (λ = 1/C)
    lr         : float  — SGD learning rate
    epochs     : int    — passes over the training data
    batch_size : int    — mini-batch size
    seed       : int    — random seed for reproducible shuffling
    """

    def __init__(self, C=1.0, lr=0.01, epochs=100, batch_size=256, seed=42):
        self.C          = C
        self.lr         = lr
        self.epochs     = epochs
        self.batch_size = batch_size
        self.seed       = seed

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _softmax(z):
        """Row-wise numerically stable softmax."""
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def _normalise(self, X):
        return (X - self._mu) / self._sigma

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #
    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        X   = np.asarray(X, dtype=np.float64)
        y   = np.asarray(y)

        # --- Fit normaliser on training data only ----------------------
        self._mu    = X.mean(axis=0)
        self._sigma = X.std(axis=0) + 1e-8        # avoid div-by-zero
        X = self._normalise(X)

        n, F          = X.shape
        self.classes_ = np.unique(y)
        K             = len(self.classes_)

        # Map original labels → 0 … K-1
        self._label_map = {int(c): i for i, c in enumerate(self.classes_)}
        y_int = np.array([self._label_map[int(c)] for c in y], dtype=np.int32)

        # Weight initialisation (zeros → unbiased start)
        self.W_ = np.zeros((F, K))
        self.b_ = np.zeros(K)

        lam = 1.0 / (self.C * n)   # L2 coefficient per sample

        for _ in range(self.epochs):
            perm = rng.permutation(n)

            for start in range(0, n, self.batch_size):
                end  = start + self.batch_size
                bx   = X[perm[start:end]]           # (bs, F)
                by   = y_int[perm[start:end]]        # (bs,)
                bs   = len(by)

                # Forward
                probs = self._softmax(bx @ self.W_ + self.b_)  # (bs, K)

                # Cross-entropy gradient w.r.t. logits
                grad = probs.copy()
                grad[np.arange(bs), by] -= 1.0
                grad /= bs                           # (bs, K)

                # Weight update with L2 penalty on W (not bias)
                self.W_ -= self.lr * (bx.T @ grad + lam * self.W_)
                self.b_ -= self.lr * grad.sum(axis=0)

        return self

    def predict(self, X):
        X      = self._normalise(np.asarray(X, dtype=np.float64))
        logits = X @ self.W_ + self.b_
        idx    = np.argmax(self._softmax(logits), axis=1)
        return self.classes_[idx]

    def get_params(self):
        return {"C": self.C, "lr": self.lr, "epochs": self.epochs}
