"""
Gaussian Naive Bayes — written from scratch.
Uses log-space arithmetic throughout to avoid underflow on 639 features.
"""
import numpy as np


class GaussianNB:
    """
    Multiclass Gaussian Naive Bayes.

    Assumes each feature is conditionally independent and Gaussian given
    the class label.  Despite the independence assumption being violated
    in FMA (features are highly correlated), GNB serves as a useful
    lower-bound baseline.

    Parameters
    ----------
    var_smoothing : float
        Fraction of the globally largest per-class variance added to all
        variances for numerical stability.  On z-scored data max_var ≈ 1,
        so the effective epsilon equals var_smoothing itself.
        Search grid: {1e-11, 1e-9 (middle/default), 1e-7}.
    """

    def __init__(self, var_smoothing=1e-9):  # 1e-9 is the middle search value
        self.var_smoothing = var_smoothing

    # ------------------------------------------------------------------ #
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        n_classes  = len(self.classes_)
        n_features = X.shape[1]

        self._theta     = np.zeros((n_classes, n_features))  # per-class means
        self._var       = np.zeros((n_classes, n_features))  # per-class variances
        self._log_prior = np.zeros(n_classes)

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self._theta[i]     = X_c.mean(axis=0)
            self._var[i]       = X_c.var(axis=0)
            self._log_prior[i] = np.log(X_c.shape[0] / X.shape[0])

        # Smoothing: add epsilon to every variance to prevent log(0)
        epsilon = self.var_smoothing * self._var.max()
        if epsilon == 0.0:                       # degenerate fallback
            epsilon = self.var_smoothing
        self._var += epsilon

        return self

    # ------------------------------------------------------------------ #
    def _log_posterior(self, X):
        """
        Compute unnormalised log-posterior for each class.

        log P(c | x) ∝ log P(c) + Σ_j log N(x_j | μ_cj, σ²_cj)
                     = log P(c) + Σ_j [ -½ log(2π σ²_cj) - (x_j - μ_cj)² / (2 σ²_cj) ]

        Returns (n_samples, n_classes).
        """
        X = np.asarray(X, dtype=np.float64)
        n_classes = len(self.classes_)
        log_posts = np.empty((X.shape[0], n_classes))

        for i in range(n_classes):
            log_pdf = -0.5 * np.sum(
                np.log(2.0 * np.pi * self._var[i])
                + (X - self._theta[i]) ** 2 / self._var[i],
                axis=1,
            )
            log_posts[:, i] = self._log_prior[i] + log_pdf

        return log_posts

    def predict(self, X):
        return self.classes_[np.argmax(self._log_posterior(X), axis=1)]

    def get_params(self):
        return {"var_smoothing": self.var_smoothing}
