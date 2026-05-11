"""
stat_test.py -- Paired t-test and Wilcoxon signed-rank test for A vs B.

Research question
-----------------
Does adding 121 constructed features (feature set B, 639-dim) significantly
improve performance over the original 518 acoustic features (feature set A)?

Two primary metrics are tested:
  1. macro-F1         -- overall classification quality across all 16 genres
  2. minority recall  -- recall on the 4 rarest genres (International, Easy
                        Listening, Blues, Spoken), the core of the research question

Both tests are hand-written with NumPy only (no scipy.stats), per assignment spec.

Statistical framework
---------------------
  H0 : mean(score_B - score_A) = 0   (feature engineering has no effect)
  H1 : mean(score_B - score_A) > 0   (B is better than A, one-tailed)

  Paired t-test  (one-tailed):
      d_i    = score_B_i - score_A_i  for i = 1..10 outer folds
      t      = mean(d) / (std(d, ddof=1) / sqrt(n))
      df     = n - 1 = 9
      p      = P(T_{df} >= t)   via hand-written regularised incomplete beta CDF
      95% CI = mean(d) +/- t_{0.025, df=9} * std(d) / sqrt(n)
               (two-tailed CI, reported alongside one-tailed p)

  Shapiro-Wilk normality check (hand-written):
      If W < 0.85 or p_SW < 0.05, flag the distribution as non-normal and
      also report Wilcoxon signed-rank test as robustness check.

  Wilcoxon signed-rank test  (one-tailed, continuity-corrected):
      Rank |d_i|, sign by sign(d_i), compute W+ and W-.
      Normal approximation with continuity correction for n >= 5.

Usage
-----
  python scripts/stat_test.py                   # GNB A vs B (default)
  python scripts/stat_test.py --model lr        # LR  A vs B
  python scripts/stat_test.py --model ft        # FT  A vs B
  python scripts/stat_test.py --model all       # all three models
"""

import argparse
import glob
import json
import os

import math
import numpy as np

# -- paths ----------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.environ.get("FMA_RESULT_DIR", os.path.join(_ROOT, "results"))


# =========================================================================== #
# Data loading                                                                 #
# =========================================================================== #

def load_fold_scores(model, feat):
    """
    Load per-fold macro_f1 and minority_group_recall for one (model, feat) pair.

    Parameters
    ----------
    model : str   e.g. "gnb", "lr", "ft"
    feat  : str   "a" or "b"

    Returns
    -------
    f1s  : np.ndarray, shape (n_folds,)
    mgrs : np.ndarray, shape (n_folds,)
    """
    pattern = os.path.join(RESULT_DIR, model, f"feat_{feat}",
                           f"{model}_{feat}_fold*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No result files found for model={model}, feat={feat}.\n"
            f"Pattern: {pattern}\n"
            "Run scripts/run_all.py first."
        )
    f1s, mgrs = [], []
    for fp in files:
        d = json.load(open(fp))
        f1s.append(d["macro_f1"])
        mgrs.append(d["minority_group_recall"])
    return np.array(f1s), np.array(mgrs)


# =========================================================================== #
# Hand-written statistical primitives                                          #
# =========================================================================== #

# -- regularised incomplete beta function --------------------------------------
# Used for the t-distribution CDF (and indirectly Shapiro-Wilk).
# Implemented via continued fraction expansion (Lentz's algorithm).

def _betacf(a, b, x, max_iter=200, eps=3e-7):
    """Continued fraction for the regularised incomplete beta function."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c   = 1.0
    d   = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        # Even step
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d  = 1.0 + aa * d
        c  = 1.0 + aa / c
        if abs(d) < 1e-30: d = 1e-30
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        h *= d * c
        # Odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d  = 1.0 + aa * d
        c  = 1.0 + aa / c
        if abs(d) < 1e-30: d = 1e-30
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        delta = d * c
        h    *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    """Regularised incomplete beta function I_x(a, b)."""
    if x < 0.0 or x > 1.0:
        raise ValueError(f"x={x} outside [0,1]")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    # Use the symmetry relation for numerical stability
    lbeta = (math.lgamma(a) + math.lgamma(b)
             - math.lgamma(a + b))
    if x < (a + 1.0) / (a + b + 2.0):
        return np.exp(a * np.log(x) + b * np.log(1.0 - x) - lbeta) \
               * _betacf(a, b, x) / a
    else:
        return 1.0 - (np.exp(b * np.log(1.0 - x) + a * np.log(x) - lbeta)
                      * _betacf(b, a, 1.0 - x) / b)


def t_cdf_upper(t, df):
    """
    P(T_{df} >= t) -- upper-tail probability of the t-distribution.
    Used for one-tailed p-value.
    """
    x = df / (df + t * t)
    p = 0.5 * _betai(df / 2.0, 0.5, x)
    return p if t >= 0 else 1.0 - p


def t_quantile(p, df, tol=1e-8):
    """
    Inverse t-distribution: find t such that P(T_{df} <= t) = p.
    Used for 95% CI (p=0.975, df=9 -> t~2.262).
    Simple bisection search.
    """
    lo, hi = 0.0, 50.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if (1.0 - t_cdf_upper(mid, df)) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# -- paired t-test -------------------------------------------------------------

def paired_ttest_onetailed(a, b):
    """
    One-tailed paired t-test: H1: mean(b) > mean(a).

    Parameters
    ----------
    a, b : array-like, same length n

    Returns
    -------
    dict with keys: d_mean, d_std, t_stat, df, p_value, ci_95_lo, ci_95_hi
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    n    = len(a)
    d    = b - a
    d_mean = d.mean()
    d_std  = d.std(ddof=1)
    se     = d_std / np.sqrt(n)
    t_stat = d_mean / se
    df     = n - 1
    p      = t_cdf_upper(t_stat, df)
    t_crit = t_quantile(0.975, df)          # two-tailed 95% CI critical value
    return {
        "d_mean"    : float(d_mean),
        "d_std"     : float(d_std),
        "t_stat"    : float(t_stat),
        "df"        : df,
        "p_value"   : float(p),
        "ci_95_lo"  : float(d_mean - t_crit * se),
        "ci_95_hi"  : float(d_mean + t_crit * se),
        "per_fold_d": d.tolist(),
    }


# -- Shapiro-Wilk normality test -----------------------------------------------
# Coefficients for n=10 from Shapiro & Wilk (1965), Table 1.

_SW_A10 = np.array([0.5739, 0.3291, 0.2141, 0.1224, 0.0399])


def shapiro_wilk_n10(x):
    """
    Shapiro-Wilk W statistic for n=10.
    Returns (W, interpretation_string).
    Approximate p-value via look-up thresholds (alpha=0.05: W<0.842).
    """
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n != 10:
        return None, "SW only implemented for n=10"
    b = sum(_SW_A10[i] * (x[n - 1 - i] - x[i]) for i in range(5))
    s2 = np.var(x, ddof=1) * (n - 1)
    W  = b ** 2 / s2
    # Approximate critical value at alpha=0.05 for n=10: W_crit ~ 0.842
    normal = "likely normal (W >= 0.842)" if W >= 0.842 else "non-normal (W < 0.842) -> use Wilcoxon"
    return float(W), normal


# -- Wilcoxon signed-rank test -------------------------------------------------

def wilcoxon_onetailed(a, b):
    """
    One-tailed Wilcoxon signed-rank test: H1: median(b) > median(a).
    Normal approximation with continuity correction (valid for n >= 5).

    Returns
    -------
    dict with keys: W_plus, W_minus, z_stat, p_value
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    d    = b - a
    d    = d[d != 0]                   # drop zero differences
    n    = len(d)
    if n == 0:
        return {"W_plus": 0, "W_minus": 0, "z_stat": 0.0, "p_value": 0.5}

    ranks = np.argsort(np.argsort(np.abs(d))) + 1   # rank |d_i|, 1-based
    W_plus  = float(ranks[d > 0].sum())
    W_minus = float(ranks[d < 0].sum())

    # Normal approximation
    mu    = n * (n + 1) / 4
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    # Continuity correction: subtract 0.5 from |W+ - mu| to move toward H0
    z = (W_plus - mu - 0.5) / sigma
    # One-tailed p: P(Z >= z) using hand-written standard normal CDF
    p = _norm_upper(z)
    return {
        "W_plus" : W_plus,
        "W_minus": W_minus,
        "z_stat" : float(z),
        "p_value": float(p),
        "n_nonzero": n,
    }


def _norm_upper(z):
    """P(Z >= z) for standard normal, via error function approximation."""
    # erf approximation (Abramowitz & Stegun 7.1.26, max error 1.5e-7)
    t = 1.0 / (1.0 + 0.3275911 * abs(z))
    poly = t * (0.254829592
                + t * (-0.284496736
                       + t * (1.421413741
                              + t * (-1.453152027
                                     + t * 1.061405429))))
    p_upper = 0.5 * poly * np.exp(-z * z)
    return p_upper if z >= 0 else 1.0 - p_upper


# =========================================================================== #
# Report printer                                                               #
# =========================================================================== #

def report(model, metric_name, scores_a, scores_b):
    """Print a full statistical comparison for one model x one metric."""
    label = metric_name.replace("_", " ").title()
    print(f"\n{'-'*60}")
    print(f"  {model.upper()}  |  {label}  |  A vs B")
    print(f"{'-'*60}")
    print(f"  Feat A : {np.mean(scores_a):.4f} +/- {np.std(scores_a, ddof=1):.4f}"
          f"  (folds: {[f'{v:.4f}' for v in scores_a]})")
    print(f"  Feat B : {np.mean(scores_b):.4f} +/- {np.std(scores_b, ddof=1):.4f}"
          f"  (folds: {[f'{v:.4f}' for v in scores_b]})")

    # -- Normality check ----------------------------------------------------
    d = np.array(scores_b) - np.array(scores_a)
    W, sw_msg = shapiro_wilk_n10(d)
    print(f"\n  Shapiro-Wilk on d=B-A : W={W:.4f}  ->  {sw_msg}")

    # -- Paired t-test ------------------------------------------------------
    tt = paired_ttest_onetailed(scores_a, scores_b)
    sig = "[PASS] significant" if tt["p_value"] < 0.05 else "[FAIL] not significant"
    print(f"\n  Paired t-test (one-tailed H1: B > A)")
    print(f"    mean(d) = {tt['d_mean']:+.4f}  std(d) = {tt['d_std']:.4f}")
    print(f"    t = {tt['t_stat']:.4f}  df = {tt['df']}")
    print(f"    p = {tt['p_value']:.4f}  ->  {sig} (alpha=0.05)")
    print(f"    95% CI of mean(d): [{tt['ci_95_lo']:+.4f}, {tt['ci_95_hi']:+.4f}]")

    # -- Wilcoxon (always shown; primary if SW flags non-normality) ---------
    wc = wilcoxon_onetailed(scores_a, scores_b)
    primary = " <- PRIMARY (non-normal)" if "non-normal" in sw_msg else ""
    print(f"\n  Wilcoxon signed-rank (one-tailed H1: B > A){primary}")
    print(f"    W+ = {wc['W_plus']:.0f}  W- = {wc['W_minus']:.0f}"
          f"  n_nonzero = {wc['n_nonzero']}")
    print(f"    z = {wc['z_stat']:.4f}  p = {wc['p_value']:.4f}")
    print(f"{'-'*60}")


# =========================================================================== #
# Entry point                                                                  #
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="A vs B paired statistical tests -- FMA-Medium"
    )
    parser.add_argument(
        "--model", choices=["gnb", "lr", "ft", "all"], default="gnb",
        help="Which model's results to test (default: gnb)"
    )
    args = parser.parse_args()

    models = ["gnb", "lr", "ft"] if args.model == "all" else [args.model]

    print("=" * 60)
    print("  FMA-Medium  |  Feature Set A vs B  |  Statistical Tests")
    print("  H1 (one-tailed): score_B > score_A")
    print("=" * 60)

    for model in models:
        try:
            f1_a,  mgr_a  = load_fold_scores(model, "a")
            f1_b,  mgr_b  = load_fold_scores(model, "b")
        except FileNotFoundError as e:
            print(f"\n[SKIP {model.upper()}] {e}")
            continue

        report(model, "macro_f1",            f1_a,  f1_b)
        report(model, "minority_group_recall", mgr_a, mgr_b)

    print("\nDone.")


if __name__ == "__main__":
    main()
