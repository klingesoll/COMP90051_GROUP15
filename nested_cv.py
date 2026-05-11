"""
nested_cv.py — Nested Stratified Cross-Validation engine (from scratch).

Why nested CV?
    A single cross-validation loop both selects hyperparameters *and*
    estimates generalisation performance on the same data, causing
    optimistic bias: the reported score reflects the best hyperparameter
    rather than true out-of-sample performance.  Nested CV removes this
    bias by keeping the outer test set completely invisible during
    hyperparameter selection.

Structure
---------
  Outer loop  (K_out folds): each fold yields one unbiased performance
              estimate on data that was never used for model selection.
  Inner loop  (K_in  folds): runs entirely inside the outer training split
              to select the best hyperparameter via cross-validation.

  For each outer fold i:
    1. Split full data  →  (train_out_i,  test_out_i)
       test_out_i is sealed and not accessed again until step 5.
    2. For each hyperparameter combination p:
           inner-CV on train_out_i  →  mean validation macro-F1
    3. best_params_i  =  argmax over p of mean inner macro-F1
    4. Retrain a fresh model on all of train_out_i with best_params_i
    5. Evaluate on test_out_i  →  one outer F1 / accuracy estimate

  Final reported performance = mean ± std of the K_out outer estimates.

Leakage prevention
------------------
  * Structural: inner splits are created from X_tr_out, so their indices
    are *relative* to that sub-array — it is structurally impossible for
    outer test samples to appear in inner folds.
  * Runtime: explicit ``assert`` checks verify disjointness of train/test
    index sets at both the outer and inner levels.

Persistence
-----------
  Each outer fold's result is written to a JSON file immediately after the
  fold completes.  If the process is interrupted (e.g. power loss on a long
  ResNet run) the completed folds are not lost and can be resumed manually.

Assignment constraints satisfied
---------------------------------
  * Cross-validation implemented from scratch (no sklearn.KFold).
  * Hyperparameter tuning implemented from scratch (no GridSearchCV).
  * Outer K=10, inner K=3, each algorithm uses the same outer splits.
"""
import json
import os
from collections import Counter

import numpy as np

from kfold import stratified_kfold
from metrics import macro_f1, accuracy, minority_group_recall


# =========================================================================== #
# Main entry point                                                             #
# =========================================================================== #

def _winsorize(X_train, X_apply):
    """
    Clip X_apply to the [p1, p99] bounds estimated from X_train only.

    Parameters
    ----------
    X_train : np.ndarray  — source for computing percentile bounds (training fold)
    X_apply : np.ndarray  — array to clip (may be X_train itself, or test/val fold)

    Returns
    -------
    np.ndarray — clipped copy of X_apply (same shape, no in-place modification)

    Why here and not in preprocessing?
        Winsorization must be fitted on training data only and applied to test
        data using training bounds — exactly like Z-score normalisation.
        Computing global percentiles on the full dataset would leak test-fold
        distribution information into the preprocessing step.
    """
    p1  = np.percentile(X_train, 1,  axis=0)
    p99 = np.percentile(X_train, 99, axis=0)
    return np.clip(X_apply, p1, p99)


def nested_cv(
    X, y,
    model_factory,
    param_grid,
    outer_k=10,
    inner_k=3,
    seed=42,
    winsorize=True,
    save_dir=None,
    model_name="model",
    verbose=True,
):
    """
    Run nested stratified cross-validation for a single model class.

    All three models (GNB, LR, ResNet-tabular) must use the same outer
    splits (same ``seed``) so their per-fold scores are paired and their
    mean ± std comparisons are valid.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.  Should already be normalised / winsorised.
    y : np.ndarray, shape (n_samples,)
        Integer class labels.
    model_factory : callable
        A class or function that accepts keyword arguments from
        ``param_grid`` and returns a model object exposing:
          - ``.fit(X_train, y_train)``
          - ``.predict(X_test) → np.ndarray``
        A fresh instance is created for every inner and outer fold to
        avoid state leakage between folds.
    param_grid : list of dict
        Each dict is one hyperparameter combination to evaluate.
        Example: [{"C": 0.01}, {"C": 0.1}, {"C": 1.0}]
        Requirement: at least 3 values per hyperparameter, and the middle
        value should be selected most often (per assignment rubric).
    outer_k : int, optional
        Number of outer folds.  Assignment requires 10.
    inner_k : int, optional
        Number of inner (validation) folds.  Assignment requires >= 3.
    winsorize : bool, optional
        If True (default), apply per-fold Winsorization [p1, p99] to each
        outer and inner split using bounds estimated from the training fold
        only.  Must be True when X is X_medium_raw_derived (unclipped
        derived features); set to False only when X is already bounded
        (e.g. X_medium.npy, used for quick debugging).
    seed : int, optional
        Master random seed.  Outer splits use ``seed``; inner splits for
        outer fold i use ``seed + i + 1`` to prevent fold alignment.
    save_dir : str or None, optional
        Directory for per-fold JSON files.  If None, results are only
        returned in memory.
    model_name : str, optional
        Prefix for JSON filenames, e.g. "gnb" → "gnb_fold00.json".
    verbose : bool, optional
        If True, print a one-line summary after each outer fold.

    Returns
    -------
    results : dict with keys:
        ``outer_f1``        — list[float], macro-F1 on outer test per fold
        ``outer_acc``       — list[float], accuracy on outer test per fold
        ``best_params``     — list[dict],  best hyperparams chosen per fold
        ``inner_f1_matrix`` — list[list[float]], mean inner val macro-F1
                              for each (outer fold, param combo) pair

    Notes
    -----
    The ``±`` values reported by ``print_summary`` are the *standard
    deviations* across outer folds, which serve as the error bars required
    by the assignment.  They quantify how much performance varies depending
    on which 10% of the data is held out.
    """
    # Outer splits span the full dataset; same seed across all models
    # ensures fold assignments are identical for fair comparison.
    outer_splits = stratified_kfold(y, outer_k, seed=seed)

    outer_f1         = []
    outer_acc        = []
    outer_min_rec    = []
    best_params_list = []
    inner_f1_matrix  = []

    for outer_i, (train_out_idx, test_out_idx) in enumerate(outer_splits):

        # ── Runtime leakage guard (outer level) ────────────────────────
        # The structural design already prevents overlap, but an explicit
        # check makes any future refactor bug immediately visible.
        overlap = np.intersect1d(train_out_idx, test_out_idx)
        assert len(overlap) == 0, (
            f"[BUG] Outer fold {outer_i}: "
            f"{len(overlap)} indices appear in both train and test"
        )

        X_tr_out = X[train_out_idx].copy()   # shape (n_train, n_features)
        y_tr_out = y[train_out_idx]
        X_te_out = X[test_out_idx].copy()    # sealed — not touched again until step 5
        y_te_out = y[test_out_idx]

        # ── Per-fold Winsorization (outer level) ───────────────────────
        # Bounds estimated from outer training data only, then applied to
        # both outer train and outer test.  This prevents the test fold's
        # distribution from influencing the clipping thresholds.
        if winsorize:
            X_tr_out = _winsorize(X_tr_out, X_tr_out)
            X_te_out = _winsorize(X[train_out_idx], X_te_out)  # use original train for bounds

        # ── Inner CV: hyperparameter selection ─────────────────────────
        # Offset seed by outer_i so inner fold boundaries do not accidentally
        # align with outer fold boundaries across iterations.
        inner_splits = stratified_kfold(
            y_tr_out, inner_k, seed=seed + outer_i + 1
        )

        inner_scores_per_param = []

        for params in param_grid:
            fold_scores = []

            for inner_j, (tr_rel, val_rel) in enumerate(inner_splits):
                # ── Runtime leakage guard (inner level) ────────────────
                # tr_rel / val_rel are indices into y_tr_out (0-based),
                # not into the original y — structural impossibility of
                # outer-test contamination, but we verify anyway.
                assert len(np.intersect1d(tr_rel, val_rel)) == 0, (
                    f"[BUG] Inner fold {inner_j}: train/val overlap"
                )
                assert val_rel.max() < len(y_tr_out), (
                    "[BUG] Inner val index exceeds outer train size — "
                    "global indices mistakenly passed to inner CV"
                )

                # ── Per-fold Winsorization (inner level) ───────────────
                # X_tr_out is already outer-Winsorized, but inner sub-folds
                # re-Winsorize using only the inner training portion so the
                # inner validation fold cannot influence the bounds.
                X_in_tr  = X_tr_out[tr_rel].copy()
                X_in_val = X_tr_out[val_rel].copy()
                if winsorize:
                    X_in_tr  = _winsorize(X_tr_out[tr_rel], X_tr_out[tr_rel])
                    X_in_val = _winsorize(X_tr_out[tr_rel], X_tr_out[val_rel])

                # Train on inner-train, evaluate on inner-validation.
                # A new model instance per fold prevents weight sharing.
                model = model_factory(**params)
                model.fit(X_in_tr, y_tr_out[tr_rel])
                preds = model.predict(X_in_val)
                assert set(preds).issubset(set(np.unique(y_tr_out[tr_rel]))), (
                    f"[BUG] {model_factory.__name__}.predict() returned labels "
                    f"outside the training label space — likely returning "
                    f"internal 0..K-1 indices instead of original class IDs"
                )
                fold_scores.append(macro_f1(y_tr_out[val_rel], preds))
                del X_in_tr, X_in_val  # free memory immediately

            # Average inner-fold scores for this hyperparameter combination
            inner_scores_per_param.append(float(np.mean(fold_scores)))

        inner_f1_matrix.append(inner_scores_per_param)

        # ── Select best hyperparameters ─────────────────────────────────
        # argmax picks the param combo with highest mean inner-fold F1.
        # Ties are broken by the earlier index (i.e. the smaller param
        # value listed first in param_grid).
        best_idx    = int(np.argmax(inner_scores_per_param))
        best_params = param_grid[best_idx]
        best_params_list.append(best_params)

        # ── Final model: retrain on full outer-train split ──────────────
        # Using *all* of train_out (not just one inner fold) maximises the
        # information available for the final estimator.
        final_model = model_factory(**best_params)
        final_model.fit(X_tr_out, y_tr_out)

        # Outer test set is used exactly once — here.
        preds_out = final_model.predict(X_te_out)
        assert set(preds_out).issubset(set(np.unique(y_tr_out))), (
            f"[BUG] {model_factory.__name__}.predict() returned labels "
            f"outside the training label space — likely returning "
            f"internal 0..K-1 indices instead of original class IDs"
        )
        f1  = macro_f1(y_te_out, preds_out)
        acc = accuracy(y_te_out, preds_out)
        min_rec = minority_group_recall(y_te_out, preds_out)

        outer_f1.append(f1)
        outer_acc.append(acc)
        outer_min_rec.append(min_rec)

        if verbose:
            inner_str = "  ".join(f"{s:.3f}" for s in inner_scores_per_param)
            print(
                f"  [fold {outer_i+1:2d}/{outer_k}]"
                f"  F1={f1:.4f}  acc={acc:.4f}  min_rec={min_rec:.4f}"
                f"  best={best_params}"
                f"  inner=[{inner_str}]"
            )

        # ── Persist result immediately ──────────────────────────────────
        # Writing after each fold (not at the end) ensures that a crash
        # mid-run (e.g. during the 20-minute ResNet training) does not
        # discard already-completed folds.
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            record = {
                "outer_fold"  : outer_i,
                "macro_f1"    : f1,
                "accuracy"    : acc,
                "minority_recall" : min_rec,
                "best_params" : best_params,
                "inner_scores": [
                    {
                        "params"        : param_grid[k],
                        "mean_inner_f1" : inner_scores_per_param[k],
                    }
                    for k in range(len(param_grid))
                ],
                "n_train": int(len(train_out_idx)),
                "n_test" : int(len(test_out_idx)),
            }
            path = os.path.join(
                save_dir, f"{model_name}_fold{outer_i:02d}.json"
            )
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(record, fh, indent=2)

    return {
        "outer_f1"       : outer_f1,
        "outer_acc"      : outer_acc,
        "outer_minority_recall": outer_min_rec,
        "best_params"    : best_params_list,
        "inner_f1_matrix": inner_f1_matrix,
    }


# =========================================================================== #
# Summary helper                                                               #
# =========================================================================== #

def print_summary(results, model_name):
    """
    Print mean ± std of macro-F1 and accuracy across all outer folds,
    plus the frequency with which each hyperparameter combination was
    selected by the inner CV.

    The std serves as the error bar for the assignment's results tables.
    The hyperparameter frequency table is used to verify the assignment
    requirement that the *middle* candidate value wins most often — if a
    boundary value wins, the search range must be extended and the
    experiment re-run.

    Parameters
    ----------
    results : dict
        Return value of ``nested_cv()``.
    model_name : str
        Display name printed in the header (e.g. "GNB").
    """
    f1s  = np.array(results["outer_f1"])
    accs = np.array(results["outer_acc"])
    min_rec = np.array(results["outer_minority_recall"])
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"  macro-F1 : {f1s.mean():.4f} ± {f1s.std():.4f}"
          f"  [{f1s.min():.4f} – {f1s.max():.4f}]")
    print(f"  accuracy : {accs.mean():.4f} ± {accs.std():.4f}")
    print(f"  minority_rec : {min_rec.mean():.4f} ± {min_rec.std():.4f}")
    freq = Counter(str(p) for p in results["best_params"])
    print(f"  best-param frequency : {dict(freq)}")
    print(f"{'='*60}")
