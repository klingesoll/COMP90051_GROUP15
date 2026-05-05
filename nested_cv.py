"""
Nested Stratified Cross-Validation engine — from scratch.

Structure
---------
  Outer loop  : K_out folds  → each provides one unbiased performance estimate
  Inner loop  : K_in  folds  → used only to select the best hyperparameter

  For each outer fold i:
    1. Split data → (train_out, test_out)                [test_out NEVER seen in inner]
    2. For each param combo p:
         inner-CV on train_out only → mean val macro-F1
    3. best_params = argmax inner F1
    4. Retrain on train_out with best_params
    5. Evaluate on test_out                              [outer test used exactly once]

Leakage guards
--------------
  * Explicit assert checks that train/test index sets are disjoint.
  * Inner splits use indices RELATIVE to X_train_out (not global), so
    it is structurally impossible for outer test samples to appear in
    inner folds.

Persistence
-----------
  Per-fold JSON written immediately after each fold so partial results
  survive if the run is interrupted.
"""
import json
import os
from collections import Counter

import numpy as np

from kfold import stratified_kfold
from metrics import macro_f1, accuracy


# =========================================================================== #
# Main function                                                                #
# =========================================================================== #

def nested_cv(
    X, y,
    model_factory,
    param_grid,
    outer_k=10,
    inner_k=3,
    seed=42,
    save_dir=None,
    model_name="model",
    verbose=True,
):
    """
    Nested stratified cross-validation.

    Parameters
    ----------
    X             : np.ndarray (n, F)
    y             : np.ndarray (n,)
    model_factory : callable(**params) → model with .fit(X,y) and .predict(X)
    param_grid    : list of dicts — one dict per hyperparameter combination
    outer_k       : int  — outer folds   (10 for assignment)
    inner_k       : int  — inner folds   (3  for assignment)
    seed          : int  — master seed (outer uses seed; inner uses seed+outer_i+1)
    save_dir      : str  — directory for per-fold JSON results; None = skip
    model_name    : str  — prefix for JSON filenames
    verbose       : bool — print per-fold summary

    Returns
    -------
    dict with keys:
      outer_f1          : list[float]       — macro-F1 on outer test per fold
      outer_acc         : list[float]       — accuracy on outer test per fold
      best_params       : list[dict]        — best hyperparams selected per fold
      inner_f1_matrix   : list[list[float]] — mean inner val F1 per (fold, param)
    """
    # Outer splits are over the full dataset
    outer_splits = stratified_kfold(y, outer_k, seed=seed)

    outer_f1        = []
    outer_acc       = []
    best_params_list = []
    inner_f1_matrix  = []

    for outer_i, (train_out_idx, test_out_idx) in enumerate(outer_splits):

        # ── Leakage guard: outer train and test must be disjoint ────────
        overlap = np.intersect1d(train_out_idx, test_out_idx)
        assert len(overlap) == 0, \
            f"[BUG] Outer fold {outer_i}: {len(overlap)} indices in both train and test"

        X_tr_out = X[train_out_idx]   # shape (n_train, F)
        y_tr_out = y[train_out_idx]
        X_te_out = X[test_out_idx]    # NEVER touched until step 5
        y_te_out = y[test_out_idx]

        # ── Inner CV (all indices are RELATIVE to X_tr_out) ────────────
        # Using a different seed per outer fold ensures inner folds are
        # not accidentally aligned with outer folds.
        inner_splits = stratified_kfold(
            y_tr_out, inner_k, seed=seed + outer_i + 1
        )

        inner_scores_per_param = []

        for params in param_grid:
            fold_scores = []

            for inner_j, (tr_rel, val_rel) in enumerate(inner_splits):
                # ── Leakage guard: inner train/val must be disjoint ────
                assert len(np.intersect1d(tr_rel, val_rel)) == 0, \
                    f"[BUG] Inner fold {inner_j}: train/val overlap"
                assert val_rel.max() < len(y_tr_out), \
                    "[BUG] Inner val index exceeds outer train size"

                model = model_factory(**params)
                model.fit(X_tr_out[tr_rel], y_tr_out[tr_rel])
                preds = model.predict(X_tr_out[val_rel])
                fold_scores.append(macro_f1(y_tr_out[val_rel], preds))

            inner_scores_per_param.append(float(np.mean(fold_scores)))

        inner_f1_matrix.append(inner_scores_per_param)

        # ── Select best hyperparameters via inner CV ────────────────────
        best_idx    = int(np.argmax(inner_scores_per_param))
        best_params = param_grid[best_idx]
        best_params_list.append(best_params)

        # ── Retrain on full outer train; evaluate on outer test ─────────
        final_model = model_factory(**best_params)
        final_model.fit(X_tr_out, y_tr_out)

        preds_out = final_model.predict(X_te_out)   # outer test used HERE only
        f1  = macro_f1(y_te_out, preds_out)
        acc = accuracy(y_te_out, preds_out)

        outer_f1.append(f1)
        outer_acc.append(acc)

        if verbose:
            inner_str = "  ".join(f"{s:.3f}" for s in inner_scores_per_param)
            print(
                f"  [fold {outer_i+1:2d}/{outer_k}]"
                f"  F1={f1:.4f}  acc={acc:.4f}"
                f"  best={best_params}"
                f"  inner=[{inner_str}]"
            )

        # ── Persist immediately so partial runs are not lost ────────────
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            record = {
                "outer_fold"   : outer_i,
                "macro_f1"     : f1,
                "accuracy"     : acc,
                "best_params"  : best_params,
                "inner_scores" : [
                    {
                        "params"          : param_grid[k],
                        "mean_inner_f1"   : inner_scores_per_param[k],
                    }
                    for k in range(len(param_grid))
                ],
                "n_train" : int(len(train_out_idx)),
                "n_test"  : int(len(test_out_idx)),
            }
            path = os.path.join(
                save_dir, f"{model_name}_fold{outer_i:02d}.json"
            )
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(record, fh, indent=2)

    return {
        "outer_f1"       : outer_f1,
        "outer_acc"      : outer_acc,
        "best_params"    : best_params_list,
        "inner_f1_matrix": inner_f1_matrix,
    }


# =========================================================================== #
# Summary helper                                                               #
# =========================================================================== #

def print_summary(results, model_name):
    f1s  = np.array(results["outer_f1"])
    accs = np.array(results["outer_acc"])
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"  macro-F1 : {f1s.mean():.4f} ± {f1s.std():.4f}"
          f"  [{f1s.min():.4f} – {f1s.max():.4f}]")
    print(f"  accuracy : {accs.mean():.4f} ± {accs.std():.4f}")
    freq = Counter(str(p) for p in results["best_params"])
    print(f"  best-param frequency : {dict(freq)}")
    print(f"{'='*60}")
