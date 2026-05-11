"""
Main experiment runner for FMA-Medium genre classification.

Usage
-----
  python run_all.py                        # all 3 models, both feature sets A & B
  python run_all.py --model gnb            # GNB only, both feature sets
  python run_all.py --model gnb --features A   # GNB on original 518 features only
  python run_all.py --model gnb --features B   # GNB on full 639 features only
  python run_all.py --quick                # sanity check: 1000 samples, 3 outer folds

Feature sets
------------
  A : original 518 acoustic features from features.csv
  B : original 518 + 121 constructed = 639 features  (main experiment)
  Both A and B use X_medium_raw_derived.npy; per-fold Winsorization is applied
  inside nested_cv.py so test-fold distribution never leaks into preprocessing.

Runtime estimates (T4 GPU on Colab)
-------------------------------------
  GNB  A+B : ~10 min   (CPU-only)
  LR   A+B : ~20 min   (CPU-only)
  FT   A+B : ~120 min  (GPU strongly recommended; 60 training runs × ~60s each)
"""
import argparse
import os
import sys
import time

# Allow running as `python scripts/run_all.py` from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.models.gnb import GaussianNB
from src.models.lr_sgd import LogisticRegressionSGD
from src.models.ft_transformer import FTTransformerClassifier
from src.nested_cv import nested_cv, print_summary

_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.environ.get("FMA_DATA_DIR",   os.path.join(_ROOT, "data"))
RESULT_DIR = os.environ.get("FMA_RESULT_DIR", os.path.join(_ROOT, "results"))

# =========================================================================== #
# Hyperparameter grids                                                         #
# =========================================================================== #

# GNB: two separate grids for feature sets A and B.
#
# Why separate grids?
#   The assignment requires the middle hyperparameter value to win most often.
#   With a single grid {1e-11, 1e-9, 1e-7}:
#     Feature set A (518 features): 1e-9 (middle) wins 10/10  → OK
#     Feature set B (639 features): 1e-11 (lower boundary) wins 7/10 → violates requirement
#
#   The optimal var_smoothing shifts left on feature set B because 121 extra
#   constructed features (CV ratios, spectral slopes, Chroma entropy) add new
#   variance dimensions; the model needs less smoothing to fit the per-class
#   Gaussian distributions accurately on the larger feature space.
#
#   Solution: shift the entire grid one step left for feature set B so that
#   1e-11 becomes the middle value, satisfying the requirement for both sets.
#
#   GNB_GRID_A : {1e-11, 1e-9, 1e-7}   — 1e-9  wins 10/10  ✓ VERIFIED (full run, seed=42)
#   GNB_GRID_B : {1e-13, 1e-11, 1e-9}  — 1e-11 wins  7/10  ✓ VERIFIED (full run, seed=42)
#                                          1e-9  wins  3/10
#     Results (full 17000 samples, May 2026):
#       macro-F1       : 0.2960 ± 0.0099
#       accuracy       : 0.4504 ± 0.0076
#       minority_recall: 0.4201 ± 0.0689
GNB_GRID_A = [
    {"var_smoothing": 1e-11},
    {"var_smoothing": 1e-9},    # middle value — wins 10/10 for feature set A
    {"var_smoothing": 1e-7},
]

GNB_GRID_B = [
    {"var_smoothing": 1e-13},
    {"var_smoothing": 1e-11},   # middle value — wins 7/10 for feature set B
    {"var_smoothing": 1e-9},
]

LR_GRID = [
    {"C": 0.01, "lr": 0.01, "epochs": 100, "batch_size": 256},
    {"C": 0.10, "lr": 0.01, "epochs": 100, "batch_size": 256},  # middle
    {"C": 1.00, "lr": 0.01, "epochs": 100, "batch_size": 256},
]

# FT-Transformer: tuned hyperparameter is d_token (token embedding dimension).
# Dropout, n_heads, n_layers, lr are fixed to keep search space small and
# computation feasible on Colab (~60 training runs total per feature set).
# epochs=50 with patience=10 keeps each run under ~60s on a T4 GPU.
# d_token=128 is the middle value and is expected to win most often.
FT_GRID = [
    {"d_token":  64, "n_heads": 8, "n_layers": 3, "dropout": 0.1,
     "lr": 1e-4, "epochs": 50, "patience": 10, "batch_size": 256},
    {"d_token": 128, "n_heads": 8, "n_layers": 3, "dropout": 0.1,
     "lr": 1e-4, "epochs": 50, "patience": 10, "batch_size": 256},  # middle
    {"d_token": 256, "n_heads": 8, "n_layers": 3, "dropout": 0.1,
     "lr": 1e-4, "epochs": 50, "patience": 10, "batch_size": 256},
]


# =========================================================================== #
# Runner                                                                       #
# =========================================================================== #

def run_model(X, y, name, factory, grid, outer_k, inner_k, seed, save_dir):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  outer_k={outer_k}  inner_k={inner_k}"
          f"  param_combos={len(grid)}"
          f"  total_inner_runs={outer_k * inner_k * len(grid)}")
    print(f"{'='*60}")

    t0 = time.time()
    results = nested_cv(
        X, y,
        model_factory=factory,
        param_grid=grid,
        outer_k=outer_k,
        inner_k=inner_k,
        seed=seed,
        winsorize=True,
        save_dir=save_dir,
        model_name=name.lower(),
        verbose=True,
    )
    elapsed = time.time() - t0

    print_summary(results, name)
    print(f"  Wall time: {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    return results


# =========================================================================== #
# Entry point                                                                  #
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="Nested CV experiment — FMA-Medium genre classification"
    )
    parser.add_argument(
        "--model", choices=["gnb", "lr", "ft"], default=None,
        help="Run only one model (default: all three)"
    )
    parser.add_argument(
        "--features", choices=["A", "B", "both"], default="both",
        help=(
            "A = original 518 features, "
            "B = 518+121=639 features, "
            "both = run A then B (default)"
        )
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Sanity check mode: 1000 samples, outer=3, inner=2, minimal epochs"
    )
    args = parser.parse_args()

    # --- Load raw derived features (pre-Winsorization) --------------------
    raw_path = os.path.join(DATA_DIR, "X_medium_raw_derived.npy")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"\n[ERROR] {raw_path} not found.\n"
            "Run feature_engineering_explore.py first to generate it.\n"
            "It saves both X_medium.npy (for EDA) and "
            "X_medium_raw_derived.npy (for CV)."
        )
    X_raw = np.load(raw_path).astype(np.float32)   # shape (17000, 639)
    y     = np.load(os.path.join(DATA_DIR, "y_medium.npy")).astype(np.int32)
    print(f"Loaded X_raw={X_raw.shape}  y={y.shape}  classes={len(np.unique(y))}")

    # Build the list of (feature_set_label, X) pairs to run
    feature_sets = []
    if args.features in ("A", "both"):
        feature_sets.append(("A", X_raw[:, :518]))   # original 518 features only
    if args.features in ("B", "both"):
        feature_sets.append(("B", X_raw))             # all 639 features

    outer_k  = 10
    inner_k  = 3
    seed     = 42
    # GNB uses a per-feature-set grid (see GNB_GRID_A / GNB_GRID_B comments above)
    gnb_grids = {"A": GNB_GRID_A, "B": GNB_GRID_B}
    lr_grid  = LR_GRID
    ft_grid  = FT_GRID

    # --- Quick / sanity-check mode ----------------------------------------
    if args.quick:
        print("\n[QUICK MODE]  1000 samples · outer=3 · inner=2 · minimal epochs")

        rng = np.random.default_rng(0)
        classes = np.unique(y)
        sub_idx = []
        for c in classes:
            c_idx = np.where(y == c)[0]
            n     = max(4, int(round(len(c_idx) * 1000 / len(y))))
            n     = min(n, len(c_idx))
            sub_idx.extend(rng.choice(c_idx, size=n, replace=False).tolist())
        sub_idx = np.array(sub_idx)

        feature_sets = [(label, X[sub_idx]) for label, X in feature_sets]
        y_quick = y[sub_idx]

        outer_k = 3
        inner_k = 2
        lr_grid = [
            {"C": c, "lr": 0.01, "epochs": 5, "batch_size": 64}
            for c in [0.01, 0.1, 1.0]
        ]
        ft_grid = [
            {"d_token": d, "n_heads": 8, "n_layers": 2, "dropout": 0.1,
             "lr": 1e-4, "epochs": 5, "patience": 3, "batch_size": 64}
            for d in [64, 128]
        ]
        print(f"  Subsample size: {len(y_quick)}  "
              f"classes present: {len(np.unique(y_quick))}")
        y = y_quick

    # --- Model registry ---------------------------------------------------
    # GNB grid is resolved per feature set inside the loop below.
    registry = {
        "gnb" : (GaussianNB,              None),      # grid set per feat_label
        "lr"  : (LogisticRegressionSGD,   lr_grid),
        "ft"  : (FTTransformerClassifier, ft_grid),
    }

    to_run = [args.model] if args.model else ["gnb", "lr", "ft"]

    all_results = {}

    for feat_label, X_feat in feature_sets:
        print(f"\n{'#'*60}")
        print(f"  FEATURE SET {feat_label}  "
              f"({'original 518' if feat_label == 'A' else '518+121=639'} features)")
        print(f"{'#'*60}")

        for key in to_run:
            factory_cls, grid = registry[key]
            if key == "gnb":
                grid = gnb_grids[feat_label]
            run_name = f"{key.upper()}_{feat_label}"   # e.g. "GNB_A", "FT_B"
            # Organised save path: results/{model}/feat_{a|b}/
            feat_dir = os.path.join(RESULT_DIR, key, f"feat_{feat_label.lower()}")
            result = run_model(
                X_feat, y,
                name=run_name,
                factory=factory_cls,
                grid=grid,
                outer_k=outer_k,
                inner_k=inner_k,
                seed=seed,
                save_dir=feat_dir,
            )
            all_results[run_name] = result

    # --- Final A vs B comparison ------------------------------------------
    if args.features == "both" and not args.quick:
        print(f"\n{'='*60}")
        print("  A vs B COMPARISON  (original 518 vs +121 constructed features)")
        print(f"{'='*60}")
        header = f"  {'Model':<10}  {'Features':<10}  macro-F1"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for key in to_run:
            for feat_label in ("A", "B"):
                run_name = f"{key.upper()}_{feat_label}"
                if run_name in all_results:
                    f1s = np.array(all_results[run_name]["outer_f1"])
                    print(f"  {key.upper():<10}  {feat_label:<10}  "
                          f"{f1s.mean():.4f} ± {f1s.std():.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
