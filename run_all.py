"""
Main experiment runner for FMA-Medium genre classification.

Usage
-----
  python run_all.py                  # full nested CV, all 3 models (~30 min)
  python run_all.py --model gnb      # GNB only
  python run_all.py --model lr       # Logistic Regression only
  python run_all.py --model resnet   # ResNet-tabular only
  python run_all.py --quick          # sanity check: 1000 samples, 3 outer folds
"""
import argparse
import time

import numpy as np

from gnb import GaussianNB
from lr_sgd import LogisticRegressionSGD
from resnet_tab import ResNetTabularClassifier
from nested_cv import nested_cv, print_summary

DATA_DIR   = r"C:\Users\Kling\fma\data"
RESULT_DIR = r"C:\Users\Kling\fma\results"

# =========================================================================== #
# Hyperparameter grids (from proposal)                                        #
# =========================================================================== #

GNB_GRID = [
    {"var_smoothing": 1e-11},
    {"var_smoothing": 1e-9},
    {"var_smoothing": 1e-7},
]

LR_GRID = [
    {"C": 0.01, "lr": 0.01, "epochs": 100, "batch_size": 256},
    {"C": 0.10, "lr": 0.01, "epochs": 100, "batch_size": 256},
    {"C": 1.00, "lr": 0.01, "epochs": 100, "batch_size": 256},
]

RESNET_GRID = [
    {"d": 64,  "n_blocks": 3, "dropout": 0.1, "lr": 1e-3,
     "epochs": 50, "patience": 10, "batch_size": 256},
    {"d": 128, "n_blocks": 3, "dropout": 0.1, "lr": 1e-3,
     "epochs": 50, "patience": 10, "batch_size": 256},
    {"d": 256, "n_blocks": 3, "dropout": 0.1, "lr": 1e-3,
     "epochs": 50, "patience": 10, "batch_size": 256},
]


# =========================================================================== #
# Runner                                                                       #
# =========================================================================== #

def run_model(X, y, name, factory, grid, outer_k, inner_k, seed):
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
        save_dir=RESULT_DIR,
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
        "--model", choices=["gnb", "lr", "resnet"], default=None,
        help="Run only one model (default: all three)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Sanity check mode: 1000 samples, outer=3, inner=2, minimal epochs"
    )
    args = parser.parse_args()

    # --- Load data --------------------------------------------------------
    print("Loading data...")
    X = np.load(f"{DATA_DIR}/X_medium.npy").astype(np.float32)
    y = np.load(f"{DATA_DIR}/y_medium.npy").astype(np.int32)
    print(f"  X={X.shape}  y={y.shape}  classes={len(np.unique(y))}")

    outer_k     = 10
    inner_k     = 3
    seed        = 42
    gnb_grid    = GNB_GRID
    lr_grid     = LR_GRID
    resnet_grid = RESNET_GRID

    # --- Quick / sanity-check mode ----------------------------------------
    if args.quick:
        print("\n[QUICK MODE]  1000 samples · outer=3 · inner=2 · minimal epochs")

        # Stratified subsample so all 16 classes appear
        rng = np.random.default_rng(0)
        classes = np.unique(y)
        sub_idx = []
        for c in classes:
            c_idx = np.where(y == c)[0]
            n     = max(4, int(round(len(c_idx) * 1000 / len(y))))
            n     = min(n, len(c_idx))
            sub_idx.extend(rng.choice(c_idx, size=n, replace=False).tolist())
        sub_idx = np.array(sub_idx)
        X, y    = X[sub_idx], y[sub_idx]

        outer_k     = 3
        inner_k     = 2
        lr_grid     = [
            {"C": c, "lr": 0.01, "epochs": 5, "batch_size": 64}
            for c in [0.01, 0.1, 1.0]
        ]
        resnet_grid = [
            {"d": d, "n_blocks": 2, "dropout": 0.1, "lr": 1e-3,
             "epochs": 5, "patience": 3, "batch_size": 64}
            for d in [32, 64]
        ]

        print(f"  Subsample size: {len(y)}  "
              f"classes present: {len(np.unique(y))}")

    # --- Model registry ---------------------------------------------------
    registry = {
        "gnb"    : (GaussianNB,              gnb_grid),
        "lr"     : (LogisticRegressionSGD,   lr_grid),
        "resnet" : (ResNetTabularClassifier, resnet_grid),
    }

    to_run = [args.model] if args.model else ["gnb", "lr", "resnet"]

    all_results = {}
    for key in to_run:
        factory_cls, grid = registry[key]
        all_results[key] = run_model(
            X, y,
            name=key.upper(),
            factory=factory_cls,
            grid=grid,
            outer_k=outer_k,
            inner_k=inner_k,
            seed=seed,
        )

    # --- Final comparison (only when multiple models ran) -----------------
    if len(to_run) > 1:
        print(f"\n{'='*60}")
        print("  FINAL COMPARISON")
        print(f"{'='*60}")
        for key in to_run:
            f1s = np.array(all_results[key]["outer_f1"])
            print(f"  {key.upper():10s}  macro-F1: {f1s.mean():.4f} ± {f1s.std():.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
