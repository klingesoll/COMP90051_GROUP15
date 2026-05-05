"""
Hand-written Stratified K-Fold splitter.
No sklearn.  Indices are pure numpy.
"""
import numpy as np


def stratified_kfold(y, k, seed=42):
    """
    Stratified K-Fold cross-validation splitter.

    Each fold's test set contains roughly 1/k of every class,
    preserving the original class distribution as closely as possible.

    Parameters
    ----------
    y    : np.ndarray, shape (n,)  — integer class labels
    k    : int                      — number of folds
    seed : int                      — random seed (reproducible shuffles)

    Returns
    -------
    splits : list of k tuples (train_idx, test_idx)
        Both arrays contain *global* indices into y (not relative indices).

    Notes
    -----
    * Uses np.array_split, so uneven classes distribute extra samples to
      the first few folds (same behaviour as sklearn StratifiedKFold).
    * For very small classes (e.g. International with 18 samples, k=10),
      some test folds get 1 sample and some get 2 — that is expected and
      handled correctly.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)

    # For each class, shuffle indices and split into k chunks
    per_class_chunks = []
    for c in classes:
        idx_c = np.where(y == c)[0]
        idx_c = rng.permutation(idx_c)
        chunks = np.array_split(idx_c, k)   # list of k arrays
        per_class_chunks.append(chunks)

    # Fold i = union of chunk i from every class
    folds = []
    for i in range(k):
        fold_i = np.concatenate([per_class_chunks[c][i]
                                  for c in range(len(classes))])
        folds.append(fold_i)

    # Build (train_idx, test_idx) pairs
    splits = []
    for i in range(k):
        test_idx  = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((train_idx, test_idx))

    return splits
