"""
kfold.py — Hand-written Stratified K-Fold splitter.

Why stratified?
    FMA-Medium has severe class imbalance (Rock: 6103 samples vs
    International: 18 samples).  Plain K-Fold would let some folds contain
    zero samples of a rare class, making per-class metrics undefined and
    fold-to-fold comparisons unfair.  Stratified splitting guarantees every
    fold mirrors the overall class distribution as closely as possible.

Why from scratch?
    COMP90051 assignment requirement: sklearn.KFold / StratifiedKFold are
    not permitted.  All index arithmetic uses pure NumPy.

Design contract
    * Returns *global* indices into the original array — callers must not
      re-index or shuffle before passing to this function.
    * Reproducible: same (y, k, seed) always produces the same splits.
    * No data copying: only index arrays are created.
"""
import numpy as np


def stratified_kfold(y, k, seed=42):
    """
    Stratified K-Fold cross-validation splitter.

    Partitions sample indices into k folds such that each fold's class
    distribution matches the overall distribution as closely as possible.
    Within each class the samples are randomly shuffled before splitting,
    ensuring the folds are not biased by the original ordering of the data.

    Parameters
    ----------
    y : np.ndarray, shape (n,)
        Integer class labels for all n samples.  Labels do not need to be
        contiguous or zero-based.
    k : int
        Number of folds.  Must satisfy 2 <= k <= min class count.
    seed : int, optional
        Seed for the NumPy random generator.  Default 42 ensures
        reproducibility across runs.  Pass a different value for the inner
        CV loop so inner folds are not accidentally aligned with outer folds.

    Returns
    -------
    splits : list of k tuples (train_idx, test_idx)
        Each tuple holds two 1-D integer arrays of *global* indices.
        ``train_idx`` is the union of all folds except fold i;
        ``test_idx``  is fold i.
        Indices refer to positions in the original ``y`` array.

    Notes
    -----
    Uneven division:
        ``np.array_split`` distributes remainder samples to the *first* few
        folds — the same behaviour as sklearn's StratifiedKFold.  For
        International (18 samples, k=10) each fold gets either 1 or 2
        samples; this is expected and handled correctly downstream.

    Leakage safety:
        The returned indices are disjoint by construction.  The caller
        (``nested_cv``) additionally asserts disjointness as a runtime
        guard.

    Example
    -------
    >>> import numpy as np
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> splits = stratified_kfold(y, k=3, seed=0)
    >>> len(splits)
    3
    >>> train, test = splits[0]
    >>> set(train) & set(test)  # must be empty
    set()
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)

    # --- Phase 1: per-class stratified chunks ----------------------------
    # For each class independently: shuffle its sample indices, then cut
    # into k roughly equal pieces.  Shuffling inside each class prevents
    # any ordering artefact from the original dataset from propagating into
    # a specific fold.
    per_class_chunks = []
    for c in classes:
        idx_c = np.where(y == c)[0]        # global indices for class c
        idx_c = rng.permutation(idx_c)     # in-class shuffle
        chunks = np.array_split(idx_c, k)  # k pieces; uneven → first get extra
        per_class_chunks.append(chunks)

    # --- Phase 2: assemble folds -----------------------------------------
    # Fold i = union of chunk i from *every* class.
    # This guarantees that each fold contains samples from all classes
    # in the correct proportion.
    folds = []
    for i in range(k):
        fold_i = np.concatenate([per_class_chunks[c][i]
                                  for c in range(len(classes))])
        folds.append(fold_i)

    # --- Phase 3: build (train, test) pairs ------------------------------
    # Test  = fold i
    # Train = all other folds concatenated
    splits = []
    for i in range(k):
        test_idx  = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((train_idx, test_idx))

    return splits
