
"""
metrics.py — Hand-written classification metrics (no sklearn).

All functions operate on integer label arrays and are computed from raw
confusion-matrix counts so that the derivation is transparent and auditable.

Assignment constraint: third-party metric libraries (sklearn.metrics,
torchmetrics, etc.) are not permitted.  Plotting libraries (matplotlib,
seaborn) may still be used to visualise results computed here.

Metrics implemented
-------------------
accuracy              — overall fraction of correct predictions
macro_f1              — unweighted mean F1 across all classes
per_class_f1          — per-class F1 dict, for per-genre diagnostics
confusion_matrix      — (n_classes × n_classes) count matrix
minority_group_recall — combined recall for the 4 rarest FMA-Medium genres

Design note on macro vs weighted averaging
    Macro-averaging gives equal weight to every class regardless of size.
    On FMA-Medium this is intentional: a model that always predicts "Rock"
    would achieve ~36% weighted-F1 but near-zero macro-F1, making macro-F1
    a much more informative signal for the class-imbalance research question.
"""
import numpy as np


# =========================================================================== #
# Basic metrics                                                                #
# =========================================================================== #

def accuracy(y_true, y_pred):
    """
    Fraction of samples where the predicted label equals the true label.

    Parameters
    ----------
    y_true : array-like, shape (n,)
        Ground-truth integer labels.
    y_pred : array-like, shape (n,)
        Predicted integer labels.

    Returns
    -------
    float
        Value in [0, 1].  Higher is better.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def macro_f1(y_true, y_pred, zero_division=0.0):
    """
    Macro-averaged F1 score across all classes present in ``y_true``.

    Formula (per class c):
        precision_c = TP_c / (TP_c + FP_c)
        recall_c    = TP_c / (TP_c + FN_c)
        F1_c        = 2 * precision_c * recall_c / (precision_c + recall_c)
        macro-F1    = mean(F1_c  for all c in classes)

    Parameters
    ----------
    y_true : array-like, shape (n,)
        Ground-truth integer labels.
    y_pred : array-like, shape (n,)
        Predicted integer labels.
    zero_division : float, optional
        Value substituted for precision when TP + FP == 0 (i.e. a class is
        never predicted).  Default 0.0 matches sklearn's ``zero_division=0``.

    Returns
    -------
    float
        Value in [0, 1].  Higher is better.

    Notes
    -----
    Only classes present in ``y_true`` contribute to the average.  Classes
    that appear in ``y_pred`` but not in ``y_true`` are ignored (FP only,
    no ground-truth reference).  This matches sklearn's behaviour and is
    appropriate for our nested-CV setting where some minority classes may
    be absent from a small test fold.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(y_true)
    f1s = []
    for c in classes:
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else zero_division
        rec  = tp / (tp + fn) if (tp + fn) > 0 else zero_division

        # Harmonic mean of precision and recall; 0 if both are 0
        f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else zero_division
        f1s.append(f1)

    return float(np.mean(f1s))


def per_class_f1(y_true, y_pred, zero_division=0.0):
    """
    Compute F1 score for every class present in ``y_true`` individually.

    Useful for diagnosing which specific genres a model struggles with,
    especially minority classes (International, Easy Listening, Blues,
    Spoken) that are hidden inside the macro average.

    Parameters
    ----------
    y_true : array-like, shape (n,)
        Ground-truth integer labels.
    y_pred : array-like, shape (n,)
        Predicted integer labels.
    zero_division : float, optional
        Substituted when precision is undefined (TP + FP == 0).

    Returns
    -------
    dict
        ``{class_label (int): f1_score (float)}`` for every class in
        ``y_true``.  Values are in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(y_true)
    result = {}
    for c in classes:
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else zero_division
        rec  = tp / (tp + fn) if (tp + fn) > 0 else zero_division

        f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else zero_division
        result[int(c)] = float(f1)

    return result


# =========================================================================== #
# Confusion matrix                                                             #
# =========================================================================== #

def confusion_matrix(y_true, y_pred):
    """
    Build an (n_classes × n_classes) confusion matrix from scratch.

    Entry cm[i, j] counts the number of samples whose true class is
    ``classes[i]`` and whose predicted class is ``classes[j]``.  The
    diagonal therefore holds all correctly classified samples.

    Parameters
    ----------
    y_true : array-like, shape (n,)
        Ground-truth integer labels.
    y_pred : array-like, shape (n,)
        Predicted integer labels.  Labels not present in ``y_true`` are
        silently ignored (counted as off-diagonal but not added as a new
        row/column).

    Returns
    -------
    cm : np.ndarray, shape (n_classes, n_classes), dtype int
        Rows = true classes; columns = predicted classes.
    classes : np.ndarray, shape (n_classes,)
        Sorted unique class labels corresponding to the row/column order.

    Notes
    -----
    Off-diagonal entries reveal systematic confusions.  For FMA-Medium,
    acoustically similar genre pairs such as Folk/Country and
    Electronic/Experimental are expected to produce elevated off-diagonal
    counts — this constitutes one of the subsidiary research questions
    in the report.

    Example
    -------
    >>> cm, cls = confusion_matrix([0,0,1,1], [0,1,1,1])
    >>> cm
    array([[1, 1],
           [0, 2]])
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(y_true)
    n = len(classes)

    # Map raw label values to 0-based row/column indices
    label_to_idx = {int(c): i for i, c in enumerate(classes)}

    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        row = label_to_idx[int(t)]
        # A predicted label absent from y_true maps to -1 and is skipped;
        # this can happen in very small folds where some classes never appear.
        col = label_to_idx.get(int(p), -1)
        if col >= 0:
            cm[row, col] += 1

    return cm, classes


# =========================================================================== #
# Minority-group recall                                                        #
# =========================================================================== #

# Bottom-25% genres by sample count in FMA-Medium (17 000 tracks, 16 genres):
#   id= 9  International   18 samples   ← fewest
#   id= 3  Easy Listening  21 samples
#   id= 0  Blues           74 samples
#   id=15  Spoken         118 samples
# These four classes together account for 231 samples (~1.4% of the dataset).
MINORITY_IDS = np.array([9, 3, 0, 15])


def minority_group_recall(y_true, y_pred, minority_ids=MINORITY_IDS):
    """
    Combined recall for the bottom-25% (minority) genre group.

    Why group recall instead of per-class F1?
        International (18 samples) and Easy Listening (21 samples) yield
        fewer than 3 test samples per outer fold on average.  A single
        misprediction shifts per-class F1 by more than 30 percentage points,
        producing error bars so wide that no conclusion can be drawn.
        Merging the four minority classes into one group (~23 test samples
        per fold) provides a stable, interpretable signal.

    Formula:
        minority_group_recall = (correctly predicted minority samples)
                                / (total minority samples in y_true)

    Parameters
    ----------
    y_true : array-like, shape (n,)
        Ground-truth integer labels.
    y_pred : array-like, shape (n,)
        Predicted integer labels.
    minority_ids : array-like, optional
        Class labels to treat as the minority group.
        Default: ``MINORITY_IDS`` = [9, 3, 0, 15] (FMA-Medium bottom-25%).

    Returns
    -------
    float
        Value in [0, 1].  Returns 0.0 if no minority-class samples appear
        in ``y_true`` (e.g. a degenerate fold during quick-mode testing).

    Notes
    -----
    This metric is the primary indicator for the research question:
    "Can feature engineering improve minority-genre recognition?"
    It is reported as mean ± std across the 10 outer folds for each of
    the three models (GNB, LR, FT-Transformer).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Boolean mask: True for every sample that belongs to a minority class
    mask = np.isin(y_true, minority_ids)

    if mask.sum() == 0:
        # No minority samples in this fold — return 0 to avoid division by zero.
        # This only occurs during quick-mode tests with very small subsets.
        return 0.0

    return float(np.mean(y_pred[mask] == y_true[mask]))
