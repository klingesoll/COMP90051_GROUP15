"""
Hand-written classification metrics.
No sklearn.  All computed from confusion matrix counts.
"""
import numpy as np


def accuracy(y_true, y_pred):
    """Fraction of exactly correct predictions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def macro_f1(y_true, y_pred, zero_division=0.0):
    """
    Macro-averaged F1 score.

    Computed over every class present in y_true.
    If a class is never predicted (tp+fp == 0), its precision defaults
    to zero_division (mirrors sklearn's zero_division=0 behaviour).
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

        if prec + rec > 0:
            f1 = 2.0 * prec * rec / (prec + rec)
        else:
            f1 = zero_division
        f1s.append(f1)

    return float(np.mean(f1s))


def per_class_f1(y_true, y_pred, zero_division=0.0):
    """
    Returns dict {class_label (int): f1_score (float)} for every
    class present in y_true.  Useful for diagnosing minority-class
    performance in the imbalanced FMA dataset.
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

        if prec + rec > 0:
            f1 = 2.0 * prec * rec / (prec + rec)
        else:
            f1 = zero_division
        result[int(c)] = float(f1)

    return result
