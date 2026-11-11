import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, log_loss)


def time_split(df, frac=0.75):
    """Chronological split. Drops one boundary row between dev and test."""
    split = int(len(df) * frac)
    dev = df.iloc[: split - 1]
    test = df.iloc[split:]
    return dev, test


def evaluate(y_true, y_pred, name, y_prob=None):
    """One results row: accuracy with a 95% interval, balanced accuracy, etc."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    acc = accuracy_score(y_true, y_pred)
    half = 1.96 * np.sqrt(acc * (1 - acc) / n)
    row = {
        "method": name,
        "n_test": n,
        "accuracy": round(acc, 4),
        "95% CI": f"{acc - half:.3f} to {acc + half:.3f}",
        "balanced_acc": round(balanced_accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_prob is not None:
        row["log_loss"] = round(log_loss(y_true, y_prob), 4)
    return row


def diff_ci(y_true, pred1, pred2, n_boot=5000, seed=0):
    """Paired bootstrap of accuracy(pred1) - accuracy(pred2) on the same days."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    c1 = (np.asarray(pred1) == y).astype(int)
    c2 = (np.asarray(pred2) == y).astype(int)
    n = len(y)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = c1[idx].mean(axis=1) - c2[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return round(c1.mean() - c2.mean(), 4), round(lo, 4), round(hi, 4)


def half_split_accuracy(y_true, preds):
    """Accuracy in the first and second half of the test period (Stage 6)."""
    y = np.asarray(y_true)
    h = len(y) // 2
    rows = []
    for name, p in preds.items():
        p = np.asarray(p)
        rows.append({"method": name,
                     "first_half_acc": round((p[:h] == y[:h]).mean(), 4),
                     "second_half_acc": round((p[h:] == y[h:]).mean(), 4)})
    return pd.DataFrame(rows)
