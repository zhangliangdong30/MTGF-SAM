from __future__ import annotations

import numpy as np


def calculate_metrics(y_true, y_pred):
    y_true = (np.asarray(y_true) > 0).astype(np.uint8)
    y_pred = (np.asarray(y_pred) > 0).astype(np.uint8)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")

    eps = 1e-8
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    oa = (tp + tn) / max(tp + tn + fp + fn, 1)

    return {
        "OA": float(oa),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "IoU": float(iou),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }

