from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    # numerically stable sigmoid
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _recalls_at(
    probs: np.ndarray, labels: np.ndarray, threshold: float
) -> tuple[float, float]:
    """Returns (machine_rec, human_rec) at a given threshold.
    Convention: y=1 means machine-generated, y=0 means human."""
    preds = (probs >= threshold).astype(np.int64)
    machine_rec = recall_score(labels, preds, pos_label=1, zero_division=0)
    human_rec = recall_score(labels, preds, pos_label=0, zero_division=0)
    return float(machine_rec), float(human_rec)


def _metrics_at(
    probs: np.ndarray, labels: np.ndarray, threshold: float, suffix: str
) -> dict[str, float]:
    """Compute the standard metric bundle at a specific threshold,
    suffixed so default-threshold and best-threshold metrics don't collide."""
    preds = (probs >= threshold).astype(np.int64)
    machine_rec, human_rec = _recalls_at(probs, labels, threshold)
    return {
        f"accuracy{suffix}": float(accuracy_score(labels, preds)),
        f"f1_macro{suffix}": float(
            f1_score(labels, preds, average="macro", zero_division=0)
        ),
        f"machine_rec{suffix}": machine_rec,
        f"human_rec{suffix}": human_rec,
        f"avg_rec{suffix}": 0.5 * (machine_rec + human_rec),
    }


def find_best_threshold(
    probs: np.ndarray, labels: np.ndarray, criterion: str = "youden"
) -> float:
    """Pick a threshold from the ROC curve.

    criterion:
      - 'youden': maximizes TPR - FPR (= max balanced accuracy on ROC).
                  Optimal under equal class costs and the standard MAGE setup.
      - 'avg_rec': sweeps thresholds and maximizes 0.5 * (machine_rec + human_rec).
                   Equivalent to 'youden' for binary classification but computed
                   directly from recalls — useful as a sanity check.
    """
    if len(np.unique(labels)) < 2:
        return 0.5  # degenerate eval batch — fall back to default

    if criterion == "youden":
        fpr, tpr, thresholds = roc_curve(labels, probs)
        # roc_curve prepends a +inf threshold; mask it so we don't pick it.
        valid = np.isfinite(thresholds)
        j = (tpr - fpr)[valid]
        return float(thresholds[valid][j.argmax()])

    if criterion == "avg_rec":
        # Sweep candidate thresholds; ROC's thresholds are exactly the points
        # where the predicted class changes for at least one sample.
        _, _, thresholds = roc_curve(labels, probs)
        thresholds = thresholds[np.isfinite(thresholds)]
        scores = np.array([
            0.5 * sum(_recalls_at(probs, labels, t)) for t in thresholds
        ])
        return float(thresholds[scores.argmax()])

    raise ValueError(f"unknown criterion: {criterion!r}")


def compute_metrics(eval_pred) -> dict[str, float]:
    """Trainer entrypoint.

    eval_pred.predictions: [B] raw logits (model returns logits as the only tensor)
    eval_pred.label_ids:   [B] in {0, 1}, 1 = machine-generated

    Reports two views:
      - <metric>: at the default 0.5 threshold (comparable to most baselines)
      - <metric>_at_best: at the threshold that maximizes Youden's J on this eval set

    The best threshold is itself logged as 'best_threshold' so you can reuse it
    when running .predict() on the test split or OOD testbeds.
    """
    logits = np.asarray(eval_pred.predictions).reshape(-1)
    labels = np.asarray(eval_pred.label_ids).reshape(-1)
    probs = sigmoid(logits)

    out: dict[str, float] = {}
    out.update(_metrics_at(probs, labels, threshold=0.5, suffix=""))

    if len(np.unique(labels)) > 1:
        out["auroc"] = float(roc_auc_score(labels, probs))

        best_t = find_best_threshold(probs, labels, criterion="youden")
        out["best_threshold"] = best_t
        out.update(_metrics_at(probs, labels, threshold=best_t, suffix="_at_best"))

    # Also log the prediction prior — useful for spotting class collapse.
    out["pred_machine_frac"] = float((probs >= 0.5).mean())
    out["label_machine_frac"] = float(labels.mean())

    return out
