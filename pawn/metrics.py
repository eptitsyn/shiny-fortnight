from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, roc_curve

ZERO_DIVISION = 0


def _recalls_at(probs: np.ndarray, labels: np.ndarray, threshold: float) -> tuple[float, float]:
    preds = (probs >= threshold).astype(np.int64)
    machine_rec = recall_score(
        labels,
        preds,
        pos_label=1,
        zero_division=ZERO_DIVISION,  # pyright: ignore[reportArgumentType]
    )
    human_rec = recall_score(
        labels,
        preds,
        pos_label=0,
        zero_division=ZERO_DIVISION,  # pyright: ignore[reportArgumentType]
    )
    return float(machine_rec), float(human_rec)


def _metrics_at(
    probs: np.ndarray, labels: np.ndarray, threshold: float, suffix: str
) -> dict[str, float]:
    preds = (probs >= threshold).astype(np.int64)
    machine_rec, human_rec = _recalls_at(probs, labels, threshold)
    return {
        f"accuracy{suffix}": float(accuracy_score(labels, preds)),
        f"f1_macro{suffix}": float(
            f1_score(
                labels,
                preds,
                average="macro",
                zero_division=ZERO_DIVISION,  # pyright: ignore[reportArgumentType]
            )
        ),
        f"machine_rec{suffix}": machine_rec,
        f"human_rec{suffix}": human_rec,
        f"avg_rec{suffix}": 0.5 * (machine_rec + human_rec),
    }


def find_best_threshold(probs: np.ndarray, labels: np.ndarray, criterion: str = "youden") -> float:
    if len(np.unique(labels)) < 2:
        return 0.5

    if criterion == "youden":
        fpr, tpr, thresholds = roc_curve(labels, probs)
        valid = np.isfinite(thresholds)
        j = (tpr - fpr)[valid]
        return float(thresholds[valid][j.argmax()])

    if criterion == "avg_rec":
        _, _, thresholds = roc_curve(labels, probs)
        thresholds = thresholds[np.isfinite(thresholds)]
        scores = np.array([0.5 * sum(_recalls_at(probs, labels, t)) for t in thresholds])
        return float(thresholds[scores.argmax()])

    raise ValueError(f"unknown criterion: {criterion!r}")


def compute_metrics(eval_pred) -> dict[str, float]:
    logits = np.asarray(eval_pred.predictions).reshape(-1)
    labels = np.asarray(eval_pred.label_ids).reshape(-1)
    probs = expit(logits)

    out: dict[str, float] = {}
    out.update(_metrics_at(probs, labels, threshold=0.5, suffix=""))

    if len(np.unique(labels)) > 1:
        out["auroc"] = float(roc_auc_score(labels, probs))
        best_t = find_best_threshold(probs, labels, criterion="youden")
        out["best_threshold"] = best_t
        out.update(_metrics_at(probs, labels, threshold=best_t, suffix="_at_best"))

    out["pred_machine_frac"] = float((probs >= 0.5).mean())
    out["label_machine_frac"] = float(labels.mean())
    return out
