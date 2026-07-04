from __future__ import annotations

import numpy as np
import pandas as pd


def _as_1d_int_array(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {array.shape}")
    return array.astype(np.int64, copy=False)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    true = _as_1d_int_array("y_true", y_true)
    pred = _as_1d_int_array("y_pred", y_pred)
    if len(true) != len(pred):
        raise ValueError(f"y_true and y_pred must have the same length, got {len(true)} and {len(pred)}")

    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    if len(true) == 0:
        return matrix

    valid = (true >= 0) & (true < num_classes) & (pred >= 0) & (pred < num_classes)
    np.add.at(matrix, (true[valid], pred[valid]), 1)
    return matrix


def classification_report_frame(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    true = _as_1d_int_array("y_true", y_true)
    pred = _as_1d_int_array("y_pred", y_pred)
    if len(true) != len(pred):
        raise ValueError(f"y_true and y_pred must have the same length, got {len(true)} and {len(pred)}")
    if not class_names:
        raise ValueError("class_names must not be empty")

    rows: list[dict[str, float | int | str]] = []
    for class_id, class_name in enumerate(class_names):
        true_positive = int(np.sum((true == class_id) & (pred == class_id)))
        false_positive = int(np.sum((true != class_id) & (pred == class_id)))
        false_negative = int(np.sum((true == class_id) & (pred != class_id)))
        support = int(np.sum(true == class_id))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "Class": class_name,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
                "Support": support,
            }
        )

    report = pd.DataFrame(rows)
    total_support = int(report["Support"].sum())
    accuracy = float(np.mean(true == pred)) if len(true) else 0.0
    macro = report[["Precision", "Recall", "F1-Score"]].mean(numeric_only=True)
    weighted = (
        report[["Precision", "Recall", "F1-Score"]].multiply(report["Support"], axis=0).sum() / total_support
        if total_support
        else pd.Series({"Precision": 0.0, "Recall": 0.0, "F1-Score": 0.0})
    )

    report = pd.concat(
        [
            report,
            pd.DataFrame(
                [
                    {"Class": "accuracy", "Precision": accuracy, "Recall": accuracy, "F1-Score": accuracy, "Support": total_support},
                    {"Class": "macro avg", "Precision": macro["Precision"], "Recall": macro["Recall"], "F1-Score": macro["F1-Score"], "Support": total_support},
                    {"Class": "weighted avg", "Precision": weighted["Precision"], "Recall": weighted["Recall"], "F1-Score": weighted["F1-Score"], "Support": total_support},
                ]
            ),
        ],
        ignore_index=True,
    )
    return report


def one_vs_rest_roc_auc(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
) -> dict[str, dict[str, np.ndarray | float]]:
    true = _as_1d_int_array("y_true", y_true)
    probs = np.asarray(probabilities)
    if probs.ndim != 2:
        raise ValueError(f"probabilities must be a 2D array, got shape {probs.shape}")
    if len(true) != probs.shape[0]:
        raise ValueError(f"y_true and probabilities must align on sample axis, got {len(true)} and {probs.shape[0]}")
    if probs.shape[1] != len(class_names):
        raise ValueError(
            f"Number of probability columns must match class_names length, got {probs.shape[1]} and {len(class_names)}"
        )

    curves: dict[str, dict[str, np.ndarray | float]] = {}
    for class_id, class_name in enumerate(class_names):
        binary_true = (true == class_id).astype(np.int32)
        scores = probs[:, class_id]
        positives = int(binary_true.sum())
        negatives = int(len(binary_true) - positives)
        if positives == 0 or negatives == 0:
            curves[class_name] = {"fpr": np.array([0.0, 1.0]), "tpr": np.array([0.0, 1.0]), "auc": float("nan")}
            continue

        order = np.argsort(-scores)
        sorted_true = binary_true[order]
        true_positive = np.cumsum(sorted_true)
        false_positive = np.cumsum(1 - sorted_true)
        tpr = np.concatenate([[0.0], true_positive / positives, [1.0]])
        fpr = np.concatenate([[0.0], false_positive / negatives, [1.0]])
        auc = float(np.trapezoid(tpr, fpr))
        curves[class_name] = {"fpr": fpr, "tpr": tpr, "auc": auc}
    return curves
