from __future__ import annotations

import numpy as np
import pandas as pd


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def classification_report_frame(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    num_classes = len(class_names)

    for class_id, class_name in enumerate(class_names):
        true_positive = int(np.sum((y_true == class_id) & (y_pred == class_id)))
        false_positive = int(np.sum((y_true != class_id) & (y_pred == class_id)))
        false_negative = int(np.sum((y_true == class_id) & (y_pred != class_id)))
        support = int(np.sum(y_true == class_id))
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
    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
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
    curves: dict[str, dict[str, np.ndarray | float]] = {}
    for class_id, class_name in enumerate(class_names):
        binary_true = (y_true == class_id).astype(np.int32)
        scores = probabilities[:, class_id]
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
        auc = float(np.trapz(tpr, fpr))
        curves[class_name] = {"fpr": fpr, "tpr": tpr, "auc": auc}
    return curves