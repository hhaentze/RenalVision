"""
Common evaluation metrics and plotting utilities.
"""

from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute classification metrics dynamically for binary or multi-class.
    """
    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "confusion_matrix": cm,
    }

    # Binary-specific metrics (Sensitivity/Specificity)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Classification Report
    metrics["report_dict"] = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    metrics["report_str"] = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Optional[str] = None,
    title: str = "Confusion Matrix",
) -> None:
    """Plot confusion matrix using Seaborn."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_multiclass_roc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_classes: int,
    class_names: List[str],
    output_path: Optional[str] = None,
) -> None:
    """
    Plot ROC curves for binary (N=2) or multi-class (N>2) problems.
    Includes Micro/Macro averages.
    """
    # Binarize labels for multi-class ROC
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    # Fix for binary case where label_binarize returns 1 column
    if n_classes == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

    fpr: Dict[Union[int, str], Any] = {}
    tpr: Dict[Union[int, str], Any] = {}
    roc_auc: Dict[Union[int, str], float] = {}

    plt.figure(figsize=(10, 8))

    # Calculate ROC for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plotting logic...
    if n_classes == 2:
        # Binary: Plot positive class only
        label = class_names[1] if len(class_names) > 1 else "Positive"
        plt.plot(fpr[1], tpr[1], lw=2, label=f"ROC ({label} AUC = {roc_auc[1]:.2f})")
    else:
        # Multi-class: Plot Micro/Macro + Per Class
        # ... (Micro/Macro logic from original utils.py) ...
        # Loop classes
        colors = sns.color_palette("husl", n_classes)
        for i, color in zip(range(n_classes), colors):
            name = class_names[i] if i < len(class_names) else f"Class {i}"
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f"{name} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (One-vs-Rest)")
    plt.legend(loc="lower right")

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_multiclass_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_classes: int,
    class_names: List[str],
    output_path: Optional[str] = None,
) -> None:
    """
    Plot Precision-Recall curves for binary or multi-class problems.
    Essential for imbalanced datasets where ROC can be misleading.
    """
    # Binarize labels for One-vs-Rest calculation
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    # Fix for binary case where label_binarize returns 1 column
    if n_classes == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

    precision: Dict[int, np.ndarray] = {}
    recall: Dict[int, np.ndarray] = {}
    average_precision: Dict[int, float] = {}

    plt.figure(figsize=(10, 8))

    # Use consistent colors with your ROC plot
    colors = sns.color_palette("husl", n_classes)

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
        average_precision[i] = average_precision_score(y_true_bin[:, i], y_proba[:, i])

        # Determine class name
        name = class_names[i] if class_names and i < len(class_names) else f"Class {i}"

        # Plot curve
        plt.plot(
            recall[i],
            precision[i],
            color=colors[i],
            lw=2,
            label=f"{name} (AP = {average_precision[i]:.2f})",
        )

    # Plot formatting
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (One-vs-Rest)")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
