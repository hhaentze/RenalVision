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


def plot_cv_roc(
    y_true_list: List[np.ndarray],
    y_proba_list: List[np.ndarray],
    n_classes: int,
    class_names: List[str],
    output_path: Optional[str] = None,
) -> None:
    """
    Plot Cross-Validated ROC curves with mean and variance (std dev).
    Requires a list of y_true and y_proba arrays (one per fold).
    """
    # Common x-axis for interpolation
    mean_fpr = np.linspace(0, 1, 100)

    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("husl", n_classes)

    # Determine which classes to plot
    # If binary, we usually only plot the positive class (index 1)
    classes_to_plot = [1] if n_classes == 2 else range(n_classes)

    for i in classes_to_plot:
        tprs = []
        aucs = []

        # Loop through each fold
        for y_true, y_proba in zip(y_true_list, y_proba_list):
            # Binarize labels for this fold
            y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

            # Fix for binary case (label_binarize returns 1 column for 2 classes)
            if n_classes == 2 and y_true_bin.shape[1] == 1:
                y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

            # Calculate ROC for this fold & class
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])

            # Interpolate TPR to map onto the common mean_fpr
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0  # Force start at 0
            tprs.append(interp_tpr)
            aucs.append(auc(fpr, tpr))

        # Calculate Means and Standard Deviations
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0  # Force end at 1
        std_tpr = np.std(tprs, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        # Labeling
        name = class_names[i] if i < len(class_names) else f"Class {i}"
        label = f"{name} (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})"
        color = colors[i]

        # Plot Mean Curve
        plt.plot(mean_fpr, mean_tpr, color=color, lw=2, alpha=0.8, label=label)

        # Plot Variance (Shaded Area)
        tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
        tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color=color, alpha=0.2)

    plt.plot([0, 1], [0, 1], "k--", lw=2, alpha=0.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Cross-Validated ROC Curve")
    plt.legend(loc="lower right")

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_cv_pr_curve(
    y_true_list: List[np.ndarray],
    y_proba_list: List[np.ndarray],
    n_classes: int,
    class_names: List[str],
    output_path: Optional[str] = None,
) -> None:
    """
    Plot Cross-Validated Precision-Recall curves with mean and variance.
    """
    # Common x-axis (Recall) for interpolation
    mean_recall = np.linspace(0, 1, 100)

    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("husl", n_classes)

    classes_to_plot = [1] if n_classes == 2 else range(n_classes)

    for i in classes_to_plot:
        precisions = []
        aps = []  # Average Precision Scores

        for y_true, y_proba in zip(y_true_list, y_proba_list):
            # Binarize
            y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
            if n_classes == 2 and y_true_bin.shape[1] == 1:
                y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

            # Calculate P-R curve
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])

            # Interpolation requires x-values (recall) to be increasing.
            # precision_recall_curve returns results sorted by threshold (descending recall).
            # We must flip them for np.interp.
            reversed_recall = recall[::-1]
            reversed_precision = precision[::-1]

            interp_precision = np.interp(mean_recall, reversed_recall, reversed_precision)
            precisions.append(interp_precision)

            aps.append(average_precision_score(y_true_bin[:, i], y_proba[:, i]))

        # Mean and Std
        mean_precision = np.mean(precisions, axis=0)
        std_precision = np.std(precisions, axis=0)
        mean_ap = np.mean(aps)
        std_ap = np.std(aps)

        # Plotting
        name = class_names[i] if i < len(class_names) else f"Class {i}"
        label = f"{name} (AP = {mean_ap:.2f} $\pm$ {std_ap:.2f})"
        color = colors[i]

        plt.plot(mean_recall, mean_precision, color=color, lw=2, alpha=0.8, label=label)

        # Shading
        prec_upper = np.minimum(mean_precision + std_precision, 1)
        prec_lower = np.maximum(mean_precision - std_precision, 0)
        plt.fill_between(mean_recall, prec_lower, prec_upper, color=color, alpha=0.2)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Cross-Validated Precision-Recall Curve")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.6)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
