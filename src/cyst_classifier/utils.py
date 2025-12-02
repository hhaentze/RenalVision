"""Utility functions for evaluation and visualization."""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import ndimage
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Union[float, np.ndarray, Dict, str]]:
    """
    Compute classification metrics dynamically for binary or multi-class.
    """
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Standard metrics
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    metrics: Dict[str, Union[float, np.ndarray, Dict, str]] = {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "confusion_matrix": cm,
    }

    # Binary-specific metrics (Sensitivity/Specificity)
    unique_labels = np.unique(y_true)
    if len(unique_labels) == 2 and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Classification Report
    # We generate both dict (for programmatic access) and string (for printing)
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )
    metrics["report_dict"] = report_dict
    metrics["report_str"] = report_str

    # AUROC
    if y_proba is not None:
        try:
            if y_proba.shape[1] == 2:
                # Binary Case: assume class 1 is positive
                fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1], pos_label=1)
                metrics["auroc"] = float(auc(fpr, tpr))
            else:
                # Multi-class Case (Weighted OvR) - implicit handling in plots
                pass
        except Exception:
            pass

    return metrics


def apply_uncertainty_threshold(
    y_proba: np.ndarray, threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply uncertainty threshold to predictions.
    Uncertain predictions are labeled as -1.
    """
    max_proba = y_proba.max(axis=1)
    predictions = y_proba.argmax(axis=1)

    if threshold > 0.5:
        predictions = predictions.astype(int)
        predictions = np.where(max_proba < threshold, -1, predictions)

    return predictions, max_proba


def compute_metrics_with_uncertainty(
    y_true: np.ndarray,
    y_pred_with_unsure: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute metrics excluding uncertain predictions (-1).
    """
    certain_mask = y_pred_with_unsure != -1
    y_true_certain = y_true[certain_mask]
    y_pred_certain = y_pred_with_unsure[certain_mask]

    n_total = len(y_true)
    n_certain = np.sum(certain_mask)
    coverage = float(n_certain / n_total) if n_total > 0 else 0.0

    if n_certain > 0:
        certain_names = (
            [class_names[cl] for cl in np.unique(y_pred_certain)]
            if class_names is not None
            else None
        )
        base_metrics = compute_metrics(y_true_certain, y_pred_certain, None, certain_names)
    else:
        base_metrics = {
            "accuracy": 0.0,
            "f1": 0.0,
            "confusion_matrix": np.zeros((2, 2)),
            "report_str": "No certain predictions.",
        }

    # Build full confusion matrix including unsure column
    n_classes = y_proba.shape[1] if y_proba is not None else len(np.unique(y_true))
    cm_with_unsure = np.zeros((n_classes, n_classes + 1), dtype=int)

    for true_label in range(n_classes):
        for pred_label in range(n_classes):
            mask = (y_true == true_label) & (y_pred_with_unsure == pred_label)
            cm_with_unsure[true_label, pred_label] = np.sum(mask)
        # Unsure column
        mask = (y_true == true_label) & (y_pred_with_unsure == -1)
        cm_with_unsure[true_label, n_classes] = np.sum(mask)

    base_metrics.update(
        {
            "coverage": coverage,
            "n_certain": int(n_certain),
            "n_unsure": int(n_total - n_certain),
            "confusion_matrix_with_unsure": cm_with_unsure,
        }
    )

    return base_metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Optional[str] = None,
    title: str = "Confusion Matrix",
) -> None:
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))

    if cm.shape[1] > cm.shape[0]:
        x_labels = class_names + ["Unsure"]
    else:
        x_labels = class_names
    y_labels = class_names

    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticklabels(y_labels)

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
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
    Includes Micro-average, Macro-average, and per-class curves for N>2.
    """
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    # Ensure y_true_bin is N_samples x N_classes, required for multi-class metrics
    if n_classes == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

    # Explicitly type the dictionaries to allow int and str keys (mypy fix)
    RocDictType = Dict[Union[int, str], Union[float, np.ndarray]]

    fpr: RocDictType = {}
    tpr: RocDictType = {}
    roc_auc: RocDictType = {}

    plt.figure(figsize=(10, 8))

    if n_classes == 2:
        # --- Case 1: Binary Classification (N=2) ---
        # Plot only the standard ROC curve for the positive class (index 1)
        fpr[1], tpr[1], _ = roc_curve(y_true_bin[:, 1], y_proba[:, 1])
        roc_auc[1] = auc(fpr[1], tpr[1])

        label_name = class_names[1] if len(class_names) > 1 else "Positive Class"

        plt.plot(
            fpr[1],
            tpr[1],
            color="darkorange",
            lw=2,
            label=f"ROC curve ({label_name} AUC = {roc_auc[1]:0.2f})",
        )
        plt.title("Binary ROC Curve")
    else:
        # --- Case 2: Multi-class Classification (N>2) ---
        # Compute ROC curve and ROC area for each class
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot micro/macro
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label=f"micro-average ROC (area = {roc_auc['micro']:0.2f})",
            color="deeppink",
            linestyle=":",
            linewidth=4,
        )
        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label=f"macro-average ROC (area = {roc_auc['macro']:0.2f})",
            color="navy",
            linestyle=":",
            linewidth=4,
        )

        # Plot each class
        cmap = sns.color_palette("husl", n_classes)
        for i in range(n_classes):
            color = cmap[i]

            label_name = class_names[i] if i < len(class_names) else f"Class {i}"
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=2,
                label=f"{label_name} (AUC = {roc_auc[i]:0.2f})",
            )

        plt.title("Multi-class ROC Curve (One-vs-Rest)")

    # Common elements for both cases
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def print_metrics_report(metrics: Dict[str, Any], class_names: List[str]) -> None:
    """Print formatted metrics report including per-class details."""
    print("\n" + "=" * 60)
    print("CLASSIFICATION METRICS")
    print("=" * 60)
    print(f"Accuracy:    {metrics.get('accuracy', 0.0):.4f}")
    print(f"Weighted F1: {metrics.get('f1', 0.0):.4f}")

    if "sensitivity" in metrics:
        print(f"Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")

    if "coverage" in metrics:
        print(
            f"Coverage:    {metrics['coverage']:.4f} "
            f"({metrics['n_certain']} certain, {metrics['n_unsure']} unsure)"
        )

    print("-" * 60)
    print("Detailed Class Report:")
    # This report_str contains the per-class precision, recall, f1-score, and support
    print(metrics.get("report_str", ""))
    print("-" * 60)

    print("Confusion Matrix:")
    if "confusion_matrix" in metrics:
        print(metrics["confusion_matrix"])
    print("=" * 60)


def extract_single_lesion_component(mask: np.ndarray, component_id: int = 1) -> np.ndarray:
    labeled_mask, num_components = ndimage.label(mask > 0)
    return (labeled_mask == component_id).astype(np.uint8)


def get_all_lesion_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    return ndimage.label(mask > 0)
