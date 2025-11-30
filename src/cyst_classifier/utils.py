"""Utility functions for evaluation and visualization."""

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    roc_curve,
)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels (0=tumor, 1=cyst or 2=tumor, 3=cyst)
        y_pred: Predicted labels (0=tumor, 1=cyst)
        y_proba: Predicted probabilities (optional, for AUROC)

    Returns:
        Dictionary with metrics: accuracy, f1, sensitivity, specificity, auroc
    """
    # Convert labels if needed
    if np.min(y_true) > 1:
        y_true_binary = np.where(y_true == 2, 0, 1)
    else:
        y_true_binary = y_true

    # Confusion matrix: [[TN, FP], [FN, TP]]
    # For our case: 0=tumor (negative), 1=cyst (positive)
    cm = confusion_matrix(y_true_binary, y_pred)

    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Sensitivity (recall for cyst class)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Specificity (true negative rate for tumor class)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # F1 score (for cyst class, positive class)
    f1 = f1_score(y_true_binary, y_pred, pos_label=1, zero_division=0)

    # Accuracy
    accuracy = accuracy_score(y_true_binary, y_pred)

    # AUROC (if probabilities provided)
    auroc = None
    if y_proba is not None:
        try:
            fpr, tpr, _ = roc_curve(y_true_binary, y_proba[:, 1])
            auroc = auc(fpr, tpr)
        except Exception:
            auroc = None

    return {
        "accuracy": accuracy,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auroc": auroc,
        "confusion_matrix": cm,
    }


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, output_path: str | None = None):
    """
    Plot ROC curve.

    Args:
        y_true: Ground truth labels
        y_proba: Predicted probabilities (n_samples, 2)
        output_path: Path to save plot (optional)
    """
    # Convert labels if needed
    if np.min(y_true) > 1:
        y_true_binary = np.where(y_true == 2, 0, 1)
    else:
        y_true_binary = y_true

    fpr, tpr, thresholds = roc_curve(y_true_binary, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("ROC Curve: Cyst vs Tumor Classification")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"ROC curve saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_confusion_matrix(cm: np.ndarray, output_path: str | None = None):
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix
        output_path: Path to save plot (optional)
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    # Labels
    classes = ["Tumor", "Cyst"]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    # Text annotations
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
                fontsize=14,
            )

    fig.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to {output_path}")
    else:
        plt.show()

    plt.close()


def print_metrics_report(metrics: Dict[str, float | np.ndarray], exclude_unsure: bool = False):
    """
    Print formatted metrics report.

    Args:
        metrics: Dictionary from compute_metrics() or compute_metrics_with_uncertainty()
        exclude_unsure: If True, adds note about excluding unsure predictions
    """
    print("\n" + "=" * 50)
    print("CLASSIFICATION METRICS")
    if exclude_unsure:
        print("(excluding unsure predictions)")
    print("=" * 50)
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"F1 Score:    {metrics['f1']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    if metrics["auroc"] is not None:
        print(f"AUROC:       {metrics['auroc']:.4f}")

    if exclude_unsure and "coverage" in metrics:
        print(
            f"\nCoverage:    {metrics['coverage']:.4f} ({metrics['n_certain']}/{metrics['n_certain'] + metrics['n_unsure']} certain)"
        )
        print(f"Unsure:      {metrics['n_unsure']} predictions")

    print("=" * 50)

    # Print confusion matrix
    cm = metrics["confusion_matrix"]
    print("\nConfusion Matrix (certain predictions only):")
    print("                Predicted")
    print("              Tumor  Cyst")
    print(f"True Tumor    {cm[0, 0]:5d}  {cm[0, 1]:5d}")  # type: ignore[index]
    print(f"     Cyst     {cm[1, 0]:5d}  {cm[1, 1]:5d}")  # type: ignore[index]

    if exclude_unsure and "confusion_matrix_with_unsure" in metrics:
        cm_full = metrics["confusion_matrix_with_unsure"]
        print("\nFull Confusion Matrix (including unsure):")
        print("                     Predicted")
        print("              Tumor  Cyst  Unsure")
        print(f"True Tumor    {cm_full[0, 0]:5d}  {cm_full[0, 1]:5d}   {cm_full[0, 2]:5d}")  # type: ignore[index]
        print(f"     Cyst     {cm_full[1, 0]:5d}  {cm_full[1, 1]:5d}   {cm_full[1, 2]:5d}")  # type: ignore[index]

    print()


def extract_single_lesion_component(mask: np.ndarray, component_id: int = 1) -> np.ndarray:
    """
    Extract a specific connected component from a mask.

    Args:
        mask: Binary or multi-label mask
        component_id: Which component to extract (1-indexed)

    Returns:
        Binary mask with only the specified component
    """
    labeled_mask, num_components = ndimage.label(mask > 0)

    if num_components == 0:
        raise ValueError("No components found in mask")

    if component_id > num_components:
        raise ValueError(f"Component {component_id} requested but only {num_components} found")

    return (labeled_mask == component_id).astype(np.uint8)


def get_all_lesion_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Get labeled mask with all connected components.

    Args:
        mask: Binary or multi-label mask

    Returns:
        labeled_mask: Mask with each component labeled 1, 2, 3, ...
        num_components: Number of components found
    """
    return ndimage.label(mask > 0)


# ============================================================================
# Uncertainty Handling
# ============================================================================


def apply_uncertainty_threshold(
    y_proba: np.ndarray, threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply uncertainty threshold to predictions.

    Args:
        y_proba: Predicted probabilities (n_samples, 2) where [:, 0]=tumor, [:, 1]=cyst
        threshold: Confidence threshold (default: 0.5 = no uncertainty)

    Returns:
        predictions: Class predictions (0=tumor, 1=cyst, 2=unsure)
        max_proba: Maximum probability for each sample
    """
    max_proba = y_proba.max(axis=1)
    predictions = y_proba.argmax(axis=1)  # 0 or 1

    # Mark uncertain predictions as class 2
    if threshold > 0.5:
        predictions = np.where(max_proba < threshold, 2, predictions)

    return predictions, max_proba


def compute_metrics_with_uncertainty(
    y_true: np.ndarray, y_pred_with_unsure: np.ndarray, y_proba: np.ndarray | None = None
) -> Dict[str, float]:
    """
    Compute metrics excluding uncertain predictions.

    Args:
        y_true: Ground truth labels (0=tumor, 1=cyst or 2=tumor, 3=cyst)
        y_pred_with_unsure: Predictions (0=tumor, 1=cyst, 2=unsure)
        y_proba: Predicted probabilities (optional, for AUROC)

    Returns:
        Dictionary with metrics computed on certain predictions only
    """
    # Convert labels if needed
    if np.min(y_true) > 1:
        y_true_binary = np.where(y_true == 2, 0, 1)
    else:
        y_true_binary = y_true

    # Separate certain and uncertain predictions
    certain_mask = y_pred_with_unsure != 2
    y_true_certain = y_true_binary[certain_mask]
    y_pred_certain = y_pred_with_unsure[certain_mask]

    # Compute coverage
    n_total = len(y_true_binary)
    n_certain = np.sum(certain_mask)
    coverage = n_certain / n_total if n_total > 0 else 0.0

    # Compute metrics on certain predictions only
    if n_certain > 0:
        cm = confusion_matrix(y_true_certain, y_pred_certain)

        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = f1_score(y_true_certain, y_pred_certain, pos_label=1, zero_division=0)
        accuracy = accuracy_score(y_true_certain, y_pred_certain)

        # AUROC on certain predictions only
        auroc = None
        if y_proba is not None:
            try:
                y_proba_certain = y_proba[certain_mask]
                fpr, tpr, _ = roc_curve(y_true_certain, y_proba_certain[:, 1])
                auroc = auc(fpr, tpr)
            except Exception:
                auroc = None
    else:
        # All predictions uncertain
        accuracy = f1 = sensitivity = specificity = auroc = 0.0
        cm = np.zeros((2, 2))

    # Build full confusion matrix including unsure
    cm_with_unsure = np.zeros((3, 3), dtype=int)
    for true_label in [0, 1]:
        for pred_label in [0, 1, 2]:
            mask = (y_true_binary == true_label) & (y_pred_with_unsure == pred_label)
            cm_with_unsure[true_label, pred_label] = np.sum(mask)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auroc": auroc,
        "coverage": coverage,
        "confusion_matrix": cm,  # 2x2 for plotting
        "confusion_matrix_with_unsure": cm_with_unsure,  # 3x3 for text
        "n_certain": n_certain,
        "n_unsure": n_total - n_certain,
    }


def find_uncertainty_thresholds(
    y_true: np.ndarray, y_proba: np.ndarray, output_dir: str | None = None
) -> pd.DataFrame:
    """
    Analyze performance across different uncertainty thresholds.

    Args:
        y_true: Ground truth labels
        y_proba: Predicted probabilities (n_samples, 2)
        output_dir: Directory to save results (optional)

    Returns:
        DataFrame with threshold analysis results
    """
    thresholds = np.arange(0.50, 0.96, 0.05)
    results = []

    for threshold in thresholds:
        y_pred_with_unsure, _ = apply_uncertainty_threshold(y_proba, float(threshold))
        metrics = compute_metrics_with_uncertainty(y_true, y_pred_with_unsure, y_proba)

        results.append(
            {
                "Threshold": threshold,
                "Accuracy": metrics["accuracy"],
                "F1": metrics["f1"],
                "Sensitivity": metrics["sensitivity"],
                "Specificity": metrics["specificity"],
                "Coverage": metrics["coverage"],
                "Error_Rate": 1 - metrics["accuracy"] if metrics["accuracy"] > 0 else 0.0,
            }
        )

    df = pd.DataFrame(results)

    # Save and plot if output_dir provided
    if output_dir:
        output_path = Path(output_dir)

        # Save table
        df.to_csv(
            output_path / "threshold_analysis.txt", index=False, sep="\t", float_format="%.4f"
        )

        # Plot trade-off
        plot_threshold_tradeoff(df, str(output_path / "threshold_tradeoff.png"))

        print(f"\nThreshold analysis saved to {output_path}")

    return df


def plot_threshold_tradeoff(df: pd.DataFrame, output_path: str):
    """
    Plot threshold vs error rate and coverage.

    Args:
        df: DataFrame from find_uncertainty_thresholds()
        output_path: Path to save plot
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = "tab:red"
    ax1.set_xlabel("Uncertainty Threshold", fontsize=12)
    ax1.set_ylabel("Error Rate (on certain predictions)", color=color1, fontsize=12)
    ax1.plot(df["Threshold"], df["Error_Rate"], "o-", color=color1, linewidth=2, markersize=6)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "tab:blue"
    ax2.set_ylabel("Coverage (% certain)", color=color2, fontsize=12)
    ax2.plot(df["Threshold"], df["Coverage"], "s-", color=color2, linewidth=2, markersize=6)
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("Uncertainty Threshold Trade-off", fontsize=14, fontweight="bold")
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix_with_unsure(
    cm: np.ndarray, output_path: str | None = None, include_unsure: bool = True
):
    """
    Plot confusion matrix with optional unsure class.

    Args:
        cm: Confusion matrix (2x2 or 3x3)
        output_path: Path to save plot (optional)
        include_unsure: If True and cm is 3x3, show unsure class
    """
    fig, ax = plt.subplots(figsize=(8, 6) if include_unsure and cm.shape[0] == 3 else (6, 5))

    # Determine labels
    if include_unsure and cm.shape[0] == 3:
        classes = ["Tumor", "Cyst", "Unsure"]
        cmap = "Blues"
    else:
        classes = ["Tumor", "Cyst"]
        cm = cm[:2, :2]  # Use only 2x2 portion
        cmap = "Blues"

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # Labels
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Text annotations
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
                fontsize=14 if cm.shape[0] == 2 else 12,
            )

    fig.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to {output_path}")
    else:
        plt.show()

    plt.close()
