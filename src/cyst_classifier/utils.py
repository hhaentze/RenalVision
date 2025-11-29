"""Utility functions for evaluation and visualization."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, 
    accuracy_score, f1_score, recall_score, precision_score
)
from scipy import ndimage
from typing import Tuple, Dict


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, float]:
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
        except:
            auroc = None
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'auroc': auroc,
        'confusion_matrix': cm
    }


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, output_path: str = None):
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
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curve: Cyst vs Tumor Classification')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, output_path: str = None):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        output_path: Path to save plot (optional)
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Labels
    classes = ['Tumor', 'Cyst']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label',
           title='Confusion Matrix')
    
    # Text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14)
    
    fig.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def print_metrics_report(metrics: Dict[str, float]):
    """
    Print formatted metrics report.
    
    Args:
        metrics: Dictionary from compute_metrics()
    """
    print("\n" + "="*50)
    print("CLASSIFICATION METRICS")
    print("="*50)
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"F1 Score:    {metrics['f1']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    if metrics['auroc'] is not None:
        print(f"AUROC:       {metrics['auroc']:.4f}")
    print("="*50)
    
    # Print confusion matrix
    cm = metrics['confusion_matrix']
    print("\nConfusion Matrix:")
    print("                Predicted")
    print("              Tumor  Cyst")
    print(f"True Tumor    {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"     Cyst     {cm[1,0]:5d}  {cm[1,1]:5d}")
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
