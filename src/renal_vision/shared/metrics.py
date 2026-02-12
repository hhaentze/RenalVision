from dataclasses import asdict, dataclass
from typing import Any, List, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
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
from sklearn.utils import resample

# --- Data Structures ---


@dataclass
class MetricStats:
    """Holds the statistical result of a metric analysis."""

    metric_name: str
    method: str  # 'bootstrap', 't-test'
    mean: float
    lower_ci: float
    upper_ci: float
    std_dev: float  # Useful for T-tests
    n_samples: int  # Number of rounds (bootstrap) or folds (t-test)

    def __str__(self):
        return (
            f"[{self.method.upper()}] {self.metric_name}: {self.mean:.3f} "
            f"(95% CI: {self.lower_ci:.3f} - {self.upper_ci:.3f})"
        )


@dataclass
class ScalarMetrics:
    """Holds standard scalar classification metrics."""

    accuracy: float
    f1_macro: float
    sensitivity: Optional[float] = None
    specificity: Optional[float] = None
    report: Optional[pd.DataFrame] = None

    def as_dict(self) -> dict:
        return asdict(self)


# --- Base Class ---


class _BaseEvaluator:
    """
    Shared core logic for all evaluators.
    Handles: Shape fixing, Binarization, and Metric Calculation.
    """

    def __init__(self, n_classes: int, class_names: Optional[List[str]] = None):
        self.n_classes = n_classes
        self.class_names = class_names if class_names else [f"Class {i}" for i in range(n_classes)]
        self.colors = sns.color_palette("husl", n_classes)

    def _ensure_matrix(self, y_proba: Any) -> np.ndarray:
        """Fixes the 'array of lists' issue common with Pandas."""
        if isinstance(y_proba, (pd.Series, list)) or (
            isinstance(y_proba, np.ndarray) and y_proba.ndim == 1
        ):
            try:
                return np.vstack(list(y_proba)).astype(float)
            except ValueError as e:
                raise ValueError(
                    "Could not stack probability lists. Ensure all rows have equal length."
                ) from e
        return np.array(y_proba)

    def _binarize(self, y: np.ndarray) -> np.ndarray:
        """Robust label binarization."""
        y_bin = label_binarize(y, classes=np.arange(self.n_classes))
        if self.n_classes == 2 and y_bin.shape[1] == 1:
            y_bin = np.hstack((1 - y_bin, y_bin))
        return y_bin

    def _get_label(self, i: int) -> str:
        return self.class_names[i] if i < len(self.class_names) else f"Class {i}"

    def _calc_score(
        self, y_true_bin: np.ndarray, y_proba: np.ndarray, metric: str, indices: List[int]
    ) -> float:
        """
        The mathematical core. Calculates the MEAN score for a subset of classes.
        """
        class_scores = []
        for i in indices:
            if metric == "auc":
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                class_scores.append(auc(fpr, tpr))
            elif metric == "ap":
                class_scores.append(average_precision_score(y_true_bin[:, i], y_proba[:, i]))
        return float(np.mean(class_scores))

    def _describe_dataset(self, y_true: np.ndarray, title: str = "Dataset Summary"):
        print(f"\n--- {title} ---")
        total = len(y_true)
        unique, counts = np.unique(y_true, return_counts=True)
        stats_dict = dict(zip(unique, counts))

        for i in range(self.n_classes):
            count = stats_dict.get(i, 0)
            pct = (count / total) * 100
            print(f"  - {self._get_label(i)}: {count} ({pct:.1f}%)")
        print("---------------------\n")


# --- Track A: Single Model Evaluator ---


class ModelEvaluator(_BaseEvaluator):
    """
    Evaluator for a single test set.
    """

    def __init__(
        self,
        y_true: Union[np.ndarray, List],
        y_proba: Union[np.ndarray, List],
        class_names: Optional[List[str]] = None,
        verbose: bool = True,
    ):
        # Fix Shapes
        self.y_true = np.array(y_true)
        y_proba_mat = self._ensure_matrix(y_proba)

        super().__init__(y_proba_mat.shape[1], class_names)

        self.y_proba = y_proba_mat
        self.y_true_bin = self._binarize(self.y_true)
        self.y_pred = np.argmax(self.y_proba, axis=1)

        if verbose:
            self._describe_dataset(self.y_true)

    def get_scalars(self) -> ScalarMetrics:
        """Compute Accuracy, F1, etc."""
        cm = confusion_matrix(self.y_true, self.y_pred)
        acc = accuracy_score(self.y_true, self.y_pred)
        f1 = f1_score(self.y_true, self.y_pred, average="weighted", zero_division=0)

        report = pd.DataFrame(
            classification_report(
                self.y_true,
                self.y_pred,
                target_names=self.class_names,
                output_dict=True,
                zero_division=0,
            )
        ).transpose()

        sens, spec = None, None
        if self.n_classes == 2:
            tn, fp, fn, tp = cm.ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return ScalarMetrics(
            accuracy=acc, f1_macro=f1, sensitivity=sens, specificity=spec, report=report
        )

    def check_ci_soundness(self, min_samples: int = 15, verbose: bool = True) -> bool:
        """
        Checks if minority class has enough sample for reliable CI.
        """
        unique, counts = np.unique(self.y_true, return_counts=True)

        if len(unique) < 2:
            print("Warning:  Only one class present in y_true. CI cannot be computed reliably.")
            return False  # Only 1 class present

        # Get the count of the smallest class
        min_count = np.min(counts)
        if min_count < min_samples:
            print(
                f"Warning:  Not enough samples for Class '{unique[np.argmin(counts)]}' (N={min_count}).\n",
                f"          Bootstrapping requires N >= {min_samples} to be reliable.",
            )
            return False

        return True

    def bootstrap_metric(
        self,
        metric: Literal["auc", "ap"] = "auc",
        target_classes: Optional[List[int]] = None,
        n_rounds: int = 1000,
        seed: int = 42,
    ) -> MetricStats:
        """
        Calculate 95% CI using Bootstrapping.
        """
        scores = []
        indices = target_classes if target_classes else list(range(self.n_classes))

        # 1. Base Score
        base_score = self._calc_score(self.y_true_bin, self.y_proba, metric, indices)
        method = "bootstrap" if self.check_ci_soundness() else "bootstrap (unreliable)"

        # 2. Bootstrap Loop
        n_samples = len(self.y_true)
        for _ in range(n_rounds):
            ix = resample(np.arange(n_samples), random_state=seed)
            score = self._calc_score(self.y_true_bin[ix], self.y_proba[ix], metric, indices)
            scores.append(score)

        # 3. Percentiles
        lower = float(np.percentile(scores, 2.5))
        upper = float(np.percentile(scores, 97.5))

        return MetricStats(
            metric_name=f"Mean {metric.upper()} (Classes {indices})",
            method=method,
            mean=base_score,
            lower_ci=lower,
            upper_ci=upper,
            std_dev=float(np.std(scores)),
            n_samples=n_rounds,
        )

    def plot_roc(self, figsize=(8, 6), output_path=None):
        plt.figure(figsize=figsize)
        classes_to_plot = [1] if self.n_classes == 2 else range(self.n_classes)
        for i in classes_to_plot:
            fpr, tpr, _ = roc_curve(self.y_true_bin[:, i], self.y_proba[:, i])
            score = auc(fpr, tpr)
            plt.plot(
                fpr,
                tpr,
                color=self.colors[i],
                lw=2,
                label=f"{self._get_label(i)} (AUC = {score:.2f})",
            )
        plt.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (One-vs-Rest)")
        plt.legend(loc="lower right")
        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
        else:
            plt.show()
        plt.close()

    def plot_pr(self, figsize=(8, 6), output_path=None):
        plt.figure(figsize=figsize)
        classes_to_plot = [1] if self.n_classes == 2 else range(self.n_classes)
        for i in classes_to_plot:
            p, r, _ = precision_recall_curve(self.y_true_bin[:, i], self.y_proba[:, i])
            score = average_precision_score(self.y_true_bin[:, i], self.y_proba[:, i])
            plt.plot(
                r,
                p,
                color=self.colors[i],
                lw=2,
                label=f"{self._get_label(i)} (AP = {score:.2f})",
            )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.grid(True, linestyle="--", alpha=0.5)
        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
        else:
            plt.show()
        plt.close()

    def plot_cm(self, figsize=(8, 6), output_path=None):
        cm = confusion_matrix(self.y_true, self.y_pred)
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
        else:
            plt.show()
        plt.close()


# --- Track B: Cross-Validation Evaluator ---


class CrossValidator(_BaseEvaluator):
    """
    Evaluator for multiple folds (List of Arrays).
    """

    def __init__(
        self,
        y_true_list: List[Union[np.ndarray, List]],
        y_proba_list: List[Union[np.ndarray, List]],
        class_names: Optional[List[str]] = None,
        verbose: bool = True,
    ):
        # Handle lists of lists -> list of arrays
        self.y_true_list = [np.array(y) for y in y_true_list]
        self.y_proba_list = [self._ensure_matrix(y) for y in y_proba_list]

        super().__init__(self.y_proba_list[0].shape[1], class_names)

        if verbose:
            print(f"--- Cross Validation Summary ({len(self.y_true_list)} folds) ---")
            total_samples = sum(len(y) for y in self.y_true_list)
            print(f"Total Samples (All Folds): {total_samples}")
            self._describe_dataset(self.y_true_list[0], title="Fold 1 Summary")

    def get_metric_stats(
        self,
        metric: Literal["auc", "ap"] = "auc",
        method: Literal["fold", "pooled"] = "fold",
        target_classes: Optional[List[int]] = None,
        n_rounds: int = 1000,
    ) -> MetricStats:
        """
        Calculate statistics for CV data.

        Args:
            metric: 'auc' or 'ap'
            method:
                'fold': Calculates T-Test based on N folds (Stability).
                'pooled': Concatenates all data and Bootstraps (Performance).
            target_classes: specific class indices to average.
        """
        indices = target_classes if target_classes else list(range(self.n_classes))

        if method == "fold":
            # --- Approach A: T-Test across folds ---
            fold_scores = []
            for y_true, y_proba in zip(self.y_true_list, self.y_proba_list):
                y_bin = self._binarize(y_true)
                score = self._calc_score(y_bin, y_proba, metric, indices)
                fold_scores.append(score)

            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores, ddof=1)  # Sample std
            n = len(fold_scores)

            # T-Interval
            # alpha=0.95, df=n-1, loc=mean, scale=SEM
            sem = std_score / np.sqrt(n)
            lower, upper = stats.t.interval(0.95, df=n - 1, loc=mean_score, scale=sem)

            return MetricStats(
                metric_name=f"Mean {metric.upper()} (Classes {indices})",
                method="t-test (fold-wise)",
                mean=float(mean_score),
                lower_ci=lower,
                upper_ci=upper,
                std_dev=float(std_score),
                n_samples=n,
            )

        elif method == "pooled":
            # --- Approach B: Bootstrap the concatenated arrays ---
            # 1. Concatenate
            y_true_all = np.concatenate(self.y_true_list)
            y_proba_all = np.vstack(self.y_proba_list)

            # 2. Delegate to ModelEvaluator logic
            # We create a temporary ModelEvaluator to run the bootstrap
            temp_eval = ModelEvaluator(y_true_all, y_proba_all, self.class_names, verbose=False)
            stats_obj = temp_eval.bootstrap_metric(
                metric, target_classes=indices, n_rounds=n_rounds
            )
            stats_obj.method = "bootstrap (pooled)"  # Update label
            return stats_obj

    def _get_interpolated_curve(self, class_idx, x_grid, curve_type="roc"):
        ys_interp = []
        scores = []

        for y_true, y_proba in zip(self.y_true_list, self.y_proba_list):
            y_bin = self._binarize(y_true)

            if curve_type == "roc":
                fpr, tpr, _ = roc_curve(y_bin[:, class_idx], y_proba[:, class_idx])
                interp_val = np.interp(x_grid, fpr, tpr)
                interp_val[0] = 0.0
                ys_interp.append(interp_val)
                scores.append(auc(fpr, tpr))

            elif curve_type == "pr":
                p, r, _ = precision_recall_curve(y_bin[:, class_idx], y_proba[:, class_idx])
                interp_val = np.interp(x_grid, r[::-1], p[::-1])
                ys_interp.append(interp_val)
                scores.append(average_precision_score(y_bin[:, class_idx], y_proba[:, class_idx]))

        mean_y = np.mean(ys_interp, axis=0)
        std_y = np.std(ys_interp, axis=0)

        if curve_type == "roc":
            mean_y[-1] = 1.0

        return mean_y, std_y, np.mean(scores), np.std(scores)

    def plot_aggregated_roc(self, figsize=(10, 8), output_path=None):
        plt.figure(figsize=figsize)
        mean_fpr = np.linspace(0, 1, 100)
        indices = [1] if self.n_classes == 2 else range(self.n_classes)

        for i in indices:
            mean_tpr, std_tpr, mean_auc, std_auc = self._get_interpolated_curve(i, mean_fpr, "roc")
            label = f"{self._get_label(i)} (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})"
            plt.plot(mean_fpr, mean_tpr, color=self.colors[i], lw=2, label=label)

            tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
            tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
            plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color=self.colors[i], alpha=0.15)

        plt.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7)
        plt.title("Cross-Validated ROC (Mean $\pm$ Std)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
        else:
            plt.show()
        plt.close()

    def plot_aggregated_pr(self, figsize=(10, 8), output_path=None):
        plt.figure(figsize=figsize)
        mean_recall = np.linspace(0, 1, 100)
        indices = [1] if self.n_classes == 2 else range(self.n_classes)

        for i in indices:
            mean_p, std_p, mean_ap, std_ap = self._get_interpolated_curve(i, mean_recall, "pr")
            label = f"{self._get_label(i)} (AP = {mean_ap:.2f} $\pm$ {std_ap:.2f})"
            plt.plot(mean_recall, mean_p, color=self.colors[i], lw=2, label=label)

            p_upper = np.minimum(mean_p + std_p, 1)
            p_lower = np.maximum(mean_p - std_p, 0)
            plt.fill_between(mean_recall, p_lower, p_upper, color=self.colors[i], alpha=0.15)

        plt.title("Cross-Validated Precision-Recall (Mean $\pm$ Std)")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="best")
        plt.grid(True, linestyle="--", alpha=0.5)
        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
        else:
            plt.show()
        plt.close()
