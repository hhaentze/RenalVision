import warnings
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils import resample


# --- Data Structure ---
@dataclass
class MetricResult:
    """
    Unified container for both for results (Bootstrap or CV).
    """

    scores: np.ndarray  # Shape: (N_rounds,)
    curves: np.ndarray | None = (
        None  # Shape: (N_rounds, grid_points) Content: Interpolated Y-values
    )
    x_grid: np.ndarray | None = None  # Shape: (grid_points,) Content: Interpolated X-values
    use_t_dist: bool = False  # CI calculation method

    @property
    def mean_score(self) -> float:
        return float(np.nanmean(self.scores))

    @property
    def ci_score(self) -> tuple[float, float]:
        # Filter NaNs
        valid_scores = self.scores[~np.isnan(self.scores)]
        n = len(valid_scores)
        if n < 2:
            return (self.mean_score, self.mean_score)

        if self.use_t_dist:
            # --- CV Logic: Student's t-distribution ---
            mean = np.mean(valid_scores)
            se = stats.sem(valid_scores)
            return stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
        else:
            # --- Bootstrap Logic: Percentiles ---
            return (
                float(np.percentile(valid_scores, 2.5)),
                float(np.percentile(valid_scores, 97.5)),
            )

    @property
    def mean_curve(self) -> np.ndarray:
        if self.curves is not None:
            return np.nanmean(self.curves, axis=0)
        return np.array([])

    @property
    def ci_band(self) -> tuple[np.ndarray, np.ndarray] | None:
        if len(self.scores) < 2 or self.curves is None:
            return None

        # NaN-aware throughout: a fold missing a class contributes an all-NaN row.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if self.use_t_dist:
                # --- CV Logic: Student's t-distribution (matches ci_score) ---
                n_valid = np.sum(~np.isnan(self.curves), axis=0)
                mean = np.nanmean(self.curves, axis=0)
                sd = np.nanstd(self.curves, axis=0, ddof=1)
                se = np.divide(
                    sd,
                    np.sqrt(np.maximum(n_valid, 1)),
                    out=np.full_like(mean, np.nan),
                    where=n_valid > 1,
                )
                t_crit = stats.t.ppf(0.975, df=np.maximum(n_valid - 1, 1))
                low, high = mean - t_crit * se, mean + t_crit * se
            else:
                # --- Bootstrap Logic: Percentiles ---
                low = np.nanpercentile(self.curves, 2.5, axis=0)
                high = np.nanpercentile(self.curves, 97.5, axis=0)

        # Both ROC and PR are bounded in [0, 1]
        return (np.clip(low, 0.0, 1.0), np.clip(high, 0.0, 1.0))


# --- The Evaluator ---
class ModelEvaluator:
    def __init__(
        self,
        y_true: np.ndarray | list | list[np.ndarray],
        y_proba: np.ndarray | list | list[np.ndarray],
        class_names: list[str] | None = None,
    ):
        """
        Smart Init:
        - If input is list of arrays -> Activates Cross-Validation Mode.
        - If input is single array/list -> Activates Bootstrap Mode.
        """
        # 1. Smart Input Detection
        self.is_cv = False
        self.folds_y_true = None
        self.folds_y_proba = None

        # Check if input is likely a list of arrays (CV folds)
        if (
            isinstance(y_true, list)
            and len(y_true) > 0
            and hasattr(y_true[0], "__len__")
            and not isinstance(y_true[0], str)
        ):
            # --- CV MODE DETECTED ---
            print(f"-> Cross-Validation Mode detected ({len(y_true)} folds).")
            self.is_cv = True

            # Store folds for distribution calculation
            self.folds_y_true = [np.array(y) for y in y_true]
            self.folds_y_proba = [self._ensure_matrix(y) for y in y_proba]

            # Flatten data for "Global/Pooled" scalar metrics
            self.y_true = np.concatenate(self.folds_y_true)
            self.y_proba = np.vstack(self.folds_y_proba)

        else:
            # --- BOOTSTRAP MODE ---
            self.y_true = np.array(y_true)
            self.y_proba = self._ensure_matrix(y_proba)

        # 2. Setup Meta-data
        self.n_classes = self.y_proba.shape[1]
        self.class_names = (
            class_names if class_names else [f"Class {i}" for i in range(self.n_classes)]
        )

        present_classes = np.unique(self.y_true)
        self.active_classes = [i for i in range(self.n_classes) if i in present_classes]

        # 3. Global Binarization (for pooled stats)
        self.y_true_bin_pooled = label_binarize(self.y_true, classes=range(self.n_classes))
        if self.n_classes == 2 and self.y_true_bin_pooled.shape[1] == 1:
            self.y_true_bin_pooled = np.hstack((1 - self.y_true_bin_pooled, self.y_true_bin_pooled))

        # 3b. Per-class prevalence == chance baseline for the PR curve.
        #     Pooled across folds in CV mode (per-fold prevalences differ only marginally).
        self.n_pos = {i: int(self.y_true_bin_pooled[:, i].sum()) for i in self.active_classes}
        self.prevalence = {
            i: float(self.y_true_bin_pooled[:, i].mean()) for i in self.active_classes
        }

        self._cache: dict[tuple[str, bool], dict[int | str, MetricResult]] = {}

        # 4. Plotting Defaults
        self.plot_style = {
            "show_legend": True,
            "show_grid": True,
            "alpha_band": 0.15,
            "linewidth": 2,
            "colors": sns.color_palette("husl", self.n_classes),
            "dpi": 300,
            "pad_inches": 0.1,
        }

    # --- Public Interface ---

    def get_scalars(self) -> pd.DataFrame:
        """Returns basic scalar metrics (Accuracy, F1, etc) as a clean DataFrame."""
        y_pred = np.argmax(self.y_proba, axis=1)

        report_dict = classification_report(
            self.y_true,
            y_pred,
            target_names=[self.class_names[i] for i in self.active_classes],
            labels=self.active_classes,
            output_dict=True,
            zero_division=0,
        )

        df = pd.DataFrame(report_dict).transpose()
        return df

    def get_auc(self, with_ci: bool = True) -> pd.DataFrame:
        """Returns a DataFrame with AUC scores (and CIs if requested)."""
        return self._format_metric_table("roc", with_ci)

    def get_ap(self, with_ci: bool = True) -> pd.DataFrame:
        """
        Returns a DataFrame with Average Precision scores (and CIs if requested).
        Includes the chance baseline (prevalence), the lift over it, and n_pos --
        AP is not interpretable without them.
        """
        return self._format_metric_table("pr", with_ci)

    def plot_roc(
        self, show_ci: bool = False, figsize=(8, 6), output_path: str | None = None, **kwargs
    ):
        results = self._get_data("roc", show_ci)
        self._plot_generic(
            results,
            "roc",
            "ROC Curve",
            "False Positive Rate",
            "True Positive Rate",
            show_ci,
            figsize,
            output_path,
            **kwargs,
        )

    def plot_pr(
        self, show_ci: bool = False, figsize=(8, 6), output_path: str | None = None, **kwargs
    ):
        results = self._get_data("pr", show_ci)
        self._plot_generic(
            results,
            "pr",
            "Precision-Recall Curve",
            "Recall",
            "Precision",
            show_ci,
            figsize,
            output_path,
            **kwargs,
        )

    def plot_pr_grid(
        self,
        show_ci: bool = False,
        ncols: int = 3,
        figsize=None,
        sharey: bool = True,
        output_path: str | None = None,
        **kwargs,
    ):
        """
        One PR panel per class, each with its own prevalence baseline.

        Unlike ROC (chance is 0.5 everywhere), the PR chance level is class-specific,
        so overlaying all classes in one axis gives them incomparable baselines.
        """
        results = self._get_data("pr", show_ci)

        valid_keys = set(self.plot_style.keys())
        for key in kwargs:
            if key not in valid_keys:
                raise ValueError(
                    f"Invalid style parameter: '{key}'. Available options: {valid_keys}"
                )
        style = {**self.plot_style, **kwargs}
        colors = style["colors"]

        n = len(self.active_classes)
        ncols = max(1, min(ncols, n))
        nrows = int(np.ceil(n / ncols))
        figsize = figsize or (4.0 * ncols, 3.6 * nrows)

        fig, axes = plt.subplots(
            nrows, ncols, figsize=figsize, sharex=True, sharey=sharey, squeeze=False
        )
        flat = axes.ravel()

        for ax, i in zip(flat, self.active_classes):
            res = results[i]
            prev = self.prevalence[i]

            if res.x_grid is not None:
                ax.plot(res.x_grid, res.mean_curve, color=colors[i], lw=style["linewidth"])
                if show_ci and res.ci_band is not None:
                    ax.fill_between(
                        res.x_grid, *res.ci_band, color=colors[i], alpha=style["alpha_band"]
                    )

            ax.axhline(prev, color="grey", ls="--", lw=1, zorder=0)

            lift = res.mean_score / prev if prev > 0 else np.nan
            sub = f"AP={res.mean_score:.2f}"
            if show_ci:
                low, high = res.ci_score
                sub += f" [{low:.2f}-{high:.2f}]"
            sub += f"\nchance={prev:.2f} ({lift:.1f}x), n₊={self.n_pos[i]}"

            ax.set_title(f"{self.class_names[i]}\n{sub}", fontsize="small")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            if style["show_grid"]:
                ax.grid(True, linestyle="--", alpha=0.5)

        for ax in flat[n:]:
            ax.set_visible(False)

        fig.supxlabel("Recall")
        fig.supylabel("Precision")
        fig.tight_layout()

        if output_path:
            fig.savefig(
                output_path, bbox_inches="tight", dpi=style["dpi"], pad_inches=style["pad_inches"]
            )
        else:
            plt.show()
        plt.close(fig)

    def plot_cm(
        self,
        normalize: bool = False,
        title: str = "Confusion Matrix",
        figsize=(8, 6),
        output_path: str | None = None,
    ):
        """Plots the global (pooled) confusion matrix."""

        style = self.plot_style
        y_pred = np.argmax(self.y_proba, axis=1)
        cm = confusion_matrix(self.y_true, y_pred, labels=self.active_classes)

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap="Blues",
            xticklabels=[self.class_names[i] for i in self.active_classes],
            yticklabels=[self.class_names[i] for i in self.active_classes],
        )
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title(title)

        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=style["dpi"])
        else:
            plt.show()
        plt.close()

    # --- Internal Logic ---

    def _format_metric_table(self, metric_type: str, with_ci: bool) -> pd.DataFrame:
        """Helper to format the internal MetricResult objects into a user-friendly DataFrame."""
        results = self._get_data(metric_type, with_ci)
        rows = []

        for cls_id in self.active_classes:
            res = results[cls_id]
            name = self.class_names[cls_id]
            rows.append(self._make_row(name, res, with_ci, metric_type, cls_id))

        if "macro" in results:
            rows.append(self._make_row("MACRO AVERAGE", results["macro"], with_ci, metric_type))

        df = pd.DataFrame(rows)
        if not df.empty:
            df.set_index("Class", inplace=True)
        return df

    def _make_row(
        self,
        name: str,
        res: MetricResult,
        with_ci: bool,
        metric_type: str = "roc",
        cls_id: int | None = None,
    ) -> dict:
        row = {"Class": name, "Mean": round(res.mean_score, 3)}

        # AP is only interpretable relative to prevalence -> carry the baseline in the table.
        if metric_type == "pr":
            if cls_id is not None:
                prev = self.prevalence[cls_id]
                row["Chance"] = round(prev, 3)
                row["Lift"] = round(res.mean_score / prev, 2) if prev > 0 else np.nan
                row["n_pos"] = self.n_pos[cls_id]
            else:
                row["Chance"] = np.nan
                row["Lift"] = np.nan
                row["n_pos"] = np.nan

        if with_ci:
            low, high = res.ci_score
            row["95% CI Lower"] = round(low, 3)
            row["95% CI Upper"] = round(high, 3)
            row["CI Range"] = f"[{low:.2f} - {high:.2f}]"
        return row

    def _get_data(self, metric: str, show_ci: bool) -> dict[int | str, MetricResult]:
        """Lazy loader: Returns cached data or triggers calculation."""
        cache_key = (metric, show_ci)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # If CV, n_rounds is ignored (we use len(folds)).
        n_rounds = 1000 if show_ci else 1

        data = self._compute_distribution(metric, n_rounds)
        self._cache[cache_key] = data
        return data

    def _compute_distribution(self, metric: str, n_rounds: int) -> dict[int | str, MetricResult]:
        """
        Calculates Class-wise AND Macro distributions.
        - If self.is_cv is True: Iterates through Folds.
        - If self.is_cv is False: Iterates through Bootstrap Resamples.
        """
        results: dict[int, dict[str, list[Any]]] = {
            i: {"scores": [], "curves": []} for i in self.active_classes
        }
        macro_scores = []
        x_grid = np.linspace(0, 1, 100)

        # --- 1. DETERMINE ITERATOR ---
        if self.is_cv:
            # Iterate through the pre-stored folds
            iterator = zip(self.folds_y_true, self.folds_y_proba)  # type: ignore
        else:
            # Bootstrap loop
            n_samples = len(self.y_true)
            indices = np.arange(n_samples)

            # Generator that yields resampled data
            def bootstrap_gen():
                for _ in range(n_rounds):
                    # No resample if n_rounds=1 (Single Run)
                    ix = (
                        resample(indices, replace=True, n_samples=n_samples, stratify=self.y_true)
                        if n_rounds > 1
                        else indices
                    )
                    yield self.y_true_bin_pooled[ix], self.y_proba[ix]  # Use pooled binarized

            iterator = bootstrap_gen()

        # --- 2. MAIN LOOP ---
        for loop_idx, (y_true_curr, y_prob_curr) in enumerate(iterator):
            # If in CV mode, y_true_curr is raw labels (not binarized yet)
            # We must binarize per-fold because a fold might miss a class
            if self.is_cv:
                y_bin_curr = label_binarize(y_true_curr, classes=range(self.n_classes))
                if self.n_classes == 2 and y_bin_curr.shape[1] == 1:
                    y_bin_curr = np.hstack((1 - y_bin_curr, y_bin_curr))
            else:
                # Bootstrap generator already yields binarized data from pooled
                y_bin_curr = y_true_curr

            round_class_scores: list[float] = []

            for i in self.active_classes:
                # --- CHECK FOR MISSING CLASS IN FOLD/SAMPLE ---
                if np.sum(y_bin_curr[:, i]) == 0:
                    if self.is_cv:
                        warnings.warn(f"Class {i} missing in Fold {loop_idx}. Returning NaN.")
                    score = np.nan
                    # Curve is all NaNs
                    y_interp = np.full_like(x_grid, np.nan)
                else:
                    # Valid Calculation
                    if metric == "roc":
                        fpr, tpr, _ = roc_curve(y_bin_curr[:, i], y_prob_curr[:, i])
                        score = auc(fpr, tpr)
                        y_interp = self._interpolate(x_grid, fpr, tpr)
                        y_interp[0], y_interp[-1] = 0.0, 1.0
                    else:
                        p, r, _ = precision_recall_curve(y_bin_curr[:, i], y_prob_curr[:, i])
                        score = average_precision_score(y_bin_curr[:, i], y_prob_curr[:, i])
                        # PR needs step-wise, not linear, interpolation (see _interpolate_pr)
                        y_interp = self._interpolate_pr(x_grid, r, p)

                results[i]["scores"].append(float(score))
                results[i]["curves"].append(y_interp)

                # Only add to macro if not NaN
                if not np.isnan(score):
                    round_class_scores.append(float(score))

            # Macro for this round (mean of valid class scores)
            if round_class_scores:
                macro_scores.append(np.mean(round_class_scores))
            else:
                macro_scores.append(np.nan)  # type: ignore

        # --- 3. PACKAGE RESULTS ---
        final: dict[int | str, MetricResult] = {}
        for i in self.active_classes:
            final[i] = MetricResult(
                scores=np.array(results[i]["scores"]),
                curves=np.array(results[i]["curves"]),
                x_grid=x_grid,
                use_t_dist=self.is_cv,  # <--- Set Flag based on mode
            )

        final["macro"] = MetricResult(
            scores=np.array(macro_scores), curves=None, x_grid=None, use_t_dist=self.is_cv
        )

        return final

    def _plot_generic(
        self,
        results: dict[int | str, MetricResult],
        metric: str,
        title,
        xlabel,
        ylabel,
        show_ci,
        figsize,
        output_path: str | None = None,
        **kwargs,
    ):
        # Validate kwargs against defined style
        valid_keys = set(self.plot_style.keys())
        for key in kwargs:
            if key not in valid_keys:
                raise ValueError(
                    f"Invalid style parameter: '{key}'. Available options: {valid_keys}"
                )

        style = {**self.plot_style, **kwargs}
        colors = style["colors"]
        plt.figure(figsize=figsize)
        for i in self.active_classes:
            res = results[i]
            label = f"{self.class_names[i]} ({res.mean_score:.2f}"
            if metric == "pr":
                # Baseline differs per class -> state it wherever the AP is stated.
                label += f", chance {self.prevalence[i]:.2f}"
            label += ")"
            if show_ci:
                label += f" [{res.ci_score[0]:.2f}-{res.ci_score[1]:.2f}]"

            if res.x_grid is not None:
                plt.plot(
                    res.x_grid, res.mean_curve, color=colors[i], lw=style["linewidth"], label=label
                )
                if show_ci and res.ci_band:
                    plt.fill_between(
                        res.x_grid, *res.ci_band, color=colors[i], alpha=style["alpha_band"]
                    )

        if "macro" in results:
            # We add the macro score to the legend, but don't plot a curve/band to avoid clutter
            plt.plot([], [], " ", label=f"MACRO AVG ({results['macro'].mean_score:.2f})")

        # --- CHANCE BASELINE ---
        if metric == "roc":
            plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        elif metric == "pr":
            # One horizontal line per class at its prevalence. Colour-matched to its curve;
            # use plot_pr_grid() if these collide.
            for i in self.active_classes:
                plt.axhline(self.prevalence[i], color=colors[i], ls=":", lw=1, alpha=0.6, zorder=0)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        if style["show_legend"]:
            plt.legend(loc="best", fontsize="small")
        if style["show_grid"]:
            plt.grid(True, linestyle="--", alpha=0.5)

        if output_path:
            plt.savefig(
                output_path, bbox_inches="tight", dpi=style["dpi"], pad_inches=style["pad_inches"]
            )
        else:
            plt.show()
        plt.close()

    # --- Helpers ---

    def _interpolate(self, x_grid, x, y):
        """Robust interpolation that sorts x to handle PR curves correctly."""
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        idx = np.argsort(x)
        return np.interp(x_grid, x[idx], y[idx])

    def _interpolate_pr(self, x_grid, recall, precision):
        """
        Step-wise interpolation for PR curves.

        Linear interpolation is wrong here for two reasons:
        1. Recall has many tied values (thresholds that only add false positives),
           which np.interp does not handle deterministically.
        2. sklearn appends a sentinel point at (recall=0, precision=1); after sorting
           it lands on the left edge of the grid and drags every curve up towards 1.

        Instead we drop the sentinel and take the max precision achievable at
        recall >= x, which is the standard envelope used for averaged PR curves.
        """
        r, p = np.asarray(recall)[:-1], np.asarray(precision)[:-1]
        mask = np.isfinite(r) & np.isfinite(p)
        r, p = r[mask], p[mask]
        if r.size == 0:
            return np.full_like(x_grid, np.nan)

        # Collapse tied recall values to their max precision FIRST. Without this,
        # np.argsort (unstable) scrambles the order within a tie group and both the
        # envelope and np.interp pick an arbitrary member of it.
        r_u, inverse = np.unique(r, return_inverse=True)
        p_u = np.zeros_like(r_u, dtype=float)
        np.maximum.at(p_u, inverse, p)

        p_env = np.maximum.accumulate(p_u[::-1])[::-1]  # max precision at recall >= r
        return np.interp(x_grid, r_u, p_env)

    def _ensure_matrix(self, data):
        """Handles lists of lists or pandas objects."""
        if (isinstance(data, (pd.Series, list)) and len(data) > 0) or (
            isinstance(data, np.ndarray) and data.ndim == 1
        ):
            try:
                return np.vstack(list(data)).astype(float)
            except ValueError as e:
                raise ValueError(
                    "Could not stack probability lists. Ensure all rows have equal length."
                ) from e
        return np.array(data)
