"""Logistic regression explainability functions."""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import statsmodels.api as sm
    from statsmodels.stats.multitest import multipletests

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. P-values will not be computed.")

from .reports import (
    format_confidence_interval,
    get_significance_stars,
    plot_feature_importance_bar,
    write_interpretation_guide,
)


def explain_logistic_regression(model_bundle, X=None, y=None, output_dir=None, verbose=False):
    """
    Generate comprehensive explanation for logistic regression model.

    Args:
        model_bundle: Trained ModelBundle with logistic regression
        X: Feature matrix (optional, needed for p-values)
        y: Labels (optional, needed for p-values)
        output_dir: Directory to save outputs
        verbose: If True, print explanations to console

    Returns:
        dict: Explanation dictionary
    """
    if model_bundle.model_type != "logistic":
        raise ValueError("This function only works with logistic regression models")

    model = model_bundle.model
    feature_names = model_bundle.feature_names

    # Extract coefficients
    coefficients = model.coef_[0]
    intercept = model.intercept_[0]
    odds_ratios = np.exp(coefficients)

    # Compute p-values and confidence intervals if data provided
    if X is not None and y is not None and STATSMODELS_AVAILABLE:
        stats_results = _compute_logistic_statistics(model_bundle, X, y)
        p_values = stats_results["p_values_adj"]
        conf_intervals = stats_results["conf_intervals"]
    else:
        p_values = None
        conf_intervals = None

    # Create explanation dictionary
    explanation = {
        "feature_names": feature_names,
        "coefficients": coefficients,
        "odds_ratios": odds_ratios,
        "p_values_adj": p_values,
        "conf_intervals": conf_intervals,
        "intercept": intercept,
        "n_samples": len(y) if y is not None else None,
    }

    # Save outputs if directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        _save_logistic_text_report(explanation, output_dir / "logistic_explanation.txt")
        _plot_feature_importance(explanation, output_dir / "feature_importance.png")
        _plot_coefficients_with_ci(explanation, output_dir / "coefficients_ci.png")

    # Print if verbose
    if verbose:
        _print_logistic_explanation(explanation)

    return explanation


def _compute_logistic_statistics(model_bundle, X, y):
    """Compute p-values and confidence intervals using statsmodels."""
    if not STATSMODELS_AVAILABLE:
        return {"p_values_adj": None, "conf_intervals": None}

    # Convert labels
    if np.min(y) > 1:
        y_binary = np.where(y == 2, 0, 1)
    else:
        y_binary = y

    # Apply same preprocessing as training
    X_processed = X.copy()
    if model_bundle.log_transform_features and model_bundle.feature_names:
        for i, fname in enumerate(model_bundle.feature_names):
            if fname in model_bundle.log_transform_features:
                X_processed[:, i] = np.log1p(X_processed[:, i])

    X_scaled = model_bundle.scaler.transform(X_processed)
    X_with_intercept = sm.add_constant(X_scaled)

    # Fit with statsmodels
    logit_model = sm.Logit(y_binary, X_with_intercept)
    result = logit_model.fit(disp=0)

    # Extract p-values and apply FDR correction
    p_values_raw = result.pvalues[1:]
    reject, p_values_adj, _, _ = multipletests(p_values_raw, method="fdr_bh")
    conf_int = result.conf_int()[1:]

    return {"p_values_raw": p_values_raw, "p_values_adj": p_values_adj, "conf_intervals": conf_int}


def _save_logistic_text_report(explanation, output_path):
    """Save logistic regression explanation to text file."""
    with open(output_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("LOGISTIC REGRESSION MODEL EXPLANATION\n")
        f.write("=" * 70 + "\n\n")

        if explanation["n_samples"]:
            f.write(f"Sample size: {explanation['n_samples']} lesions\n")
            if explanation["n_samples"] < 100:
                f.write("⚠ WARNING: Small sample size (n < 100). Interpret with caution.\n")
            f.write("\n")

        f.write("Feature Analysis:\n")
        f.write("-" * 70 + "\n")
        f.write(
            f"{'Feature':<20} {'Coef':>8} {'Odds Ratio':>12} {'95% CI':>20} {'P-adj':>10} {'Sig':>5}\n"
        )
        f.write("-" * 70 + "\n")

        # Sort by absolute coefficient
        sorted_indices = np.argsort(np.abs(explanation["coefficients"]))[::-1]

        for idx in sorted_indices:
            fname = explanation["feature_names"][idx]
            coef = explanation["coefficients"][idx]
            odds = explanation["odds_ratios"][idx]

            # Confidence interval
            if explanation["conf_intervals"] is not None:
                ci_lower = np.exp(explanation["conf_intervals"][idx, 0])
                ci_upper = np.exp(explanation["conf_intervals"][idx, 1])
                ci_str = format_confidence_interval(ci_lower, ci_upper)
            else:
                ci_str = "N/A"

            # P-value and significance
            if explanation["p_values_adj"] is not None:
                p_val = explanation["p_values_adj"][idx]
                p_str = f"{p_val:.4f}" if p_val >= 0.001 else "<0.001"
                sig = get_significance_stars(p_val)
            else:
                p_str = "N/A"
                sig = ""

            f.write(f"{fname:<20} {coef:>8.3f} {odds:>12.3f} {ci_str:>20} {p_str:>10} {sig:>5}\n")

        f.write("-" * 70 + "\n")
        if explanation["p_values_adj"] is not None:
            f.write("Significance: *** p < 0.001, ** p < 0.01, * p < 0.05 (FDR-corrected)\n")
        f.write("\n")

        write_interpretation_guide(f, model_type="logistic")


def _print_logistic_explanation(explanation):
    """Print logistic explanation to console."""
    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION MODEL EXPLANATION")
    print("=" * 70 + "\n")

    if explanation["n_samples"]:
        print(f"Sample size: {explanation['n_samples']} lesions")
        if explanation["n_samples"] < 100:
            print("⚠ WARNING: Small sample size (n < 100). Interpret with caution.")
        print()

    print("Feature Analysis:")
    print("-" * 70)
    print(f"{'Feature':<20} {'Coef':>8} {'Odds Ratio':>12} {'95% CI':>20} {'P-adj':>10} {'Sig':>5}")
    print("-" * 70)

    sorted_indices = np.argsort(np.abs(explanation["coefficients"]))[::-1]

    for idx in sorted_indices:
        fname = explanation["feature_names"][idx]
        coef = explanation["coefficients"][idx]
        odds = explanation["odds_ratios"][idx]

        if explanation["conf_intervals"] is not None:
            ci_lower = np.exp(explanation["conf_intervals"][idx, 0])
            ci_upper = np.exp(explanation["conf_intervals"][idx, 1])
            ci_str = format_confidence_interval(ci_lower, ci_upper)
        else:
            ci_str = "N/A"

        if explanation["p_values_adj"] is not None:
            p_val = explanation["p_values_adj"][idx]
            p_str = f"{p_val:.4f}" if p_val >= 0.001 else "<0.001"
            sig = get_significance_stars(p_val)
        else:
            p_str = "N/A"
            sig = ""

        print(f"{fname:<20} {coef:>8.3f} {odds:>12.3f} {ci_str:>20} {p_str:>10} {sig:>5}")

    print("-" * 70)
    if explanation["p_values_adj"] is not None:
        print("Significance: *** p < 0.001, ** p < 0.01, * p < 0.05 (FDR-corrected)")
    print()


def _plot_feature_importance(explanation, output_path):
    """Plot feature importance for logistic regression."""
    importance = np.abs(explanation["coefficients"])
    plot_feature_importance_bar(
        explanation["feature_names"],
        importance,
        output_path,
        title="Feature Importance (Logistic Regression)",
        xlabel="Absolute Coefficient",
    )


def _plot_coefficients_with_ci(explanation, output_path):
    """Plot coefficients with 95% confidence intervals."""
    if explanation["conf_intervals"] is None:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    feature_names = explanation["feature_names"]
    coefficients = explanation["coefficients"]
    conf_intervals = explanation["conf_intervals"]

    # Sort by coefficient
    sorted_indices = np.argsort(coefficients)
    y_pos = np.arange(len(feature_names))

    # Plot coefficients
    colors = ["red" if c < 0 else "green" for c in coefficients[sorted_indices]]
    ax.barh(y_pos, coefficients[sorted_indices], color=colors, alpha=0.6)

    # Plot confidence intervals
    for i, idx in enumerate(sorted_indices):
        ci_lower = conf_intervals[idx, 0]
        ci_upper = conf_intervals[idx, 1]
        ax.plot([ci_lower, ci_upper], [i, i], "k-", linewidth=2)
        ax.plot([ci_lower, ci_lower], [i - 0.1, i + 0.1], "k-", linewidth=2)
        ax.plot([ci_upper, ci_upper], [i - 0.1, i + 0.1], "k-", linewidth=2)

    ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in sorted_indices])
    ax.set_xlabel("Coefficient (95% CI)", fontsize=12)
    ax.set_title(
        "Logistic Regression Coefficients with Confidence Intervals", fontsize=14, fontweight="bold"
    )
    ax.grid(axis="x", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="green", alpha=0.6, label="Favors Cyst (positive)"),
        Patch(facecolor="red", alpha=0.6, label="Favors Tumor (negative)"),
    ]
    ax.legend(handles=legend_elements, loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
