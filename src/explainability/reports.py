"""Shared utilities for generating reports and visualizations."""

import matplotlib.pyplot as plt
import numpy as np


def plot_feature_importance_bar(
    feature_names, importance_values, output_path, title="Feature Importance", xlabel="Importance"
):
    """
    Plot feature importance as horizontal bar chart.

    Args:
        feature_names: List of feature names
        importance_values: Array of importance values
        output_path: Path to save plot
        title: Plot title
        xlabel: X-axis label
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by importance
    sorted_indices = np.argsort(importance_values)

    # Plot
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, importance_values[sorted_indices], color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in sorted_indices])
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def format_confidence_interval(ci_lower, ci_upper):
    """Format confidence interval as string."""
    return f"[{ci_lower:5.2f}, {ci_upper:5.2f}]"


def get_significance_stars(p_value):
    """Get significance stars for p-value."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return ""


def write_interpretation_guide(f, model_type="logistic"):
    """
    Write interpretation guide to file.

    Args:
        f: Open file handle
        model_type: "logistic" or "tree"
    """
    f.write("\n" + "=" * 70 + "\n")
    f.write("INTERPRETATION GUIDE\n")
    f.write("=" * 70 + "\n\n")

    if model_type == "logistic":
        f.write("Coefficients:\n")
        f.write("  - Positive coefficient → feature increases cyst probability\n")
        f.write("  - Negative coefficient → feature increases tumor probability\n")
        f.write("  - Magnitude indicates strength of association\n")
        f.write("  - Note: Values are for standardized features\n\n")

        f.write("Odds Ratios (OR):\n")
        f.write("  - OR > 1: Feature associated with cysts\n")
        f.write("  - OR < 1: Feature associated with tumors\n")
        f.write("  - OR = 2.0 means doubling the feature value (after standardization)\n")
        f.write("    doubles the odds of being a cyst\n")
        f.write("  - OR = 0.5 means doubling the feature value halves the odds\n\n")

        f.write("P-values (FDR-adjusted):\n")
        f.write("  - Tests if feature has significant predictive power\n")
        f.write("  - Values corrected for multiple testing (Benjamini-Hochberg)\n")
        f.write("  - p < 0.05 suggests feature is reliably predictive\n\n")

        f.write("95% Confidence Intervals:\n")
        f.write("  - Range for odds ratio with 95% confidence\n")
        f.write("  - If interval includes 1.0, effect may not be significant\n")
        f.write("  - Wider intervals indicate more uncertainty\n\n")

    elif model_type == "tree":
        f.write("Feature Importance:\n")
        f.write("  - Measures how much each feature contributes to splitting decisions\n")
        f.write("  - Higher values = more important for classification\n")
        f.write("  - Values sum to 1.0\n")
        f.write("  - Features with 0 importance are not used in the tree\n\n")

        f.write("Decision Rules:\n")
        f.write("  - See decision_rules.txt for complete decision paths\n")
        f.write("  - Each path shows conditions leading to a prediction\n")
        f.write("  - Confidence indicates % of training samples at that leaf\n")
        f.write("  - Apply rules sequentially from top to bottom\n\n")

        f.write("Clinical Use:\n")
        f.write("  - Follow the decision tree from root to leaf\n")
        f.write("  - At each node, check if feature value satisfies condition\n")
        f.write("  - Continue down appropriate branch until reaching prediction\n")
        f.write("  - Simpler trees (depth 3-5) are more interpretable for clinicians\n\n")
