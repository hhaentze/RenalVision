"""Model explainability and interpretation tools."""

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

try:
    import graphviz
    from sklearn.tree import export_graphviz, export_text, plot_tree

    # Test if graphviz executables are actually available
    try:
        graphviz.Source("digraph { a -> b }").pipe(format="svg")
        GRAPHVIZ_AVAILABLE = True
    except (graphviz.backend.ExecutableNotFound, FileNotFoundError):
        GRAPHVIZ_AVAILABLE = False
        warnings.warn(
            "Graphviz executables not found. Interactive HTML tree will not be generated.\n"
            "To enable: Install Graphviz from https://graphviz.org/download/ and add to PATH."
        )
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    warnings.warn("graphviz package not available. Tree visualizations will be limited.")


# ============================================================================
# Logistic Regression Explainability
# ============================================================================


def explain_logistic_regression(model_bundle, X=None, y=None, output_dir=None, verbose=False):
    """
    Generate comprehensive explanation for logistic regression model.

    Args:
        model_bundle: Trained ModelBundle with logistic regression
        X: Feature matrix (optional, needed for p-values)
        y: Labels (optional, needed for p-values)
        output_dir: Directory to save outputs (optional)
        verbose: If True, print explanations to console

    Returns:
        dict: Explanation dictionary with coefficients, odds ratios, p-values, etc.
    """
    if model_bundle.model_type != "logistic":
        raise ValueError("This function only works with logistic regression models")

    model = model_bundle.model
    feature_names = model_bundle.feature_names

    # Extract coefficients (note: after standardization)
    coefficients = model.coef_[0]
    intercept = model.intercept_[0]

    # Compute odds ratios
    odds_ratios = np.exp(coefficients)

    # Compute p-values and confidence intervals if data provided
    if X is not None and y is not None and STATSMODELS_AVAILABLE:
        stats_results = compute_logistic_statistics(model_bundle, X, y)
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

    # Generate outputs
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save text report
        save_logistic_text_report(explanation, output_dir / "logistic_explanation.txt")

        # Save visualizations
        plot_feature_importance(
            explanation, output_dir / "feature_importance.png", model_type="logistic"
        )
        plot_coefficients_with_ci(explanation, output_dir / "coefficients_ci.png")

    # Print if verbose
    if verbose:
        print_logistic_explanation(explanation)

    return explanation


def compute_logistic_statistics(model_bundle, X, y):
    """
    Compute statistical significance for logistic regression coefficients.

    Uses statsmodels to compute p-values and confidence intervals.
    Applies FDR (Benjamini-Hochberg) correction for multiple testing.

    Args:
        model_bundle: Trained ModelBundle
        X: Feature matrix
        y: Labels (2=tumor, 3=cyst or 0=tumor, 1=cyst)

    Returns:
        dict: Statistical results including p-values and confidence intervals
    """
    if not STATSMODELS_AVAILABLE:
        return {"p_values_adj": None, "conf_intervals": None}

    # Convert labels if needed
    if np.min(y) > 1:
        y_binary = np.where(y == 2, 0, 1)
    else:
        y_binary = y

    # Apply same preprocessing as in training
    X_processed = X.copy()
    if model_bundle.log_transform_features and model_bundle.feature_names:
        for i, fname in enumerate(model_bundle.feature_names):
            if fname in model_bundle.log_transform_features:
                X_processed[:, i] = np.log1p(X_processed[:, i])

    # Standardize
    X_scaled = model_bundle.scaler.transform(X_processed)

    # Add intercept for statsmodels
    X_with_intercept = sm.add_constant(X_scaled)

    # Fit logistic regression with statsmodels
    logit_model = sm.Logit(y_binary, X_with_intercept)
    result = logit_model.fit(disp=0)

    # Extract p-values (skip intercept)
    p_values_raw = result.pvalues[1:]

    # Apply FDR correction (Benjamini-Hochberg)
    reject, p_values_adj, _, _ = multipletests(p_values_raw, method="fdr_bh")

    # Extract 95% confidence intervals (skip intercept)
    conf_int = result.conf_int()[1:]

    return {
        "p_values_raw": p_values_raw,
        "p_values_adj": p_values_adj,
        "conf_intervals": conf_int,
        "statsmodels_result": result,
    }


def save_logistic_text_report(explanation, output_path):
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

        # Sort by absolute coefficient value
        sorted_indices = np.argsort(np.abs(explanation["coefficients"]))[::-1]

        for idx in sorted_indices:
            fname = explanation["feature_names"][idx]
            coef = explanation["coefficients"][idx]
            odds = explanation["odds_ratios"][idx]

            # Confidence interval
            if explanation["conf_intervals"] is not None:
                ci_lower = np.exp(explanation["conf_intervals"][idx, 0])
                ci_upper = np.exp(explanation["conf_intervals"][idx, 1])
                ci_str = f"[{ci_lower:5.2f}, {ci_upper:5.2f}]"
            else:
                ci_str = "N/A"

            # P-value and significance
            if explanation["p_values_adj"] is not None:
                p_val = explanation["p_values_adj"][idx]
                p_str = f"{p_val:.4f}" if p_val >= 0.001 else "<0.001"

                if p_val < 0.001:
                    sig = "***"
                elif p_val < 0.01:
                    sig = "**"
                elif p_val < 0.05:
                    sig = "*"
                else:
                    sig = ""
            else:
                p_str = "N/A"
                sig = ""

            f.write(f"{fname:<20} {coef:>8.3f} {odds:>12.3f} {ci_str:>20} {p_str:>10} {sig:>5}\n")

        f.write("-" * 70 + "\n")
        if explanation["p_values_adj"] is not None:
            f.write("Significance: *** p < 0.001, ** p < 0.01, * p < 0.05 (FDR-corrected)\n")
        f.write("\n")

        # Interpretation guide
        f.write("\n" + "=" * 70 + "\n")
        f.write("INTERPRETATION GUIDE\n")
        f.write("=" * 70 + "\n\n")

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


def print_logistic_explanation(explanation):
    """Print logistic regression explanation to console."""
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

    # Sort by absolute coefficient value
    sorted_indices = np.argsort(np.abs(explanation["coefficients"]))[::-1]

    for idx in sorted_indices:
        fname = explanation["feature_names"][idx]
        coef = explanation["coefficients"][idx]
        odds = explanation["odds_ratios"][idx]

        # Confidence interval
        if explanation["conf_intervals"] is not None:
            ci_lower = np.exp(explanation["conf_intervals"][idx, 0])
            ci_upper = np.exp(explanation["conf_intervals"][idx, 1])
            ci_str = f"[{ci_lower:5.2f}, {ci_upper:5.2f}]"
        else:
            ci_str = "N/A"

        # P-value and significance
        if explanation["p_values_adj"] is not None:
            p_val = explanation["p_values_adj"][idx]
            p_str = f"{p_val:.4f}" if p_val >= 0.001 else "<0.001"

            if p_val < 0.001:
                sig = "***"
            elif p_val < 0.01:
                sig = "**"
            elif p_val < 0.05:
                sig = "*"
            else:
                sig = ""
        else:
            p_str = "N/A"
            sig = ""

        print(f"{fname:<20} {coef:>8.3f} {odds:>12.3f} {ci_str:>20} {p_str:>10} {sig:>5}")

    print("-" * 70)
    if explanation["p_values_adj"] is not None:
        print("Significance: *** p < 0.001, ** p < 0.01, * p < 0.05 (FDR-corrected)")
    print()


# ============================================================================
# Decision Tree Explainability
# ============================================================================


def explain_decision_tree(model_bundle, output_dir=None, verbose=False, rule_format="nested"):
    """
    Generate comprehensive explanation for decision tree model.

    Args:
        model_bundle: Trained ModelBundle with decision tree
        output_dir: Directory to save outputs (optional)
        verbose: If True, print explanations to console
        rule_format: "nested" (if-else) or "flat" (list of paths)

    Returns:
        dict: Explanation dictionary with feature importance, rules, etc.
    """
    if model_bundle.model_type != "tree":
        raise ValueError("This function only works with decision tree models")

    model = model_bundle.model
    feature_names = model_bundle.feature_names

    # Extract feature importance
    feature_importance = model.feature_importances_

    # Get tree rules in text format
    tree_rules_text = export_tree_rules(model_bundle, format=rule_format)

    # Create explanation dictionary
    explanation = {
        "feature_names": feature_names,
        "feature_importance": feature_importance,
        "tree_rules": tree_rules_text,
        "tree_depth": model.get_depth(),
        "n_leaves": model.get_n_leaves(),
        "n_features_used": np.sum(feature_importance > 0),
    }

    # Generate outputs
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save text report
        save_tree_text_report(explanation, output_dir / "tree_explanation.txt")

        # Save decision rules
        with open(output_dir / "decision_rules.txt", "w") as f:
            f.write(tree_rules_text)

        # Save visualizations
        plot_feature_importance(
            explanation, output_dir / "feature_importance.png", model_type="tree"
        )
        plot_tree_visualization(model_bundle, output_dir / "tree_visualization.png", format="png")

        if GRAPHVIZ_AVAILABLE:
            plot_tree_visualization(
                model_bundle, output_dir / "tree_interactive.html", format="html"
            )

    # Print if verbose
    if verbose:
        print_tree_explanation(explanation)

    return explanation


def export_tree_rules(model_bundle, format="nested"):
    """
    Export decision tree as human-readable rules.

    Args:
        model_bundle: Trained ModelBundle with decision tree
        format: "nested" (if-else structure) or "flat" (list of paths)

    Returns:
        str: Formatted decision rules
    """
    model = model_bundle.model
    feature_names = model_bundle.feature_names

    if format == "nested":
        # Use sklearn's export_text for nested structure
        tree_rules = export_text(
            model,
            feature_names=feature_names,
            class_names=["Tumor", "Cyst"],
            decimals=2,
            show_weights=True,
        )
        return tree_rules

    elif format == "flat":
        # Extract all paths from root to leaves
        tree = model.tree_
        feature = tree.feature
        threshold = tree.threshold

        def recurse(node, depth, path, paths):
            indent = "  " * depth

            if tree.feature[node] != -2:  # Not a leaf
                fname = feature_names[feature[node]]
                thresh = threshold[node]

                # Left child (<=)
                left_path = path + [f"{fname} <= {thresh:.2f}"]
                recurse(tree.children_left[node], depth + 1, left_path, paths)

                # Right child (>)
                right_path = path + [f"{fname} > {thresh:.2f}"]
                recurse(tree.children_right[node], depth + 1, right_path, paths)
            else:
                # Leaf node
                values = tree.value[node][0]
                total = sum(values)
                class_idx = np.argmax(values)
                class_name = "Tumor" if class_idx == 0 else "Cyst"
                confidence = values[class_idx] / total * 100

                path_str = " AND ".join(path) if path else "Root"
                result = f"Path {len(paths) + 1}: {path_str}\n  → {class_name} (confidence: {confidence:.1f}%)\n"
                paths.append(result)

        paths = []
        recurse(0, 0, [], paths)
        return "\n".join(paths)

    else:
        raise ValueError(f"Unknown format: {format}. Use 'nested' or 'flat'")


def save_tree_text_report(explanation, output_path):
    """Save decision tree explanation to text file."""
    with open(output_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("DECISION TREE MODEL EXPLANATION\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Tree depth: {explanation['tree_depth']}\n")
        f.write(f"Number of leaves: {explanation['n_leaves']}\n")
        f.write(
            f"Features used: {explanation['n_features_used']} / {len(explanation['feature_names'])}\n\n"
        )

        f.write("Feature Importance:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Feature':<25} {'Importance':>12} {'Bar':>20}\n")
        f.write("-" * 70 + "\n")

        # Sort by importance
        sorted_indices = np.argsort(explanation["feature_importance"])[::-1]

        for idx in sorted_indices:
            if explanation["feature_importance"][idx] > 0:
                fname = explanation["feature_names"][idx]
                importance = explanation["feature_importance"][idx]
                bar_length = int(importance * 50)
                bar = "█" * bar_length
                f.write(f"{fname:<25} {importance:>12.4f} {bar:>20}\n")

        f.write("-" * 70 + "\n\n")

        # Interpretation guide
        f.write("\n" + "=" * 70 + "\n")
        f.write("INTERPRETATION GUIDE\n")
        f.write("=" * 70 + "\n\n")

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


def print_tree_explanation(explanation):
    """Print decision tree explanation to console."""
    print("\n" + "=" * 70)
    print("DECISION TREE MODEL EXPLANATION")
    print("=" * 70 + "\n")

    print(f"Tree depth: {explanation['tree_depth']}")
    print(f"Number of leaves: {explanation['n_leaves']}")
    print(
        f"Features used: {explanation['n_features_used']} / {len(explanation['feature_names'])}\n"
    )

    print("Feature Importance:")
    print("-" * 70)
    print(f"{'Feature':<25} {'Importance':>12} {'Bar':>20}")
    print("-" * 70)

    # Sort by importance
    sorted_indices = np.argsort(explanation["feature_importance"])[::-1]

    for idx in sorted_indices:
        if explanation["feature_importance"][idx] > 0:
            fname = explanation["feature_names"][idx]
            importance = explanation["feature_importance"][idx]
            bar_length = int(importance * 50)
            bar = "█" * bar_length
            print(f"{fname:<25} {importance:>12.4f} {bar:>20}")

    print("-" * 70 + "\n")


# ============================================================================
# Visualization Functions
# ============================================================================


def plot_feature_importance(explanation, output_path, model_type="logistic"):
    """
    Plot feature importance as horizontal bar chart.

    Args:
        explanation: Explanation dictionary from explain_* functions
        output_path: Path to save plot
        model_type: "logistic" or "tree"
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    feature_names = explanation["feature_names"]

    if model_type == "logistic":
        # Use absolute coefficient values
        importance = np.abs(explanation["coefficients"])
        title = "Feature Importance (Logistic Regression)"
        xlabel = "Absolute Coefficient"
    else:
        # Use feature importance from tree
        importance = explanation["feature_importance"]
        title = "Feature Importance (Decision Tree)"
        xlabel = "Importance"

    # Sort by importance
    sorted_indices = np.argsort(importance)

    # Plot
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, importance[sorted_indices], color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in sorted_indices])
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_coefficients_with_ci(explanation, output_path):
    """
    Plot logistic regression coefficients with 95% confidence intervals.

    Args:
        explanation: Explanation dictionary from explain_logistic_regression
        output_path: Path to save plot
    """
    if explanation["conf_intervals"] is None:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    feature_names = explanation["feature_names"]
    coefficients = explanation["coefficients"]
    conf_intervals = explanation["conf_intervals"]

    # Sort by coefficient value
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

    # Add vertical line at 0
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in sorted_indices])
    ax.set_xlabel("Coefficient (95% CI)", fontsize=12)
    ax.set_title(
        "Logistic Regression Coefficients with Confidence Intervals", fontsize=14, fontweight="bold"
    )
    ax.grid(axis="x", alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="green", alpha=0.6, label="Favors Cyst (positive)"),
        Patch(facecolor="red", alpha=0.6, label="Favors Tumor (negative)"),
    ]
    ax.legend(handles=legend_elements, loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_tree_visualization(model_bundle, output_path, format="png"):
    """
    Visualize decision tree structure.

    Args:
        model_bundle: Trained ModelBundle with decision tree
        output_path: Path to save visualization
        format: 'png' or 'html'
    """
    model = model_bundle.model
    feature_names = model_bundle.feature_names
    class_names = ["Tumor", "Cyst"]

    if format == "png":
        # Use matplotlib
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(
            model,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            fontsize=10,
            ax=ax,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    elif format == "html":
        if not GRAPHVIZ_AVAILABLE:
            warnings.warn(
                "Graphviz executables not available. Skipping HTML visualization.\n"
                "Install from: https://graphviz.org/download/\n"
                "After installation, add to PATH and restart your terminal."
            )
            return

        # Use graphviz for interactive HTML
        try:
            dot_data = export_graphviz(
                model,
                out_file=None,
                feature_names=feature_names,
                class_names=class_names,
                filled=True,
                rounded=True,
                special_characters=True,
                proportion=True,
            )

            graph = graphviz.Source(dot_data)

            # Save as SVG embedded in HTML
            svg_data = graph.pipe(format="svg").decode("utf-8")

        except (graphviz.backend.ExecutableNotFound, FileNotFoundError) as e:
            warnings.warn(
                f"Failed to generate HTML visualization: {e}\n"
                "Graphviz executables not found on PATH.\n"
                "Install from: https://graphviz.org/download/"
            )
            return

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Decision Tree Visualization</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .info {{
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        svg {{
            display: block;
            margin: 0 auto;
            cursor: move;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Interactive Decision Tree</h1>
        <div class="info">
            <p><strong>Instructions:</strong> The decision tree shows the classification logic.</p>
            <ul>
                <li>Each box represents a decision node or prediction</li>
                <li>Orange boxes = Tumor predictions</li>
                <li>Blue boxes = Cyst predictions</li>
                <li>Top line shows the decision rule</li>
                <li>gini = measure of impurity (0 = pure)</li>
                <li>samples = number of training samples at this node</li>
                <li>value = [tumor_count, cyst_count]</li>
            </ul>
        </div>
        {svg_data}
    </div>
</body>
</html>
"""

        with open(output_path, "w") as f:
            f.write(html_content)


# ============================================================================
# Master Report Generation
# ============================================================================


def generate_explanation_report(
    model_bundle, output_dir, X=None, y=None, verbose=False, rule_format="nested"
):
    """
    Generate comprehensive explanation report for any model type.

    This is the main function called from main.py when --explain flag is set.

    Args:
        model_bundle: Trained ModelBundle
        output_dir: Directory to save all outputs
        X: Feature matrix (optional, needed for statistical tests)
        y: Labels (optional, needed for statistical tests)
        verbose: If True, print explanations to console
        rule_format: For trees, "nested" or "flat" rule format
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_bundle.model_type == "logistic":
        explanation = explain_logistic_regression(model_bundle, X, y, output_dir, verbose)
    elif model_bundle.model_type == "tree":
        explanation = explain_decision_tree(model_bundle, output_dir, verbose, rule_format)
    else:
        raise ValueError(f"Unknown model type: {model_bundle.model_type}")

    if verbose:
        print(f"\nExplanation files saved to: {output_dir}")

    return explanation
