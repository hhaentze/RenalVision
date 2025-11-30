"""Decision tree explainability functions."""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import graphviz
    from sklearn.tree import export_graphviz, plot_tree

    # Test if graphviz executables are available
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

from .reports import plot_feature_importance_bar, write_interpretation_guide


def explain_decision_tree(
    model_bundle, output_dir=None, verbose=False, rule_format="nested", uncertain_leaves=None
):
    """
    Generate comprehensive explanation for decision tree model.

    Args:
        model_bundle: Trained ModelBundle with decision tree
        output_dir: Directory to save outputs
        verbose: If True, print explanations to console
        rule_format: "nested" (if-else) or "flat" (list of paths)
        uncertain_leaves: Optional set of leaf node IDs that are uncertain (for future use)

    Returns:
        dict: Explanation dictionary
    """
    if model_bundle.model_type != "tree":
        raise ValueError("This function only works with decision tree models")

    model = model_bundle.model
    feature_names = model_bundle.feature_names

    # Extract feature importance
    feature_importance = model.feature_importances_

    # Get tree rules
    tree_rules_text = _export_tree_rules(
        model_bundle, format=rule_format, uncertain_leaves=uncertain_leaves
    )

    # Create explanation dictionary
    explanation = {
        "feature_names": feature_names,
        "feature_importance": feature_importance,
        "tree_rules": tree_rules_text,
        "tree_depth": model.get_depth(),
        "n_leaves": model.get_n_leaves(),
        "n_features_used": np.sum(feature_importance > 0),
        "uncertain_leaves": uncertain_leaves,  # Store for visualization
    }

    # Save outputs if directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        _save_tree_text_report(explanation, output_dir / "tree_explanation.txt")

        # Save decision rules
        rules_filename = (
            "decision_rules_uncertain.txt" if uncertain_leaves else "decision_rules.txt"
        )
        with open(output_dir / rules_filename, "w") as f:
            f.write(tree_rules_text)

        # Save visualizations
        _plot_feature_importance(explanation, output_dir / "feature_importance.png")

        viz_suffix = "_with_uncertainty" if uncertain_leaves else "_visualization"
        _plot_tree_visualization(
            model_bundle,
            output_dir / f"tree{viz_suffix}.png",
            format="png",
            uncertain_leaves=uncertain_leaves,
        )

        if GRAPHVIZ_AVAILABLE:
            _plot_tree_visualization(
                model_bundle,
                output_dir / f"tree{viz_suffix}.html",
                format="html",
                uncertain_leaves=uncertain_leaves,
            )

    # Print if verbose
    if verbose:
        _print_tree_explanation(explanation)

    return explanation


def _export_tree_rules(model_bundle, format="nested", uncertain_leaves=None):
    """
    Export decision tree as human-readable rules.

    Args:
        model_bundle: Trained ModelBundle
        format: "nested" or "flat"
        uncertain_leaves: Set of leaf node IDs that are uncertain

    Returns:
        str: Formatted decision rules
    """
    model = model_bundle.model
    feature_names = model_bundle.feature_names
    tree = model.tree_

    if format == "nested":
        # Custom nested format with uncertainty marking
        # We can't use sklearn's export_text because it doesn't give us node IDs

        def export_nested(node, depth=0):
            indent = "  " * depth
            result = []

            if tree.feature[node] != -2:  # Not a leaf (internal node)
                fname = feature_names[tree.feature[node]]
                thresh = tree.threshold[node]

                # Node line
                result.append(f"{indent}|--- {fname} <= {thresh:.2f}")
                # Left subtree
                result.extend(export_nested(tree.children_left[node], depth + 1))

                # Right node line
                result.append(f"{indent}|--- {fname} > {thresh:.2f}")
                # Right subtree
                result.extend(export_nested(tree.children_right[node], depth + 1))

            else:  # Leaf node
                values = tree.value[node][0]
                total = sum(values)
                class_idx = np.argmax(values)
                confidence = values[class_idx] / total * 100

                # Determine class and uncertainty
                if uncertain_leaves and node in uncertain_leaves:
                    class_name = "Unsure [UNCERTAIN]"
                else:
                    class_name = "Tumor" if class_idx == 0 else "Cyst"

                # Show proportions, not raw counts
                tumor_prop = values[0] / total
                cyst_prop = values[1] / total

                result.append(
                    f"{indent}|--- class: {class_name} "
                    f"(tumor: {tumor_prop:.3f}, cyst: {cyst_prop:.3f}, "
                    f"confidence: {confidence:.1f}%, node_id={node})"
                )

            return result

        lines = export_nested(0)
        tree_rules = "\n".join(lines)

        if uncertain_leaves and len(uncertain_leaves) > 0:
            tree_rules += f"\n\n[{len(uncertain_leaves)} leaves marked as [UNCERTAIN] based on confidence threshold]\n"

        return tree_rules

    elif format == "flat":
        # Extract all paths with uncertainty marking
        feature = tree.feature
        threshold = tree.threshold

        def recurse(node, depth, path, paths):
            if tree.feature[node] != -2:  # Not a leaf
                fname = feature_names[feature[node]]
                thresh = threshold[node]

                # Left child
                left_path = path + [f"{fname} <= {thresh:.2f}"]
                recurse(tree.children_left[node], depth + 1, left_path, paths)

                # Right child
                right_path = path + [f"{fname} > {thresh:.2f}"]
                recurse(tree.children_right[node], depth + 1, right_path, paths)
            else:
                # Leaf node
                values = tree.value[node][0]
                total = sum(values)
                class_idx = np.argmax(values)
                confidence = values[class_idx] / total * 100

                # Show proportions
                tumor_prop = values[0] / total
                cyst_prop = values[1] / total

                # Determine class name
                if uncertain_leaves and node in uncertain_leaves:
                    class_name = "Unsure"
                    class_marker = " [UNCERTAIN]"
                else:
                    class_name = "Tumor" if class_idx == 0 else "Cyst"
                    class_marker = ""

                path_str = " AND ".join(path) if path else "Root"
                result = (
                    f"Path {len(paths) + 1}: {path_str}\n"
                    f"  → {class_name}{class_marker} "
                    f"(tumor: {tumor_prop:.3f}, cyst: {cyst_prop:.3f}, "
                    f"confidence: {confidence:.1f}%, node_id={node})\n"
                )
                paths.append(result)

        paths = []
        recurse(0, 0, [], paths)

        output = "\n".join(paths)
        if uncertain_leaves and len(uncertain_leaves) > 0:
            output += f"\n\n{len(uncertain_leaves)} leaves marked as [UNCERTAIN] with confidence below threshold\n"

        return output

    else:
        raise ValueError(f"Unknown format: {format}")


def _save_tree_text_report(explanation, output_path):
    """Save decision tree explanation to text file."""
    with open(output_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("DECISION TREE MODEL EXPLANATION\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Tree depth: {explanation['tree_depth']}\n")
        f.write(f"Number of leaves: {explanation['n_leaves']}\n")
        f.write(
            f"Features used: {explanation['n_features_used']} / {len(explanation['feature_names'])}\n"
        )

        if explanation["uncertain_leaves"]:
            f.write(f"Uncertain leaves: {len(explanation['uncertain_leaves'])}\n")

        f.write("\n")

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

        write_interpretation_guide(f, model_type="tree")


def _print_tree_explanation(explanation):
    """Print decision tree explanation to console."""
    print("\n" + "=" * 70)
    print("DECISION TREE MODEL EXPLANATION")
    print("=" * 70 + "\n")

    print(f"Tree depth: {explanation['tree_depth']}")
    print(f"Number of leaves: {explanation['n_leaves']}")
    print(f"Features used: {explanation['n_features_used']} / {len(explanation['feature_names'])}")

    if explanation["uncertain_leaves"]:
        print(f"Uncertain leaves: {len(explanation['uncertain_leaves'])}")

    print("\nFeature Importance:")
    print("-" * 70)
    print(f"{'Feature':<25} {'Importance':>12} {'Bar':>20}")
    print("-" * 70)

    sorted_indices = np.argsort(explanation["feature_importance"])[::-1]

    for idx in sorted_indices:
        if explanation["feature_importance"][idx] > 0:
            fname = explanation["feature_names"][idx]
            importance = explanation["feature_importance"][idx]
            bar_length = int(importance * 50)
            bar = "█" * bar_length
            print(f"{fname:<25} {importance:>12.4f} {bar:>20}")

    print("-" * 70 + "\n")


def _plot_feature_importance(explanation, output_path):
    """Plot feature importance for decision tree."""
    plot_feature_importance_bar(
        explanation["feature_names"],
        explanation["feature_importance"],
        output_path,
        title="Feature Importance (Decision Tree)",
        xlabel="Importance",
    )


def _plot_tree_visualization(model_bundle, output_path, format="png", uncertain_leaves=None):
    """
    Visualize decision tree structure.

    Args:
        model_bundle: Trained ModelBundle
        output_path: Path to save visualization
        format: 'png' or 'html'
        uncertain_leaves: Set of leaf node IDs that are uncertain (for future color coding)
    """
    model = model_bundle.model
    feature_names = model_bundle.feature_names
    class_names = ["Tumor", "Cyst"]

    if format == "png":
        # Use matplotlib with proportions
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(
            model,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            fontsize=10,
            ax=ax,
            proportion=True,  # Show proportions instead of raw counts
        )

        # Note: uncertain_leaves could be used here for custom coloring in the future
        # For now, we use standard sklearn coloring

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    elif format == "html":
        if not GRAPHVIZ_AVAILABLE:
            warnings.warn(
                "Graphviz executables not available. Skipping HTML visualization.\n"
                "Install from: https://graphviz.org/download/"
            )
            return

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
            svg_data = graph.pipe(format="svg").decode("utf-8")

            # Create HTML wrapper
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

        except (graphviz.backend.ExecutableNotFound, FileNotFoundError) as e:
            warnings.warn(
                f"Failed to generate HTML visualization: {e}\n"
                "Graphviz executables not found on PATH."
            )
