"""Decision tree explainability functions."""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .html_helper import wrap_svg

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

        total_samples = tree.n_node_samples[0]

        def export_nested(node, depth=0):
            indent = "  " * depth
            result = []

            node_samples = tree.n_node_samples[node]
            sample_pct = (node_samples / total_samples) * 100

            if tree.feature[node] != -2:  # Not a leaf (internal node)
                fname = feature_names[tree.feature[node]]
                thresh = tree.threshold[node]

                # Node line with bold decision rule
                result.append(
                    f"{indent}|--- {fname} <= {thresh:.2f} [samples: {sample_pct:.1f}% (n={node_samples})]"
                )
                # Left subtree
                result.extend(export_nested(tree.children_left[node], depth + 1))

                # Right node line
                result.append(
                    f"{indent}|--- {fname} > {thresh:.2f} [samples: {sample_pct:.1f}% (n={node_samples})]"
                )
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
                    f"confidence: {confidence:.1f}%, samples: {sample_pct:.1f}% (n={node_samples}), node_id={node})"
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
        total_samples = tree.n_node_samples[0]

        def recurse(node, depth, path, paths):
            node_samples = tree.n_node_samples[node]
            sample_pct = (node_samples / total_samples) * 100

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
                    f"confidence: {confidence:.1f}%, samples: {sample_pct:.1f}% (n={node_samples}), node_id={node})\n"
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
    Visualize decision tree structure with custom coloring for uncertainty.

    Args:
        model_bundle: Trained ModelBundle
        output_path: Path to save visualization
        format: 'png' or 'html'
        uncertain_leaves: Set of leaf node IDs that are uncertain
    """
    model = model_bundle.model
    tree = model.tree_  # Extract tree structure
    feature_names = model_bundle.feature_names
    class_names = ["Tumor", "Cyst"]

    if format == "png" or format == "html":
        # Generate DOT string
        dot_data = export_graphviz(
            model,
            out_file=None,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            special_characters=True,
            proportion=True,  # Show proportions (normalized values)
        )

        # Customize DOT string for uncertainty-aware coloring and formatting
        total_samples = tree.n_node_samples[0]  # Total samples at root
        if uncertain_leaves and len(uncertain_leaves) > 0:
            dot_data = _customize_dot_for_uncertainty(
                dot_data, tree, uncertain_leaves, total_samples
            )
        else:
            # Even without uncertainty, apply formatting improvements
            dot_data = _customize_dot_for_uncertainty(dot_data, tree, set(), total_samples)

        if format == "png":
            # Render to PNG using graphviz
            if not GRAPHVIZ_AVAILABLE:
                warnings.warn("Graphviz not available. Using matplotlib fallback.")
                # Fallback to matplotlib
                fig, ax = plt.subplots(figsize=(20, 10))
                plot_tree(
                    model,
                    feature_names=feature_names,
                    class_names=class_names,
                    filled=True,
                    rounded=True,
                    fontsize=10,
                    ax=ax,
                    proportion=True,
                )
                plt.tight_layout()
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                plt.close()
            else:
                try:
                    graph = graphviz.Source(dot_data)
                    graph.render(
                        filename=output_path.stem,
                        directory=output_path.parent,
                        format="png",
                        cleanup=True,
                    )
                except Exception as e:
                    warnings.warn(f"Graphviz rendering failed: {e}. Using matplotlib fallback.")
                    fig, ax = plt.subplots(figsize=(20, 10))
                    plot_tree(
                        model,
                        feature_names=feature_names,
                        class_names=class_names,
                        filled=True,
                        rounded=True,
                        fontsize=10,
                        ax=ax,
                        proportion=True,
                    )
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
                graph = graphviz.Source(dot_data)
                svg_data = graph.pipe(format="svg").decode("utf-8")

                # Create HTML wrapper
                html_content = wrap_svg(svg_data)

                with open(output_path, "w") as f:
                    f.write(html_content)

            except (graphviz.backend.ExecutableNotFound, FileNotFoundError) as e:
                warnings.warn(
                    f"Failed to generate HTML visualization: {e}\n"
                    "Graphviz executables not found on PATH."
                )


def _customize_dot_for_uncertainty(dot_string, tree, uncertain_leaves, total_samples):
    """
    Customize DOT string to handle uncertainty and improve visualization.

    Args:
        dot_string: Original DOT string from export_graphviz
        tree: sklearn tree object (model.tree_)
        uncertain_leaves: Set of uncertain leaf node IDs
        total_samples: Total number of samples in training set

    Returns:
        Modified DOT string
    """
    lines = dot_string.split("\n")

    # First pass: identify all nodes and their properties
    node_info = {}  # node_id -> {is_leaf, class_idx, color, label}

    # Build parent-child relationships
    children = {}  # parent_id -> (left_child, right_child)
    for node_id in range(tree.node_count):
        if tree.feature[node_id] != -2:  # Internal node
            children[node_id] = (tree.children_left[node_id], tree.children_right[node_id])

    # Analyze each node
    for node_id in range(tree.node_count):
        is_leaf = tree.feature[node_id] == -2

        if is_leaf:
            values = tree.value[node_id][0]
            class_idx = np.argmax(values)

            if node_id in uncertain_leaves:
                # Uncertain leaf
                node_info[node_id] = {
                    "is_leaf": True,
                    "class_idx": -1,  # Special: uncertain
                    "color_base": "gray",
                    "is_uncertain": True,
                }
            else:
                # Certain leaf
                node_info[node_id] = {
                    "is_leaf": True,
                    "class_idx": class_idx,
                    "color_base": "orange" if class_idx == 0 else "blue",
                    "is_uncertain": False,
                }
        else:
            # Internal node - will determine color based on children
            node_info[node_id] = {
                "is_leaf": False,
                "class_idx": None,
                "color_base": "yellow",  # Default neutral
                "is_uncertain": False,
            }

    # Propagate colors upward: if both children have same class, parent gets that class
    for node_id in reversed(range(tree.node_count)):  # Bottom-up
        if node_id in children:
            left_child, right_child = children[node_id]
            left_info = node_info[left_child]
            right_info = node_info[right_child]

            # Check if both children have same class (and neither is uncertain)
            if (
                not left_info["is_uncertain"]
                and not right_info["is_uncertain"]
                and left_info["class_idx"] == right_info["class_idx"]
                and left_info["class_idx"] is not None
            ):
                # Both children agree on class
                node_info[node_id]["class_idx"] = left_info["class_idx"]
                node_info[node_id]["color_base"] = left_info["color_base"]
            # Otherwise keep neutral yellow

    # Second pass: modify DOT lines
    modified_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Check if this is a node definition line (contains node ID and label)
        if " [label=" in line and "fillcolor=" in line:
            # Extract node ID (format: "0 [label=...")
            node_id_match = line.split("[")[0].strip()
            try:
                node_id = int(node_id_match)
                info = node_info.get(node_id)

                if info:
                    # Modify the line
                    modified_line = _modify_node_line(line, node_id, info, tree, total_samples)
                    modified_lines.append(modified_line)
                else:
                    modified_lines.append(line)
            except Exception:
                modified_lines.append(line)
        else:
            modified_lines.append(line)

        i += 1

    return "\n".join(modified_lines)


def _modify_node_line(line, node_id, info, tree, total_samples):
    """
    Modify a single node line in DOT format.

    Applies:
    - Custom colors based on class and uncertainty
    - Bold formatting for decision rules and class labels
    - Percentage display for samples

    Args:
        line: Original DOT line
        node_id: Node ID
        info: Node info dict from _customize_dot_for_uncertainty
        tree: sklearn tree object
        total_samples: Total samples in training set

    Returns:
        Modified DOT line
    """
    import re

    # Get node samples
    node_samples = tree.n_node_samples[node_id]
    sample_percentage = (node_samples / total_samples) * 100

    if info["is_leaf"]:
        # Leaf node
        if info["is_uncertain"]:
            # Uncertain: gray color, change class label to bold Unsure
            line = re.sub(r"class = \w+", "<b>class = Unsure</b>", line)
            line = re.sub(r'fillcolor="[^"]+"', 'fillcolor="#d3d3d3"', line)
        else:
            # Certain: keep original color (preserves purity hue), bold class
            class_name = "Tumor" if info["class_idx"] == 0 else "Cyst"
            line = re.sub(r"class = \w+", f"<b>class = {class_name}</b>", line)
    else:
        # Internal node
        if info["class_idx"] is not None:
            # Both children have same class - keep propagated color and bold class
            class_name = "Tumor" if info["class_idx"] == 0 else "Cyst"
            line = re.sub(r"class = \w+", f"<b>class = {class_name}</b>", line)
            # Keep original color for internal nodes with propagated class
        else:
            # Neutral internal node - remove class label, use neutral color
            line = re.sub(r"<br/>class = \w+", "", line)  # Remove class label
            line = re.sub(r'fillcolor="[^"]+"', 'fillcolor="#ffffcc"', line)  # Light yellow

    # Extract label content and prepare for HTML formatting
    label_match = re.search(r"label=<(.+)(?<!/)>", line)
    if not label_match:
        return line

    label_content = label_match.group(1)
    parts = label_content.split("<br/>")

    # Process first line - check if it's a decision rule
    if len(parts) > 0:
        first_line = parts[0]
        # Graphviz uses HTML entities: &le; for <=, &gt; for >, etc.
        if "&le;" in first_line or "&gt;" in first_line:
            # This is a decision rule - make it bold
            parts[0] = f"<b>{parts[0]}</b>"

    # Remove values and gini from fist node
    # (the values are incorrectly calculated to be (0.5,0.5), not sure why)
    if node_id == 0:
        parts = [parts[0], parts[2]]

    # Process samples line - update with percentage
    for i, part in enumerate(parts):
        if "samples = " in part:
            # Replace the entire samples line (avoid partial matches)
            # parts[i] = f"samples = {sample_percentage:.1f}% (n={node_samples})"
            parts[i] = f"n={node_samples} ({sample_percentage:.1f}%)"
        elif "class = " in part:
            # Replace the entire samples line (avoid partial matches)
            parts[i] = parts[i].replace("class = ", "")

    # Reconstruct content and update line
    new_label = "<br/>".join(parts)
    line = line.replace(f"label=<{label_content}>", f"label=<{new_label}>")

    return line
