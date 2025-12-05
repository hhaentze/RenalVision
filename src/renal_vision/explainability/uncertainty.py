"""Uncertainty-aware explainability functions for evaluation mode."""

from pathlib import Path

import numpy as np

from .logistic import explain_logistic_regression
from .tree import explain_decision_tree


def explain_with_uncertainty(
    model_bundle, y_proba, uncertainty_threshold, output_dir, verbose=False
):
    """
    Generate uncertainty-aware explanations during evaluation.

    This function is called during evaluation mode when --explain flag is set
    and an uncertainty threshold is active. It generates explanations that
    incorporate uncertainty information.

    Args:
        model_bundle: Trained ModelBundle
        y_proba: Predicted probabilities from evaluation set (n_samples, 2)
        uncertainty_threshold: Threshold for marking predictions as uncertain
        output_dir: Directory to save explanations
        verbose: If True, print to console

    Returns:
        dict: Explanation dictionary with uncertainty information
    """
    output_dir = Path(output_dir)
    uncertainty_dir = output_dir / "uncertainty_explanations"
    uncertainty_dir.mkdir(parents=True, exist_ok=True)

    if model_bundle.model_type == "tree":
        # For trees: identify uncertain leaves and generate adapted visualization
        uncertain_leaves = _identify_uncertain_leaves(
            model_bundle.model, y_proba, uncertainty_threshold
        )

        explanation = explain_decision_tree(
            model_bundle,
            output_dir=uncertainty_dir,
            verbose=verbose,
            rule_format="nested",
            uncertain_leaves=uncertain_leaves,
        )

        # Add uncertainty guidance
        _save_tree_uncertainty_guide(
            uncertainty_dir,
            uncertainty_threshold,
            uncertain_leaves,
            model_bundle.model.get_n_leaves(),
        )

    elif model_bundle.model_type == "logistic":
        # For logistic: generate standard explanation + uncertainty guidance
        explanation = explain_logistic_regression(
            model_bundle,
            X=None,  # Don't recompute statistics
            y=None,
            output_dir=uncertainty_dir,
            verbose=verbose,
        )

        # Add uncertainty guidance
        _save_logistic_uncertainty_guide(
            uncertainty_dir, uncertainty_threshold, y_proba, explanation
        )

    else:
        raise ValueError(f"Unknown model type: {model_bundle.model_type}")

    if verbose:
        print(f"\nUncertainty-aware explanations saved to: {uncertainty_dir}")

    return explanation


def _identify_uncertain_leaves(tree_model, y_proba, threshold):
    """
    Identify which leaf nodes would produce uncertain predictions.

    For each leaf, if max(class_probability) < threshold, mark as uncertain.

    Args:
        tree_model: Trained DecisionTreeClassifier
        y_proba: Predicted probabilities (not used currently, for future enhancements)
        threshold: Uncertainty threshold

    Returns:
        set: Set of leaf node IDs that are uncertain
    """
    tree = tree_model.tree_
    uncertain_leaves = set()

    # Iterate through all nodes
    for node_id in range(tree.node_count):
        # Check if it's a leaf (feature == -2 means no split)
        if tree.feature[node_id] == -2:
            # Get class distribution at this leaf from training
            values = tree.value[node_id][0]
            total = sum(values)

            if total > 0:
                # Calculate confidence (probability of predicted class)
                max_prob = max(values) / total

                # Mark as uncertain if below threshold
                if max_prob < threshold:
                    uncertain_leaves.add(node_id)

    print(f"  Identified {len(uncertain_leaves)} uncertain leaves (out of {tree.n_leaves} total)")

    return uncertain_leaves


def _save_tree_uncertainty_guide(output_dir, threshold, uncertain_leaves, total_leaves):
    """Save uncertainty guidance for decision trees."""
    output_path = output_dir / "uncertainty_guide.txt"

    with open(output_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("UNCERTAINTY-AWARE DECISION TREE GUIDE\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Uncertainty Threshold: {threshold:.2f}\n")
        f.write(f"Uncertain Leaves: {len(uncertain_leaves)} / {total_leaves}\n\n")

        f.write("Interpretation:\n")
        f.write("-" * 70 + "\n")
        f.write(f"When the model's confidence (max class probability) is below {threshold:.2f},\n")
        f.write("the prediction is marked as 'Unsure'.\n\n")

        f.write(
            f"In this tree, {len(uncertain_leaves)} leaf nodes produce uncertain predictions.\n"
        )
        f.write("These correspond to regions in feature space where the training data\n")
        f.write("had mixed class labels, making confident classification difficult.\n\n")

        f.write("Clinical Implications:\n")
        f.write("-" * 70 + "\n")
        f.write("- Patients whose features lead to an uncertain leaf should be flagged\n")
        f.write("  for additional imaging or biopsy\n")
        f.write("- The decision tree paths show which feature combinations lead to\n")
        f.write("  uncertain predictions\n")
        f.write("- Consider collecting more training data for these ambiguous cases\n\n")

        f.write("Files Generated:\n")
        f.write("-" * 70 + "\n")
        f.write("- tree_with_uncertainty.png: Visualization showing decision structure\n")
        f.write("- decision_rules_uncertain.txt: Text rules with uncertainty markers\n")
        f.write("- tree_explanation.txt: Feature importance and tree statistics\n\n")


def _save_logistic_uncertainty_guide(output_dir, threshold, y_proba, explanation):
    """Save uncertainty guidance for logistic regression."""
    output_path = output_dir / "uncertainty_guide.txt"

    # Compute uncertainty statistics
    max_proba = y_proba.max(axis=1)
    uncertain_mask = max_proba < threshold
    n_total = len(y_proba)
    n_uncertain = np.sum(uncertain_mask)
    uncertainty_rate = n_uncertain / n_total if n_total > 0 else 0

    with open(output_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("UNCERTAINTY-AWARE LOGISTIC REGRESSION GUIDE\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Uncertainty Threshold: {threshold:.2f}\n")
        f.write("Predictions in Evaluation Set:\n")
        f.write(f"  Total: {n_total}\n")
        f.write(f"  Uncertain: {n_uncertain} ({uncertainty_rate * 100:.1f}%)\n")
        f.write(f"  Certain: {n_total - n_uncertain} ({(1 - uncertainty_rate) * 100:.1f}%)\n\n")

        f.write("Interpretation:\n")
        f.write("-" * 70 + "\n")
        f.write(f"The model predicts 'Unsure' when max(P(tumor), P(cyst)) < {threshold:.2f}.\n\n")

        f.write("This occurs when the logistic function output is close to 0.5,\n")
        f.write("indicating the weighted sum of features doesn't strongly favor\n")
        f.write("either class.\n\n")

        f.write("Features Contributing to Uncertainty:\n")
        f.write("-" * 70 + "\n")
        f.write("Uncertain predictions typically occur when:\n")
        f.write("1. Positive and negative feature contributions cancel out\n")
        f.write("2. Feature values are in ambiguous ranges\n")
        f.write("3. The lesion has mixed characteristics\n\n")

        f.write("Key Features (from most to least important):\n")
        sorted_indices = np.argsort(np.abs(explanation["coefficients"]))[::-1]
        for i, idx in enumerate(sorted_indices[:5], 1):
            fname = explanation["feature_names"][idx]
            coef = explanation["coefficients"][idx]
            direction = "increases cyst likelihood" if coef > 0 else "increases tumor likelihood"
            f.write(f"  {i}. {fname}: {direction}\n")
        f.write("\n")

        f.write("Clinical Implications:\n")
        f.write("-" * 70 + "\n")
        f.write(
            f"- With threshold {threshold:.2f}, approximately {uncertainty_rate * 100:.1f}% of cases\n"
        )
        f.write("  are flagged as uncertain\n")
        f.write("- These cases should undergo additional diagnostic procedures\n")
        f.write("- Consider adjusting threshold based on clinical risk tolerance:\n")
        f.write("  - Higher threshold (e.g., 0.80): Fewer errors, more uncertain cases\n")
        f.write("  - Lower threshold (e.g., 0.60): More coverage, higher error risk\n\n")

        f.write("Files Generated:\n")
        f.write("-" * 70 + "\n")
        f.write("- logistic_explanation.txt: Coefficients, odds ratios, p-values\n")
        f.write("- feature_importance.png: Bar chart of feature importance\n")
        f.write("- coefficients_ci.png: Coefficients with confidence intervals\n\n")
