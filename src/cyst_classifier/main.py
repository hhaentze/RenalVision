"""Main entry point for Cyst classifier."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .explainability import (
    explain_decision_tree,
    explain_logistic_regression,
    explain_with_uncertainty,
)
from .features import Feature, extract_features
from .inference import Predictor
from .models import ModelBundle, compute_feature_correlations, predict, predict_proba, train_model
from .parser import create_parser, validate_args
from .preprocessing import CTPreprocessor, extract_lesions
from .utils import (
    apply_uncertainty_threshold,
    compute_metrics,
    compute_metrics_with_uncertainty,
    find_uncertainty_thresholds,
    plot_confusion_matrix,
    plot_confusion_matrix_with_unsure,
    plot_roc_curve,
    print_metrics_report,
)


def is_feature_csv(df):
    """
    Check if CSV contains pre-extracted features or raw image paths.

    Args:
        df: pandas DataFrame

    Returns:
        bool: True if CSV contains features, False if it contains image paths
    """
    # Feature CSV has: case, lesion_id, label, volume_voxels, and feature columns
    # Image CSV has: seg_path, image_path

    has_image_paths = "image_path" in df.columns and "seg_path" in df.columns
    has_features = "label" in df.columns and "case" in df.columns

    # Check for at least some feature columns
    feature_cols = ["mean_hu", "std_hu"]
    has_feature_cols = all(col in df.columns for col in feature_cols)

    if has_features and has_feature_cols:
        return True
    elif has_image_paths:
        return False
    else:
        raise ValueError(
            "Unrecognized CSV format. Expected either:\n"
            "  - Image CSV: columns 'image_path', 'seg_path'\n"
            "  - Feature CSV: columns 'case', 'lesion_id', 'label', and feature columns"
        )


def load_feature_data(df, feature_names):
    """
    Load pre-extracted features from CSV.

    Args:
        df: Feature DataFrame
        feature_names: List of feature column names to extract

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
    """
    # Check all requested features are present
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in CSV: {missing_features}")

    X = df[feature_names].values
    y = df["label"].values

    return X, y


def train_mode(args):
    """
    Train a classifier on labeled data.

    Supports two input formats:
    1. Image CSV (seg_path, image_path) - extracts features on-the-fly
    2. Feature CSV (pre-extracted features) - loads directly (much faster)
    """
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)

    # Determine which features to use
    if args.features:
        feature_list = [Feature[f.upper()] for f in args.features]
    else:
        feature_list = list(Feature)

    feature_names = [f.value for f in feature_list]
    print(f"Using {len(feature_names)} features: {feature_names}")

    # Auto-detect CSV format
    if is_feature_csv(df):
        print("\n✓ Detected pre-extracted feature CSV (fast mode)")
        X, y = load_feature_data(df, feature_names)

        print("\nDataset summary:")
        print(f"  Total lesions: {len(y)}")
        print(f"  Tumors (label=2): {np.sum(y == 2)}")
        print(f"  Cysts (label=3): {np.sum(y == 3)}")

    else:
        print("\n⊙ Detected image path CSV (extracting features...)")
        print("  Tip: Use extract_features_script.py to pre-extract features for faster training")

        # Extract features from images (original slow path)
        all_features = []
        all_labels = []

        print("\nExtracting features from lesions...")
        preprocessor = CTPreprocessor()
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            try:
                # Load and preprocess
                image, seg, _ = preprocessor.process_files(
                    row["image_path"],
                    row["seg_path"],
                )

                # Extract individual lesions
                lesions = extract_lesions(image, seg, min_voxels=args.min_voxels)

                # Compute features for each lesion
                for lesion_img, lesion_mask, label in lesions:
                    features = extract_features(lesion_img, lesion_mask, feature_list)

                    # Convert to feature vector
                    feature_vector = [features[f.value] for f in feature_list]
                    all_features.append(feature_vector)
                    all_labels.append(label)

            except Exception as e:
                print(f"\nWarning: Failed to process {row['image_path']}: {e}")
                continue

        if len(all_features) == 0:
            print("Error: No valid lesions found in dataset!")
            sys.exit(1)

        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(all_labels)

        print("\nDataset summary:")
        print(f"  Total lesions: {len(y)}")
        print(f"  Tumors (label=2): {np.sum(y == 2)}")
        print(f"  Cysts (label=3): {np.sum(y == 3)}")

    # Compute and print feature correlations
    print("\nComputing feature correlations...")
    corr_info = compute_feature_correlations(X, feature_names)

    if corr_info["high_correlations"]:
        print("\nHighly correlated feature pairs (|r| > 0.85):")
        for pair in corr_info["high_correlations"]:
            print(f"  {pair['feature1']} <-> {pair['feature2']}: r = {pair['correlation']:.3f}")
        print("\nConsider removing one feature from each highly correlated pair.")
    else:
        print("\nNo highly correlated features found.")

    # Train model
    print(f"\nTraining {args.model} model...")
    model_bundle = train_model(
        X,
        y,
        model_type=args.model,
        feature_names=feature_names,
        tree_max_depth=args.tree_depth if args.model == "tree" else 5,
    )

    # Save model
    output_dir = Path(args.output_dir)
    model_path = output_dir / "model.pkl"
    model_bundle.save(model_path)
    print(f"\nModel saved to {model_path}")

    # Generate explanations if requested
    if args.explain:
        explanation_dir = output_dir / "explanations"
        print("\nGenerating model explanations...")

        if model_bundle.model_type == "logistic":
            explain_logistic_regression(
                model_bundle, X, y, output_dir=explanation_dir, verbose=True
            )
        elif model_bundle.model_type == "tree":
            explain_decision_tree(
                model_bundle, output_dir=explanation_dir, verbose=True, rule_format=args.rule_format
            )

    # Print feature importance for tree
    if args.model == "tree":
        importances = model_bundle.model.feature_importances_
        print("\nFeature importances:")
        for fname, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
            print(f"  {fname}: {imp:.4f}")


def infer_mode(args):
    """
    Run inference on new data.

    Two modes:
    1. Single lesion: Return predicted class
    2. Multi-lesion: Return updated segmentation with predictions

    Note: Inference always requires actual images (not feature CSV).
    """
    print(f"Loading model from {args.model}...")
    predictor = Predictor(args.model)

    if args.multi_lesion:
        # Process all lesions
        print("Processing multiple lesions...")

        predictor.infer_mask(
            image=args.image,
            seg=args.seg,
            certainty_threshold=args.uncertainty_threshold,
            output=args.output,
        )

    else:
        # Single lesion mode
        print("Processing single lesion...")
        prediction = predictor.infer_lesion(
            image=args.image,
            seg=args.seg,
            certainty_threshold=args.uncertainty_threshold,
        )

        # Print result
        if prediction == -1:
            class_name = "Unsure"
        elif prediction == 0:
            class_name = "Tumor"
        elif prediction == 1:
            class_name = "Cyst"
        else:
            raise ValueError(f"Unknown predicted class: {prediction}")

        print(f"\nPrediction: {class_name}")


def eval_mode(args):
    """
    Evaluate model on test data.

    Supports two input formats:
    1. Image CSV - extracts features on-the-fly
    2. Feature CSV - loads directly (much faster)

    Two evaluation modes:
    1. --find-threshold: Analyze uncertainty thresholds (for validation)
    2. --uncertainty-threshold: Evaluate with specific threshold
    """
    print(f"Loading model from {args.model}...")
    model_bundle = ModelBundle.load(args.model)

    print(f"Loading test data from {args.data}...")
    df = pd.read_csv(args.data)

    # Get feature list from model
    feature_list = [Feature(fname) for fname in model_bundle.feature_names]
    feature_names = model_bundle.feature_names

    # Auto-detect CSV format
    if is_feature_csv(df):
        print("\n✓ Detected pre-extracted feature CSV (fast mode)")
        X, y_true = load_feature_data(df, feature_names)

        # Predict
        print("Running inference on test set...")
        y_proba = predict_proba(model_bundle, X)

        print("\nEvaluation summary:")
        print(f"  Total lesions: {len(y_true)}")
        print(f"  Tumors (label=2): {np.sum(y_true == 2)}")
        print(f"  Cysts (label=3): {np.sum(y_true == 3)}")

    else:
        print("\n⊙ Detected image path CSV (extracting features...)")
        print("  Tip: Use extract_features_script.py to pre-extract features for faster evaluation")

        # Extract features and predict (original slow path)
        all_labels = []
        all_probabilities = []

        print("\nRunning inference on test set...")
        preprocessor = CTPreprocessor()
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            try:
                # Load and preprocess
                image, seg, _ = preprocessor.process_files(
                    row["image_path"],
                    row["seg_path"],
                )

                # Extract lesions
                lesions = extract_lesions(image, seg, min_voxels=args.min_voxels)

                # Process each lesion
                for lesion_img, lesion_mask, label in lesions:
                    # Extract features
                    features = extract_features(lesion_img, lesion_mask, feature_list)
                    feature_vector = np.array([[features[f.value] for f in feature_list]])

                    # Predict
                    pred_proba = predict_proba(model_bundle, feature_vector)[0]

                    all_labels.append(label)
                    all_probabilities.append(pred_proba)

            except Exception as e:
                print(f"\nWarning: Failed to process {row['image_path']}: {e}")
                continue

        if len(all_labels) == 0:
            print("Error: No valid predictions made!")
            sys.exit(1)

        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_proba = np.array(all_probabilities)

        print("\nEvaluation summary:")
        print(f"  Total lesions: {len(y_true)}")
        print(f"  Tumors (label=2): {np.sum(y_true == 2)}")
        print(f"  Cysts (label=3): {np.sum(y_true == 3)}")

    # Create output directory
    output_dir = Path(args.output_dir)

    # Handle uncertainty analysis modes
    if args.find_threshold:
        # Threshold analysis mode
        print("\nAnalyzing uncertainty thresholds...")
        df_thresholds = find_uncertainty_thresholds(y_true, y_proba, output_dir)
        print("\nThreshold Analysis:")
        print(df_thresholds.to_string(index=False))

    else:
        # Standard evaluation with optional uncertainty threshold
        threshold = args.uncertainty_threshold
        use_uncertainty = threshold > 0.5

        if use_uncertainty:
            print(f"\nUsing uncertainty threshold: {threshold}")
            y_pred_with_unsure, _ = apply_uncertainty_threshold(y_proba, threshold)
            metrics = compute_metrics_with_uncertainty(y_true, y_pred_with_unsure, y_proba)

            # Print metrics
            print_metrics_report(metrics, exclude_unsure=True)

            # Save and plot ROC curve (only on certain predictions)
            roc_path = output_dir / "roc_curve.png"
            certain_mask = y_pred_with_unsure != 2
            if np.sum(certain_mask) > 0:
                plot_roc_curve(y_true[certain_mask], y_proba[certain_mask], str(roc_path))

            # Plot confusion matrix (2x2, excluding unsure)
            cm_path = output_dir / "confusion_matrix.png"
            plot_confusion_matrix_with_unsure(
                metrics["confusion_matrix_with_unsure"], str(cm_path), include_unsure=False
            )

            # Save metrics to text file
            metrics_path = output_dir / "metrics.txt"
            with open(metrics_path, "w") as f:
                f.write("CLASSIFICATION METRICS (excluding unsure)\n")
                f.write("=" * 50 + "\n")
                f.write(f"Uncertainty threshold: {threshold}\n\n")
                f.write(f"Accuracy:    {metrics['accuracy']:.4f}\n")
                f.write(f"F1 Score:    {metrics['f1']:.4f}\n")
                f.write(f"Sensitivity: {metrics['sensitivity']:.4f}\n")
                f.write(f"Specificity: {metrics['specificity']:.4f}\n")
                if metrics["auroc"] is not None:
                    f.write(f"AUROC:       {metrics['auroc']:.4f}\n")
                f.write(
                    f"\nCoverage:    {metrics['coverage']:.4f} ({metrics['n_certain']}/{metrics['n_certain'] + metrics['n_unsure']} certain)\n"
                )
                f.write(f"Unsure:      {metrics['n_unsure']} predictions\n\n")
                cm = metrics["confusion_matrix"]
                f.write("Confusion Matrix (certain predictions only):\n")
                f.write("                Predicted\n")
                f.write("              Tumor  Cyst\n")
                f.write(f"True Tumor    {cm[0, 0]:5d}  {cm[0, 1]:5d}\n")
                f.write(f"     Cyst     {cm[1, 0]:5d}  {cm[1, 1]:5d}\n\n")
                cm_full = metrics["confusion_matrix_with_unsure"]
                f.write("Full Confusion Matrix (including unsure):\n")
                f.write("                     Predicted\n")
                f.write("              Tumor  Cyst  Unsure\n")
                f.write(
                    f"True Tumor    {cm_full[0, 0]:5d}  {cm_full[0, 1]:5d}   {cm_full[0, 2]:5d}\n"
                )
                f.write(
                    f"     Cyst     {cm_full[1, 0]:5d}  {cm_full[1, 1]:5d}   {cm_full[1, 2]:5d}\n"
                )

        else:
            # Standard evaluation (no uncertainty)
            y_pred = predict(model_bundle, X) if is_feature_csv(df) else y_proba.argmax(axis=1)
            metrics = compute_metrics(y_true, y_pred, y_proba)

            # Print metrics
            print_metrics_report(metrics)

            # Save and plot ROC curve
            roc_path = output_dir / "roc_curve.png"
            plot_roc_curve(y_true, y_proba, str(roc_path))

            # Save and plot confusion matrix
            cm_path = output_dir / "confusion_matrix.png"
            plot_confusion_matrix(metrics["confusion_matrix"], str(cm_path))

            # Save metrics to text file
            metrics_path = output_dir / "metrics.txt"
            with open(metrics_path, "w") as f:
                f.write("CLASSIFICATION METRICS\n")
                f.write("=" * 50 + "\n")
                f.write(f"Accuracy:    {metrics['accuracy']:.4f}\n")
                f.write(f"F1 Score:    {metrics['f1']:.4f}\n")
                f.write(f"Sensitivity: {metrics['sensitivity']:.4f}\n")
                f.write(f"Specificity: {metrics['specificity']:.4f}\n")
                if metrics["auroc"] is not None:
                    f.write(f"AUROC:       {metrics['auroc']:.4f}\n")
                f.write("\n")
                cm = metrics["confusion_matrix"]
                f.write("Confusion Matrix:\n")
                f.write("                Predicted\n")
                f.write("              Tumor  Cyst\n")
                f.write(f"True Tumor    {cm[0, 0]:5d}  {cm[0, 1]:5d}\n")
                f.write(f"     Cyst     {cm[1, 0]:5d}  {cm[1, 1]:5d}\n")

        print(f"\nResults saved to {args.output_dir}")

        # Generate explanations if requested
        if args.explain:
            print("\nGenerating model explanations...")
            if use_uncertainty:
                # Uncertainty-aware explanations
                explain_with_uncertainty(model_bundle, y_proba, threshold, output_dir, verbose=True)
            else:
                # Standard explanations
                explanation_dir = output_dir / "explanations"
                if model_bundle.model_type == "logistic":
                    explain_logistic_regression(
                        model_bundle,
                        X if is_feature_csv(df) else None,
                        y_true,
                        output_dir=explanation_dir,
                        verbose=True,
                    )
                elif model_bundle.model_type == "tree":
                    explain_decision_tree(
                        model_bundle, output_dir=explanation_dir, verbose=True, rule_format="nested"
                    )


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Validate arguments
    try:
        validate_args(args)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Route to appropriate mode
    if args.mode == "train":
        train_mode(args)
    elif args.mode == "infer":
        infer_mode(args)
    elif args.mode == "eval":
        eval_mode(args)


if __name__ == "__main__":
    main()
