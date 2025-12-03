"""
Evaluation workflow logic.
Loads a trained model and test features, calculates metrics, and saves reports.
"""

import json
from pathlib import Path

import numpy as np

from classifier.models import ModelBundle, predict_proba
from feature_handler.dataset import FeatureDatasetProcessor
from shared.metrics import (
    compute_metrics,
    plot_confusion_matrix,
    plot_multiclass_roc,
)


def run_evaluation(
    data_path: str,
    model_path: str,
    output_dir: str,
) -> None:
    """
    Execute the evaluation pipeline.

    Args:
        data_path: Path to the test features (Parquet/CSV).
        model_path: Path to the trained model.pkl.
        output_dir: Directory to save metrics and plots.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Data & Model
    print(f"Loading test data from {data_path}...")
    df = FeatureDatasetProcessor.load_features(data_path)

    print(f"Loading model from {model_path}...")
    model_bundle = ModelBundle.load(model_path)

    # 2. Align Features
    # Ensure we strictly use the features the model expects
    missing_features = [f for f in model_bundle.feature_names if f not in df.columns]
    if missing_features:
        raise ValueError(f"Test data is missing features required by model: {missing_features}")

    # Extract X (in the correct order) and y
    X = df[model_bundle.feature_names].values

    # Target column check
    target_col = "class_id" if "class_id" in df.columns else "label"
    if target_col not in df.columns:
        raise ValueError("Test data missing 'class_id' or 'label' column.")
    y_true = df[target_col].values.astype(int)

    # 3. Predict
    # predict_proba handles the internal scaling/log-transforms
    print("Running predictions...")
    y_proba = predict_proba(model_bundle, X)
    y_pred = y_proba.argmax(axis=1)

    # 4. Compute Metrics
    # We reconstruct the list of class names based on the model's knowledge
    # The model stores {int: str}, we need a list [str] sorted by index
    sorted_class_indices = sorted(model_bundle.class_names.keys())
    class_names_list = [model_bundle.class_names[i] for i in sorted_class_indices]

    metrics = compute_metrics(
        y_true=y_true, y_pred=y_pred, y_proba=y_proba, class_names=class_names_list
    )

    # 5. Print & Save Report
    print("\n" + "=" * 40)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("=" * 40)
    print("Classification Report:")
    print(metrics["report_str"])

    # Save raw metrics
    # Convert numpy types to native python for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path / "metrics.json", "w") as f:
        json.dump(metrics, f, default=convert_numpy, indent=4)

    # 6. Generate Plots
    # Confusion Matrix
    if "confusion_matrix" in metrics:
        plot_confusion_matrix(
            cm=np.array(metrics["confusion_matrix"]),
            class_names=class_names_list,
            output_path=str(output_path / "confusion_matrix.png"),
        )

    # ROC Curves
    # Note: plot_multiclass_roc handles both binary and multi-class logic
    plot_multiclass_roc(
        y_true=y_true,
        y_proba=y_proba,
        n_classes=model_bundle.n_classes,
        class_names=class_names_list,
        output_path=str(output_path / "roc_curves.png"),
    )

    print(f"Results saved to {output_dir}")
