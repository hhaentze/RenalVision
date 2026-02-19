"""
Evaluation workflow logic.
Loads a trained model and test features, calculates metrics, and saves reports.
"""

from pathlib import Path
from typing import Any, Dict

from renal_vision.features.dataset import FeatureDatasetProcessor
from renal_vision.modeling.models import ModelBundle, predict_proba
from renal_vision.shared.metrics import ModelEvaluator


def run_evaluation(
    data_path: str,
    model_path: str,
    output_dir: str,
    class_column: str = "class_id",
    verbose: bool = True,
    return_preds: bool = False,
) -> Dict[str, Any]:
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
    if verbose:
        print(f"Loading test data from {data_path}...")
    df = FeatureDatasetProcessor.load_features(data_path)

    if verbose:
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
    if class_column not in df.columns:
        raise ValueError(f"Input data missing {class_column} column.")
    y_true = df[class_column].values.astype(int)

    # 3. Predict
    # predict_proba handles the internal scaling/log-transforms
    if verbose:
        print("Running predictions...")
    y_proba = predict_proba(model_bundle, X)
    y_pred = y_proba.argmax(axis=1)

    # 4. Compute Metrics
    # We reconstruct the list of class names based on the model's knowledge
    # The model stores {int: str}, we need a list [str] sorted by index
    sorted_class_indices = sorted(model_bundle.class_names.keys())
    class_names_list = [model_bundle.class_names[i] for i in sorted_class_indices]

    evaluator = ModelEvaluator(y_true, y_proba, class_names=class_names_list)

    scalar_df = evaluator.get_scalars()
    metrics = {}
    metrics["f1_macro"] = scalar_df.loc["macro avg"]["f1-score"]
    metrics["scalar_df"] = scalar_df
    # 5. Print & Save Report
    if verbose:
        print("\n" + "=" * 40)
        print(f"F1 Score: {metrics['f1_macro']:.4f}")

        # 6. Generate Plots
        evaluator.plot_cm(output_path=output_path / "confusion_matrix.png")
        evaluator.plot_roc(output_path=output_path / "roc_curves.png")
        evaluator.plot_pr(output_path=output_path / "pr_curves.png")
        print(f"Results saved to {output_dir}")

    if return_preds:
        pred_df = df.drop(model_bundle.feature_names, axis=1)
        pred_df["y_true"] = y_true
        pred_df["y_proba"] = list(y_proba)
        for cl in range(len(class_names_list)):
            pred_df[f"y_{cl}_proba"] = [p[cl] for p in y_proba]
        pred_df["y_pred"] = y_pred
        metrics["pred_df"] = pred_df

    return metrics
