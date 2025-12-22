"""
Training workflow logic.
Loads features, trains the classifier, and saves the ModelBundle.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from renal_vision.features.dataset import FeatureDatasetProcessor

from .models import train_classifier


def load_class_config(json_path: Optional[str]) -> Dict[int, str]:
    """Load class names from JSON (e.g. {'1': 'Tumor'})."""
    if not json_path:
        return {}

    with open(json_path, "r") as f:
        data = json.load(f)
        # Handle different json structures
        if "class_names" in data:
            mapping = data["class_names"]
        else:
            mapping = data

        return {int(k): str(v) for k, v in mapping.items()}


def run_training(
    data_path: str,
    output_dir: str,
    extractor_config_path: str,
    model_type: str = "logistic",
    class_config_path: Optional[str] = None,
    tree_max_depth: int = 5,
    class_column: str = "class_id",
) -> None:
    """
    Execute the training pipeline.

    Args:
        data_path: Path to the training features (Parquet/CSV).
        output_dir: Directory to save the model.
        model_type: 'logistic', 'tree', or 'xgboost'.
        class_config_path: JSON file defining class names (e.g. {1: 'Tumor'}).
        extractor_config_path: JSON file from features describing extraction settings.
        tree_max_depth: Hyperparameter for tree-based models.
    """
    print(f"Loading training data from {data_path}...")
    df = FeatureDatasetProcessor.load_features(data_path)

    # 1. Identify Feature Columns
    with open(extractor_config_path, "r") as f:
        extractor_config = json.load(f)
    feature_names = extractor_config["feature_names"]
    available_feature_names = [c for c in df.columns if c in feature_names]
    if len(feature_names) != len(available_feature_names):
        print(
            f"WARNING: Feature names do not match extractor config. Missing featues: {set(feature_names) - set(available_feature_names)}"
        )
    print(f"Detected {len(available_feature_names)} features: {available_feature_names}")

    # 2. Prepare X and y
    if class_column not in df.columns:
        raise ValueError(f"Input data missing {class_column} column.")
    X = df[available_feature_names].values
    y = df[class_column].values.astype(int)

    # 3. Resolve Class Names
    # If config provided, use it. Otherwise, generate generic names.
    unique_classes = [int(cl) for cl in np.unique(y)]
    n_classes = len(unique_classes)
    loaded_names = load_class_config(class_config_path)
    class_names: Dict[int, str] = {}

    for cls in unique_classes:
        if cls in loaded_names:
            class_names[cls] = loaded_names[cls]
        else:
            class_names[cls] = f"Class {cls}"
            print(f"No name provided for Class {cls}. Using default.")

    print(f"Training on {len(df)} samples ({n_classes} classes).")
    print(f"Class mapping: {class_names}")

    # 5. Train Model
    model_bundle = train_classifier(
        X=X,
        y=y,
        model_type=model_type,
        feature_names=available_feature_names,
        class_names=class_names,
        extractor_config=extractor_config,
        tree_max_depth=tree_max_depth,
        apply_log_transform=True,
    )

    # 6. Save
    out_path = Path(output_dir) / "model.pkl"
    model_bundle.save(out_path)
    print(f"Successfully saved model to {out_path}")
