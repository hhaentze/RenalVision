"""Model training and inference utilities."""

import json
import pickle
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None  # type: ignore


class ModelType(Enum):
    LOGISTIC = "logistic"
    TREE = "tree"
    XGBOOST = "xgboost"


class ModelBundle:
    """
    Container for trained model, scaler, and metadata.
    """

    def __init__(
        self,
        model: Any,
        scaler: StandardScaler,
        feature_names: List[str],
        model_type: str,
        n_classes: int,
        class_names: Optional[List[str]] = None,
        log_transform_features: Optional[List[str]] = None,
    ):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.model_type = model_type
        self.n_classes = n_classes
        self.class_names = class_names if class_names else [f"Class {i}" for i in range(n_classes)]
        self.log_transform_features = log_transform_features or []

    def save(self, path: Union[str, Path]) -> None:
        """Save model bundle to pickle file and metadata to JSON."""
        path = Path(path)
        # 1. Save pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

        # 2. Save metadata JSON next to pickle
        metadata = {
            "n_classes": self.n_classes,
            "class_names": self.class_names,
            "feature_names": self.feature_names,
            "model_type": self.model_type,
            "log_transform_features": self.log_transform_features,
        }
        json_path = path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=4)

    @staticmethod
    def load(path: Union[str, Path]) -> "ModelBundle":
        """Load model bundle from pickle file."""
        path = Path(path)
        with open(path, "rb") as f:
            bundle = pickle.load(f)

        # Ensure metadata is synced if JSON exists
        json_path = path.with_suffix(".json")
        if json_path.exists():
            with open(json_path, "r") as f:
                metadata = json.load(f)
                bundle.class_names = metadata.get("class_names", bundle.class_names)
                bundle.n_classes = metadata.get("n_classes", bundle.n_classes)

        return bundle


class ModelFactory:
    """Factory to create and configure models dynamically."""

    @staticmethod
    def create_model(model_type: str, n_classes: int, **kwargs: Any) -> Any:
        if model_type == ModelType.LOGISTIC.value:
            return LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight="balanced",
            )

        elif model_type == ModelType.TREE.value:
            max_depth = kwargs.get("tree_max_depth", 5)
            return DecisionTreeClassifier(
                max_depth=max_depth,
                random_state=42,
                class_weight="balanced",
                min_samples_leaf=5,
            )

        elif model_type == ModelType.XGBOOST.value:
            if XGBClassifier is None:
                raise ImportError("XGBoost is not installed. Run `pip install xgboost`.")

            # XGBoost requires specific objectives for binary vs multi
            if n_classes == 2:
                objective = "binary:logistic"
                num_class = None
            else:
                objective = "multi:softprob"
                num_class = n_classes

            return XGBClassifier(
                n_estimators=100,
                max_depth=kwargs.get("tree_max_depth", 3),
                learning_rate=0.1,
                objective=objective,
                num_class=num_class,
                random_state=42,
                eval_metric="mlogloss",
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    n_classes: int,
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    tree_max_depth: int = 5,
    apply_log_transform: bool = True,
) -> ModelBundle:
    """
    Train a classifier on feature matrix.
    """
    X_processed = X.copy()
    log_transform_features: List[str] = []

    # Apply log transform to skewed features
    if apply_log_transform and feature_names:
        skewed_features = ["gradient_magnitude", "sphericity"]
        for i, fname in enumerate(feature_names):
            if fname in skewed_features:
                X_processed[:, i] = np.log1p(X_processed[:, i])
                log_transform_features.append(fname)

    # Initialize scaler
    scaler = StandardScaler()

    if model_type == ModelType.LOGISTIC.value:
        X_scaled = scaler.fit_transform(X_processed)
    else:
        scaler.fit(X_processed)
        X_scaled = X_processed

    # Create and train model
    model = ModelFactory.create_model(model_type, n_classes, tree_max_depth=tree_max_depth)
    model.fit(X_scaled, y)

    # Ensure feature_names is not None for bundle
    final_feature_names = feature_names if feature_names else []

    return ModelBundle(
        model=model,
        scaler=scaler,
        feature_names=final_feature_names,
        model_type=model_type,
        n_classes=n_classes,
        class_names=class_names,
        log_transform_features=log_transform_features,
    )


def _preprocess_for_inference(model_bundle: ModelBundle, X: np.ndarray) -> np.ndarray:
    """Helper to apply log transform and scaling."""
    X_processed = X.copy()

    if model_bundle.log_transform_features and model_bundle.feature_names:
        for i, fname in enumerate(model_bundle.feature_names):
            if fname in model_bundle.log_transform_features:
                X_processed[:, i] = np.log1p(X_processed[:, i])

    if model_bundle.model_type == ModelType.LOGISTIC.value:
        X_scaled = model_bundle.scaler.transform(X_processed)
    else:
        X_scaled = X_processed

    return X_scaled


def predict(model_bundle: ModelBundle, X: np.ndarray) -> np.ndarray:
    """Make predictions using trained model."""
    X_processed = _preprocess_for_inference(model_bundle, X)
    return model_bundle.model.predict(X_processed)


def predict_proba(model_bundle: ModelBundle, X: np.ndarray) -> np.ndarray:
    """Predict class probabilities."""
    X_processed = _preprocess_for_inference(model_bundle, X)
    return model_bundle.model.predict_proba(X_processed)


def compute_feature_correlations(X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
    """Compute pairwise feature correlations."""
    corr_matrix = np.corrcoef(X.T)
    high_corr_pairs = []
    n_features = len(feature_names)

    for i in range(n_features):
        for j in range(i + 1, n_features):
            corr_val = corr_matrix[i, j]
            if abs(corr_val) > 0.85:
                high_corr_pairs.append(
                    {
                        "feature1": feature_names[i],
                        "feature2": feature_names[j],
                        "correlation": float(corr_val),
                    }
                )

    return {
        "correlation_matrix": corr_matrix,
        "feature_names": feature_names,
        "high_correlations": high_corr_pairs,
    }
