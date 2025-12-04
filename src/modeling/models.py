"""
Model definitions, training logic, and persistence.
"""

import json
import pickle
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator
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
    Container for trained model, scaler, and full configuration.
    Acts as a self-contained unit for inference.
    """

    def __init__(
        self,
        model: BaseEstimator,
        scaler: Optional[StandardScaler],
        feature_names: List[str],
        model_type: str,
        n_classes: int,
        extractor_config: Dict[str, Any],
        class_names: Dict[int, str],
        log_transform_features: Optional[List[str]] = None,
    ):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.model_type = model_type
        self.n_classes = n_classes
        self.extractor_config = extractor_config
        self.class_names = class_names
        self.log_transform_features = log_transform_features or []

    def save(self, path: Union[str, Path]) -> None:
        """Save bundle to pickle and metadata to JSON for inspection."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self, f)

        # Save human-readable metadata
        metadata = {
            "model_type": self.model_type,
            "n_classes": self.n_classes,
            "class_names": self.class_names,
            "feature_names": self.feature_names,
            "extractor_config": self.extractor_config,
            "log_transform_features": self.log_transform_features,
        }
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(metadata, f, indent=4)

    @staticmethod
    def load(path: Union[str, Path]) -> "ModelBundle":
        with open(path, "rb") as f:
            return pickle.load(f)


class ModelFactory:
    """Factory to create and configure sklearn/xgboost models."""

    @staticmethod
    def create_model(model_type: str, n_classes: int, **kwargs: Any) -> BaseEstimator:
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

            # XGBoost objective handling
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


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    feature_names: List[str],
    class_names: Dict[int, str],
    extractor_config: Dict[str, Any],
    tree_max_depth: int = 5,
    apply_log_transform: bool = True,
) -> ModelBundle:
    """
    Train a classifier. Handles preprocessing (scaling/log) and ModelBundle creation.

    Args:
        X: Feature matrix (N_samples, N_features)
        y: Label vector (N_samples,)
        model_type: 'logistic', 'tree', or 'xgboost'
        feature_names: List of column names corresponding to X
        class_names: Mapping of int labels to string names
        extractor_config: Metadata from the feature extractor used to generate X
    """
    X_processed = X.copy()
    log_features: List[str] = []
    n_classes = len(np.unique(y))

    # 1. Log Transform (Logic preserved from original code)
    # We apply log to specific skewed features if they exist in the dataset
    if apply_log_transform:
        skewed_candidates = ["gradient_magnitude", "sphericity"]
        for i, fname in enumerate(feature_names):
            if fname in skewed_candidates:
                # log1p safely handles zeros
                X_processed[:, i] = np.log1p(X_processed[:, i])
                log_features.append(fname)

    # 2. Scaling
    # Tree models don't strictly need scaling, but it helps convergence for some implementations
    # and doesn't hurt. Logistic absolutely needs it.
    scaler = None
    if model_type == ModelType.LOGISTIC.value:
        scaler = StandardScaler()
        X_final = scaler.fit_transform(X_processed)
    else:
        # Optional: You could scale for trees too, but typically raw is fine.
        # Original code fit a scaler but didn't always use it.
        # Let's fit it if we want to support it later, or skip.
        # For strict parity with typical ML ops, we'll skip scaling for trees.
        X_final = X_processed

    # 3. Train
    model = ModelFactory.create_model(model_type, n_classes, tree_max_depth=tree_max_depth)
    model.fit(X_final, y)

    # 4. Bundle
    return ModelBundle(
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        model_type=model_type,
        n_classes=n_classes,
        extractor_config=extractor_config,
        class_names=class_names,
        log_transform_features=log_features,
    )


def _preprocess_for_inference(model_bundle: ModelBundle, X: np.ndarray) -> np.ndarray:
    """Helper to apply the saved log transforms and scaling to new data."""
    X_processed = X.copy()

    # 1. Apply Log Transform
    if model_bundle.log_transform_features:
        # We need to find the indices of these features.
        # This assumes X columns match model_bundle.feature_names order.
        # In inference.py we must ensure this alignment.
        for fname in model_bundle.log_transform_features:
            try:
                idx = model_bundle.feature_names.index(fname)
                X_processed[:, idx] = np.log1p(X_processed[:, idx])
            except ValueError:
                pass  # Feature not found (should not happen if pipeline is correct)

    # 2. Apply Scaling
    if model_bundle.scaler:
        X_processed = model_bundle.scaler.transform(X_processed)

    return X_processed


def predict_proba(model_bundle: ModelBundle, X: np.ndarray) -> np.ndarray:
    """Predict class probabilities (handles preprocessing internally)."""
    X_input = _preprocess_for_inference(model_bundle, X)
    return model_bundle.model.predict_proba(X_input)


def predict(model_bundle: ModelBundle, X: np.ndarray) -> np.ndarray:
    """Predict class labels (handles preprocessing internally)."""
    X_input = _preprocess_for_inference(model_bundle, X)
    return model_bundle.model.predict(X_input)
