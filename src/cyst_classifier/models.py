"""Model training and inference utilities."""

import pickle
from typing import Any, Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


class ModelBundle:
    """
    Container for trained model, scaler, and metadata.

    Attributes:
        model: Trained sklearn classifier
        scaler: StandardScaler for feature normalization
        feature_names: List of feature names used
        model_type: 'logistic' or 'tree'
        log_transform_features: Features that were log-transformed
    """

    def __init__(self, model, scaler, feature_names, model_type, log_transform_features=None):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.model_type = model_type
        self.log_transform_features = log_transform_features or []

    def save(self, path):
        """Save model bundle to pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """Load model bundle from pickle file."""
        with open(path, "rb") as f:
            return pickle.load(f)


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "logistic",
    feature_names: List[str] = None,
    tree_max_depth: int = 5,
    apply_log_transform: bool = True,
) -> ModelBundle:
    """
    Train a classifier on feature matrix.

    Preprocessing:
    - Optionally log-transform highly skewed features (gradient_magnitude, sphericity)
    - Z-score standardization for logistic regression
    - No standardization for decision tree

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,) - 0 for cyst, 1 for tumor (or 2=tumor, 3=cyst)
        model_type: 'logistic' or 'tree'
        feature_names: List of feature names
        tree_max_depth: Maximum depth for decision tree
        apply_log_transform: Whether to log-transform skewed features

    Returns:
        ModelBundle: Trained model with scaler and metadata
    """
    # Convert labels: 2 -> 0 (tumor), 3 -> 1 (cyst), or keep as 0/1
    if np.min(y) > 1:
        y_binary = np.where(y == 2, 0, 1)  # 2=tumor->0, 3=cyst->1
    else:
        y_binary = y

    X_processed = X.copy()
    log_transform_features = []

    # Apply log transform to skewed features
    if apply_log_transform and feature_names:
        skewed_features = ["gradient_magnitude", "sphericity"]
        for i, fname in enumerate(feature_names):
            if fname in skewed_features:
                # Log(x + 1) to handle zeros
                X_processed[:, i] = np.log1p(X_processed[:, i])
                log_transform_features.append(fname)

    # Initialize scaler
    scaler = StandardScaler()

    if model_type == "logistic":
        # Standardize features for logistic regression
        X_scaled = scaler.fit_transform(X_processed)

        # Train logistic regression
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced",  # Handle class imbalance
        )
        model.fit(X_scaled, y_binary)

    elif model_type == "tree":
        # Decision tree doesn't need scaling, but we fit scaler for consistency
        scaler.fit(X_processed)
        X_scaled = X_processed

        # Train decision tree
        model = DecisionTreeClassifier(
            max_depth=tree_max_depth,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=5,
        )
        model.fit(X_scaled, y_binary)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return ModelBundle(
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        model_type=model_type,
        log_transform_features=log_transform_features,
    )


def predict(model_bundle: ModelBundle, X: np.ndarray) -> np.ndarray:
    """
    Make predictions using trained model.

    Args:
        model_bundle: Trained ModelBundle
        X: Feature matrix (n_samples, n_features)

    Returns:
        predictions: Predicted class labels (0=tumor, 1=cyst)
    """
    X_processed = X.copy()

    # Apply same log transform as training
    if model_bundle.log_transform_features and model_bundle.feature_names:
        for i, fname in enumerate(model_bundle.feature_names):
            if fname in model_bundle.log_transform_features:
                X_processed[:, i] = np.log1p(X_processed[:, i])

    # Apply scaling
    if model_bundle.model_type == "logistic":
        X_scaled = model_bundle.scaler.transform(X_processed)
    else:
        X_scaled = X_processed

    return model_bundle.model.predict(X_scaled)


def predict_proba(model_bundle: ModelBundle, X: np.ndarray) -> np.ndarray:
    """
    Predict class probabilities.

    Args:
        model_bundle: Trained ModelBundle
        X: Feature matrix (n_samples, n_features)

    Returns:
        probabilities: Predicted probabilities (n_samples, 2)
                      [:, 0] = P(tumor), [:, 1] = P(cyst)
    """
    X_processed = X.copy()

    # Apply same log transform as training
    if model_bundle.log_transform_features and model_bundle.feature_names:
        for i, fname in enumerate(model_bundle.feature_names):
            if fname in model_bundle.log_transform_features:
                X_processed[:, i] = np.log1p(X_processed[:, i])

    # Apply scaling
    if model_bundle.model_type == "logistic":
        X_scaled = model_bundle.scaler.transform(X_processed)
    else:
        X_scaled = X_processed

    return model_bundle.model.predict_proba(X_scaled)


def compute_feature_correlations(X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
    """
    Compute pairwise feature correlations.

    Args:
        X: Feature matrix (n_samples, n_features)
        feature_names: List of feature names

    Returns:
        Dictionary with correlation matrix and highly correlated pairs
    """
    corr_matrix = np.corrcoef(X.T)

    # Find highly correlated pairs (|r| > 0.85)
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
                        "correlation": corr_val,
                    }
                )

    return {
        "correlation_matrix": corr_matrix,
        "feature_names": feature_names,
        "high_correlations": high_corr_pairs,
    }
