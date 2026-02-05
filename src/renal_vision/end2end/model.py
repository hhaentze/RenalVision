from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import resnet50

from renal_vision.features.embeddings_fmcib import LoadModel, get_model_weights


def build_model(
    num_classes: int, fmcib_pretrained: bool = False, weights_path: Optional[str] = None
) -> nn.Module:
    if fmcib_pretrained:
        trunk = resnet50(
            pretrained=False,
            n_input_channels=1,
            widen_factor=2,
            conv1_t_stride=2,
            feed_forward=False,
            bias_downsample=True,
        )
        if not weights_path:
            weights_url = "https://zenodo.org/records/10528450/files/model_weights.torch?download=1"
            weights_path = get_model_weights(weights_url)
            model = LoadModel(
                trunk=trunk, weights_path=Path(weights_path), heads=[4096, 2048, num_classes]
            )
        else:
            model = LoadModel(
                trunk=trunk, weights_path=Path(weights_path), heads=[4096, 2048, num_classes]
            )

    else:
        trunk = resnet50(
            pretrained=not weights_path,
            n_input_channels=1,
            feed_forward=False,
            bias_downsample=False,
            num_classes=num_classes,
        )
        model = LoadModel(trunk=trunk, weights_path=weights_path, heads=[2048, 512, num_classes])

    return model


class TorchModelWrapper:
    """Adapter to make PyTorch models compatible with sklearn BaseEstimator interface"""

    def __init__(
        self,
        model_params: Dict[str, Any],
        weights_path: Union[str, Path],
    ) -> None:
        self.model_params = model_params
        self.weights_path = Path(weights_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {self.weights_path}")

        # Load model architecture and weights
        self._load_model()

    def _load_model(self) -> None:
        # Create fresh model instance
        self.model = build_model(**self.model_params, weights_path=str(self.weights_path)).to(
            self.device
        )
        self.model.eval()

    def _prepare_input(self, X: np.ndarray) -> torch.Tensor:
        """
        Convert numpy array to torch tensor on correct device.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(X).__name__}")

        if X.ndim < 3:
            raise ValueError(f"Expected at least 3D array, got {X.ndim}D. ")
        X_tensor = torch.from_numpy(X).float()

        return X_tensor.to(self.device)

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities with softmax."""
        X_tensor = self._prepare_input(X)

        # Forward pass through model (returns logits)
        logits = self.model(X_tensor)

        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)

        # Convert back to numpy on CPU
        return probs.cpu().numpy()

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels (argmax of probabilities)."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def __getstate__(self) -> Dict[str, Any]:
        """Prepare object for pickling."""

        return {
            "model_params": self.model_params,
            "weights_path": self.weights_path,
            "device": self.device,
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore object from pickle."""
        self.__dict__.update(state)
        self._load_model()

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return (
            f"TorchModelWrapper("
            f"weights_path={self.weights_path}, "
            f"device={self.device}, "
            f"model={self.model.__class__.__name__}"
            f")"
        )
