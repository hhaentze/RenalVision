"""
Radiomics feature extractor implementation.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from monai.networks.nets import resnet50
from torch import nn

from .base import BaseFeatureExtractor
from .preprocessing import BasePreprocessor, StaticCropPreprocessor


# Helper function to manage the cache logic
def get_model_weights(url, filename="model_weights.torch", custom_cache_dir=None):
    """
    Resolves the path to the weights.
    Priority:
    1. Env Var (FMCIB_CACHE_DIR)
    2. Default User Cache (~/.cache/fmcib)
    """

    # Check if an environment variable is set (Best for Docker/Cluster)
    if "FMCIB_CACHE_DIR" in os.environ:
        cache_dir = Path(os.environ["FMCIB_CACHE_DIR"])
    elif custom_cache_dir:
        cache_dir = Path(custom_cache_dir)
    else:
        # Fallback to standard user cache (Linux: ~/.cache, Mac: ~/Library/Caches)
        # This ensures the file is shared across all your projects on this machine.
        cache_dir = Path.home() / ".cache" / "fmcib"

    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / filename

    if not destination.exists():
        print(f"Downloading weights to {destination}...")
        # Use torch's built-in downloader (handles progress bars & retries better than wget)
        torch.hub.download_url_to_file(url, str(destination))

    return destination


# Source - https://github.com/AIM-Harvard/foundation-cancer-image-biomarker/blob/master/fmcib/models/__init__.py
class LoadModel(nn.Module):
    """
    A class representing a loaded model.

    Args:
        trunk (nn.Module, optional): The trunk of the model. Defaults to None.
        weights_path (str, optional): The path to the weights file. Defaults to None.
        heads (list, optional): The list of head layers in the model. Defaults to [].

    Attributes:
        trunk (nn.Module): The trunk of the model.
        heads (nn.Sequential): The concatenated head layers of the model.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor: Forward pass through the model.
        load(weights): Load the pretrained model weights.
    """

    def __init__(self, trunk=None, weights_path=None, heads=[]) -> None:
        """
        Initialize the model.

        Args:
            trunk (optional): The trunk of the model.
            weights_path (optional): The path to the weights file.
            heads (list, optional): A list of layer sizes for the heads of the model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.trunk = trunk
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        head_layers = []
        for idx in range(len(heads) - 1):
            current_layers = []
            current_layers.append(nn.Linear(heads[idx], heads[idx + 1], bias=True))

            if idx != (len(heads) - 2):
                current_layers.append(nn.ReLU(inplace=True))  # type: ignore

            head_layers.append(nn.Sequential(*current_layers))

        if len(head_layers):
            self.heads = nn.Sequential(*head_layers)
        else:
            self.heads = nn.Identity()  # type: ignore

        if weights_path is not None:
            self.load(weights_path)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        out = self.trunk(x)
        out = self.heads(out)
        return out

    def load(self, weights):
        """
        Load pretrained model weights from a file.

        Args:
            weights (str): The path to the file containing the pretrained model weights.

        Raises:
            KeyError: If the input weights file does not contain the expected keys.
            Exception: If there is an error when loading the pretrained heads.

        Returns:
            None.

        Note:
            This function assumes that the pretrained model weights file is in the format expected by the model architecture.

        Warnings:
            - Missing keys: This warning message indicates the keys in the pretrained model weights file that are missing from the current model.
            - Unexpected keys: This warning message indicates the keys in the pretrained model weights file that are not expected by the current model.

        Raises the appropriate warnings and logs informational messages.
        """
        pretrained_model = torch.load(weights, map_location=self.device, weights_only=True)

        if "trunk_state_dict" in pretrained_model:  # Loading ViSSL pretrained model
            trained_trunk = pretrained_model["trunk_state_dict"]
            msg = self.trunk.load_state_dict(trained_trunk, strict=False)
            print(f"Model Trunk - Missing keys: {msg[0]} and unexpected keys: {msg[1]}")

        # Load trained heads
        if "head_state_dict" in pretrained_model:
            trained_heads = pretrained_model["head_state_dict"]

            try:
                msg = self.heads.load_state_dict(trained_heads, strict=False)
            except Exception as e:
                print(
                    f"Failed to load trained heads with error {e}. This is expected if the models do not match!"
                )
            print(f"Model Head - Missing keys: {msg[0]} and unexpected keys: {msg[1]}")

        # Loading Lighter and other pretrained model
        if "state_dict" in pretrained_model:
            trained_model = pretrained_model["state_dict"]

            # match the keys (https://github.com/Project-MONAI/MONAI/issues/6811)
            weights = {key.replace("module.", ""): value for key, value in trained_model.items()}
            weights = {
                key.replace("model.trunk.", ""): value for key, value in trained_model.items()
            }
            msg = self.trunk.load_state_dict(weights, strict=False)
            print(f"Model Trunk - Missing keys: {msg[0]} and unexpected keys: {msg[1]}")

            weights = {
                key.replace("model.heads.", ""): value
                for key, value in trained_model.items()
                if key.startswith("model.heads")
            }
            msg = self.heads.load_state_dict(weights, strict=False)
            print(f"Model Head - Missing keys: {msg[0]} and unexpected keys: {msg[1]}")

        print("Loaded pretrained model weights \n")


def fmcib_model(eval_mode: bool = True, weights_path: Optional[Path | str] = None):
    trunk = resnet50(
        pretrained=False,
        n_input_channels=1,
        widen_factor=2,
        conv1_t_stride=2,
        feed_forward=False,
        bias_downsample=True,
    )

    # If the user didn't provide a manual path, fetch from cache/URL
    if weights_path is None:
        weights_url = "https://zenodo.org/records/10528450/files/model_weights.torch?download=1"
        weights_path = get_model_weights(weights_url)

    # LoadModel logic here
    model = LoadModel(trunk=trunk, weights_path=Path(weights_path), heads=[])

    if eval_mode:
        model.eval()

    return model


class EmbeddingExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        preprocessor: Optional[BasePreprocessor] = None,
        min_voxels: int = 10,
    ) -> None:
        if preprocessor is None:
            preprocessor = StaticCropPreprocessor(normalize=True)

        super().__init__(preprocessor, min_voxels)

        self._active_features = [f"F{f}" for f in range(4096)]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = fmcib_model().to(device)

    @property
    def feature_names(self) -> List[str]:
        return self._active_features

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "radiomics",
            "feature_names": self.feature_names,
            "min_voxels": self.min_voxels,
            "preprocessor": self.preprocessor.get_config(),
        }

    def _extract_single_lesion(
        self, image: np.ndarray, lesion_mask: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate radiomics for the isolated binary lesion mask.
        """
        features: Dict[str, float] = {}

        # Safety check (should be caught by min_voxels, but good for robustness)
        if np.sum(lesion_mask) == 0:
            raise ValueError("Lesion mask is empty during feature extraction.")

        image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
        prediction = self.model(image_tensor)
        features = {
            fname: float(val)
            for fname, val in zip(self.feature_names, prediction.squeeze().tolist())
        }

        return features
