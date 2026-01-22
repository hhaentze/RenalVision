"""
Mevis embeddings extractor implementation.
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn

from .base import BaseFeatureExtractor
from .preprocessing import BasePreprocessor, MevisPreprocessor

os.environ["MMM_LICENSE_ACCEPTED"] = "i accept"
from mmm.labelstudio_ext.NativeBlocks import DEFAULT_MODEL, MMM_MODELS, NativeBlocks


class MevisExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        preprocessor: Optional[BasePreprocessor] = None,
        min_voxels: int = 10,
    ) -> None:
        if preprocessor is None:
            preprocessor = MevisPreprocessor(normalize=True)

        super().__init__(preprocessor, min_voxels)

        self._active_features = [f"F{f}_max" for f in range(512)]
        self._active_features += [f"F{f}_mean" for f in range(512)]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = NativeBlocks(MMM_MODELS[DEFAULT_MODEL], device_identifier=device)

    @property
    def feature_names(self) -> List[str]:
        return self._active_features

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "MevisEmbeddings",
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

        prediction = self._extract_lesion_features(image, lesion_mask)
        features = {
            fname: float(val)
            for fname, val in zip(self.feature_names, prediction.squeeze().tolist())
        }

        return features

    def _extract_lesion_features(
        self, img: np.ndarray, seg: np.ndarray, step_size: int = 3
    ) -> np.ndarray:
        """
        img, seg: numpy arrays of shape (H, W, D)
        model: The MedicalMultitaskingModel (UMedPT) encoder
        n_step: Distance between slab centers
        """

        depth = img.shape[-1]
        embeddings = []
        weights = []

        with torch.inference_mode():
            # Iterate through the depth dimension
            # We use a sliding window of 3 to create the "RGB" style input
            for z in range(1, depth - 1, step_size):
                # Extract 3 consecutive slices: (i-1, i, i+1)
                # Reshape from (224, 224, 3) -> (1, 3, 224, 224) to match 2D model input
                slab = torch.tensor(img[..., z - 1 : z + 2].transpose(2, 0, 1)[None])

                # Extract features
                feature_pyramid: list[torch.Tensor] = self.model["encoder"](
                    slab.to(self.model.device)
                )
                feature_vector = nn.Flatten(1)(self.model["squeezer"](feature_pyramid)[1])
                embeddings.append(feature_vector)

                # Calculate the mask weight: How much of this slab is actually lesion?
                # We take the sum of the mask in these 3 slices, add 10 to give a minor weight to the margins
                mask_sum = seg[..., z - 1 : z + 2].sum() + 10
                weights.append(mask_sum)

        # Convert lists to tensors
        embeddings_tensor = torch.stack(embeddings)  # Shape: (NumSlabs, 1, FeatureDim)
        weights_tensor = torch.tensor(weights).view(-1, 1, 1)  # Shape: (NumSlabs, 1, 1)

        # 1. Simple Max Pooling
        max_pooled = torch.max(embeddings_tensor, dim=0).values

        # 2. Mask-Weighted Average Pooling
        # Avoid division by zero if mask is empty
        total_weight = weights_tensor.sum() + 1e-6
        weighted_avg = (embeddings_tensor * weights_tensor).sum(dim=0) / total_weight
        combined = torch.cat([max_pooled, weighted_avg], dim=-1)

        return combined.squeeze().numpy()
