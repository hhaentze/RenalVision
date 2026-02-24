"""
Feature extractor implementation using CTFM embeddings.
"""

# Imports
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from lighter_zoo import SegResEncoder

from .base_extractor import BaseFeatureExtractor
from .base_preprocessor import BasePreprocessor
from .preprocessing import CTPreprocessor


class CTFMExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        preprocessor: Optional[BasePreprocessor] = None,
        feature_names: Optional[List[str]] = None,
        min_volume: int = 400,
    ) -> None:
        if preprocessor is None:
            preprocessor = CTPreprocessor()

        super().__init__(preprocessor, min_volume)

        if feature_names is None:
            self._active_features = [f"F{f}" for f in range(512)]
        else:
            self._active_features = feature_names

        self.model = SegResEncoder.from_pretrained("project-lighter/ct_fm_feature_extractor")
        self.model.eval()

    @property
    def feature_names(self) -> List[str]:
        return self._active_features

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "CTFM_embeddings",
            "feature_names": self.feature_names,
            "min_volume": self.min_volume,
            "preprocessor": self.preprocessor.get_config(),
        }

    def _extract_single_lesion(
        self, image: np.ndarray, lesion_mask: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate radiomics for the isolated binary lesion mask.
        """
        features: Dict[str, float] = {}

        # Safety check (should be caught by min_volume, but good for robustness)
        if np.sum(lesion_mask) == 0:
            raise ValueError("Lesion mask is empty during feature extraction.")

        image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = self.model(image_tensor)[-1]
            # Average pooling compressed the feature vector across all patches. If this is not desired, remove this line and
            # use the output tensor directly which will give you the feature maps in a low-dimensional space.
            prediction = torch.nn.functional.adaptive_avg_pool3d(output, 1).squeeze()

        features = {
            fname: float(val)
            for fname, val in zip(self.feature_names, prediction.squeeze().tolist())
        }

        return features
