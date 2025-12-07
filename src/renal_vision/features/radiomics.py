"""
Radiomics feature extractor implementation.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor

from .base import BaseFeatureExtractor
from .preprocessing import BasePreprocessor, CTPreprocessor


class RadiomicsExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        preprocessor: Optional[BasePreprocessor] = None,
        feature_names: Optional[List[str]] = None,
        min_voxels: int = 10,
    ) -> None:
        # 1. Define Preprocessor
        if preprocessor is None:
            preprocessor = CTPreprocessor(normalize=False)
        super().__init__(preprocessor, min_voxels)

        # 2. Configure PyRadiomics Extractor
        rad_settings = {
            "binWidth": 25.0,
            "resampledPixelSpacing": None,  # Assumes input is 1x1x1
            "interpolator": "sitkNearestNeighbor",
            "verbose": False,
        }
        self.engine = featureextractor.RadiomicsFeatureExtractor(**rad_settings)
        radiomics_logger = logging.getLogger("radiomics")
        radiomics_logger.setLevel(logging.ERROR)

        # 3. Enable standard classes (Lesion Classification Core Set)
        self.engine.disableAllFeatures()
        core_classes = ["shape", "firstorder", "glcm", "glrlm", "glszm"]
        for c in core_classes:
            self.engine.enableFeatureClassByName(c)

        # 4. Resolve _active_features to a concrete list
        if feature_names is None:
            self._active_features = self._get_all_possible_feature_names()
        else:
            self._active_features = feature_names

    def _get_all_possible_feature_names(self) -> list[str]:
        """
        Runs a dummy extraction on a tiny synthetic image to determine
        EXACTLY which keys this version of PyRadiomics returns.
        """
        # 1. Create a tiny synthetic image (5x5x5)
        # We use a simple pattern to ensure valid texture calculation
        size = (5, 5, 5)
        image = np.zeros(size, dtype=np.float32)
        mask = np.zeros(size, dtype=np.uint8)

        # Fill a small cube in the center
        image[1:4, 1:4, 1:4] = 100  # Set intensity
        mask[1:4, 1:4, 1:4] = 1  # Set label 1

        # 2. Convert to SimpleITK (PyRadiomics expects SITK)
        sitk_image = sitk.GetImageFromArray(image)
        sitk_mask = sitk.GetImageFromArray(mask)

        # 3. Run execution

        result = self.engine.execute(sitk_image, sitk_mask)

        # 4. Extract and Filter Keys
        feature_names = []
        for key in result.keys():
            # Exclude metadata/diagnostics
            if not key.startswith("diagnostics_"):
                feature_names.append(key)

        return sorted(feature_names)

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

    def _extract_single_lesion(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """
        Calculate radiomics for the isolated binary lesion mask.
        """

        if np.sum(mask) == 0:
            raise ValueError("Empty lesion mask provided for feature extraction.")

        # 1. Convert Numpy -> SimpleITK (Required by pyradiomics)
        sitk_img = sitk.GetImageFromArray(image)
        sitk_mask = sitk.GetImageFromArray(mask)

        # 2. Execute (returns an OrderedDict containing ~100 features)
        raw_results = self.engine.execute(sitk_img, sitk_mask)

        # 3. Filter & Return (only return the keys present in self._active_features)
        results = {}
        for key in self._active_features:
            if key in raw_results:
                results[key] = float(raw_results[key])
            else:
                # DEBUG: This will trigger for your error
                print(f"MISSING KEY: {key}")
                print(f"  Available keys: {list(raw_results.keys())}")
                print(f"  Image Max: {np.max(image)}, Min: {np.min(image)}")
                print(f"  Mask Sum: {np.sum(mask)}")

                # Assign NaN or 0.0 to prevent crash, or raise error
                results[key] = float("nan")

        return results
