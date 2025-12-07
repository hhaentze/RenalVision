"""
Radiomics feature extractor implementation.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import ndimage
from scipy.stats import entropy as scipy_entropy
from skimage.feature import graycomatrix, graycoprops

from .base import BaseFeatureExtractor
from .preprocessing import BasePreprocessor, CTPreprocessor


class RadiomicsFeature(Enum):
    MEAN_HU = "mean_hu"
    STD_HU = "std_hu"
    COV = "coefficient_of_variation"
    P10 = "percentile_10"
    P90 = "percentile_90"
    ENTROPY = "entropy"
    GLCM_CONTRAST = "glcm_contrast"
    GRADIENT_MAG = "gradient_magnitude"
    SPHERICITY = "sphericity"
    FRAC_BELOW_20HU = "fraction_below_20hu"


class RadiomicsExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        preprocessor: Optional[BasePreprocessor] = None,
        feature_names: Optional[List[str]] = None,
        min_voxels: int = 10,
    ) -> None:
        # Default: preserve HU values (normalize=False)
        if preprocessor is None:
            preprocessor = CTPreprocessor(normalize=False)

        super().__init__(preprocessor, min_voxels)

        if feature_names:
            self._active_features = [RadiomicsFeature(f) for f in feature_names]
        else:
            self._active_features = list(RadiomicsFeature)

    @property
    def feature_names(self) -> List[str]:
        return [f.value for f in self._active_features]

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

        # The base class guarantees lesion_mask is boolean and specific to one component
        lesion_voxels = image[lesion_mask]

        # Safety check (should be caught by min_voxels, but good for robustness)
        if lesion_voxels.size == 0:
            raise ValueError("Lesion mask is empty during feature extraction.")

        for feature in self._active_features:
            if feature == RadiomicsFeature.MEAN_HU:
                features[feature.value] = float(np.mean(lesion_voxels))
            elif feature == RadiomicsFeature.STD_HU:
                features[feature.value] = float(np.std(lesion_voxels))
            elif feature == RadiomicsFeature.COV:
                mu = float(np.mean(lesion_voxels))
                features[feature.value] = float(np.std(lesion_voxels) / mu) if mu != 0 else 0.0
            elif feature == RadiomicsFeature.P10:
                features[feature.value] = float(np.percentile(lesion_voxels, 10))
            elif feature == RadiomicsFeature.P90:
                features[feature.value] = float(np.percentile(lesion_voxels, 90))
            elif feature == RadiomicsFeature.ENTROPY:
                hist, _ = np.histogram(lesion_voxels, bins=32, density=True)
                features[feature.value] = float(scipy_entropy(hist[hist > 0]))
            elif feature == RadiomicsFeature.GLCM_CONTRAST:
                features[feature.value] = self._compute_glcm_contrast(image, lesion_mask)
            elif feature == RadiomicsFeature.GRADIENT_MAG:
                features[feature.value] = self._compute_gradient_magnitude(image, lesion_mask)
            elif feature == RadiomicsFeature.SPHERICITY:
                features[feature.value] = self._compute_sphericity(lesion_mask)
            elif feature == RadiomicsFeature.FRAC_BELOW_20HU:
                features[feature.value] = float(np.sum(lesion_voxels < 20) / len(lesion_voxels))

        return features

    # --- Math Helpers ---
    # In src/features/radiomics.py (inside RadiomicsExtractor)

    @staticmethod
    def _get_bounding_box(mask: np.ndarray) -> tuple[slice, slice, slice]:
        """Calculates the bounding box slices for a 3D binary mask."""
        coords = np.argwhere(mask)
        if coords.size == 0:
            raise ValueError("Mask is empty")

        # Get min and max indices along each axis
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0) + 1

        # Return slices (z, y, x)
        return (
            slice(min_coords[0], max_coords[0]),
            slice(min_coords[1], max_coords[1]),
            slice(min_coords[2], max_coords[2]),
        )

    def _compute_glcm_contrast(self, image: np.ndarray, mask: np.ndarray) -> float:
        """
        Computes GLCM Contrast using a 3D approximation by averaging contrast
        over the three principal planes (XY, XZ, YZ) of the lesion's bounding box.
        """

        # Access parameters from the instantiated Preprocessor
        if not isinstance(self.preprocessor, CTPreprocessor):
            raise ValueError("Preprocessor must be CTPreprocessor for GLCM computation.")
        center = self.preprocessor.window_center
        width = self.preprocessor.window_width

        # 1. Define Fixed Quantization Range (HU values are fixed after clipping)
        lower_bound = center - width / 2.0

        # Define a Fixed Bin Width (e.g., 25 HU, which is a common value in radiomics)
        # Using a fixed width preserves the physical meaning of contrast.
        bin_width = 25.0

        bbox = RadiomicsExtractor._get_bounding_box(mask)
        cropped_image = image[bbox]
        cropped_mask = mask[bbox]

        # 2. Quantize the Cropped Volume using Fixed Bin Width
        # Calculate the number of bins based on the window width
        num_levels = int(np.ceil(width / bin_width))
        if num_levels < 2:
            num_levels = 2  # Ensure minimum 2 levels

        # Shift the image, divide by bin width, and clamp/cast to discrete levels
        quantized_volume = np.floor((cropped_image - lower_bound) / bin_width)

        # Clip the result to ensure it stays within [0, num_levels - 1]
        quantized_volume = np.clip(quantized_volume, 0, num_levels - 1).astype(np.uint8)

        contrast_values = []

        # 3. Iterate over 2D planes (Z, Y, X axes)
        for axis in range(3):
            num_slices = quantized_volume.shape[axis]

            for i in range(num_slices):
                # ... (slice extraction logic remains the same, using current_slice and current_mask) ...
                if axis == 0:  # XY plane (iterate through Z)
                    current_slice = quantized_volume[i, :, :]
                    current_mask = cropped_mask[i, :, :]
                elif axis == 1:  # XZ plane (iterate through Y)
                    current_slice = quantized_volume[:, i, :]
                    current_mask = cropped_mask[:, i, :]
                else:  # YZ plane (iterate through X)
                    current_slice = quantized_volume[:, :, i]
                    current_mask = cropped_mask[:, :, i]

                # Only process slices that contain the lesion
                if np.any(current_mask):
                    try:
                        # Compute GLCM on the 2D slice
                        glcm = graycomatrix(
                            current_slice,
                            distances=[1],
                            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                            levels=num_levels,  # Use the calculated level count
                            symmetric=True,
                            normed=True,
                        )
                        contrast = graycoprops(glcm, "contrast").mean()
                        contrast_values.append(contrast)
                    except Exception:
                        continue

        # 4. Average the contrast values
        if not contrast_values:
            return 0.0

        return float(np.mean(contrast_values))

    @staticmethod
    def _compute_gradient_magnitude(image: np.ndarray, mask: np.ndarray) -> float:
        grad = np.array(np.gradient(image))
        grad_mag = np.sqrt(np.sum(grad**2, axis=0))
        return float(np.mean(grad_mag[mask]))

    @staticmethod
    def _compute_sphericity(mask: np.ndarray) -> float:
        vol = np.sum(mask)
        if vol < 10:
            return 0.0
        # Simple approximation
        surface = np.sum(mask) - np.sum(ndimage.binary_erosion(mask))
        if surface == 0:
            return 0.0
        return float((np.pi ** (1 / 3) * (6 * vol) ** (2 / 3)) / surface)
