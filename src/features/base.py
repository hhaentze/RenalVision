"""
Abstract base class for feature extraction.
Handles component analysis and orchestration.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import ndimage

from .preprocessing import BasePreprocessor


class BaseFeatureExtractor(ABC):
    """
    Interface for all feature extractors.

    Responsibilities:
    1. Preprocessing (via injected Preprocessor).
    2. Segmentation Parsing (finding/sorting connected components).
    3. Delegation (calling _extract_single_lesion for each component).
    """

    def __init__(self, preprocessor: BasePreprocessor, min_voxels: int = 10) -> None:
        self.preprocessor = preprocessor
        self.min_voxels = min_voxels

    def extract(
        self,
        image: Union[str, Any],  # str or Path or np.ndarray
        seg: Union[str, Any],
        augment: bool = False,
        affine: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Public interface: Preprocess -> Find Lesions -> Extract per Lesion.

        Returns:
            List of dictionaries (one per valid lesion found in the image).
        """
        # 1. Run coupled preprocessor
        img_arr, seg_arr, affine_arr = self.preprocessor(image, seg, augment=augment, affine=affine)

        # 2. Identify and sort all lesion components
        components = self._find_and_sort_components(seg_arr)

        results: List[Dict[str, Any]] = []

        # 3. Iterate over sorted lesions
        for lesion_id, (lesion_mask, class_id, volume) in enumerate(components, start=1):
            # Delegate specific math to the subclass
            feats = self._extract_single_lesion(img_arr, lesion_mask)

            # Calculate centroid
            cz, cy, cx = ndimage.center_of_mass(lesion_mask)
            voxel_coord = np.array([cx, cy, cz, 1.0])
            world_coord = affine_arr @ voxel_coord

            # Append metadata
            feats["lesion_id"] = lesion_id
            feats["class_id"] = class_id
            feats["volume_voxels"] = volume
            feats["centroid_world_x"] = world_coord[0]
            feats["centroid_world_y"] = world_coord[1]
            feats["centroid_world_z"] = world_coord[2]

            results.append(feats)

        return results

    def _find_and_sort_components(self, seg: np.ndarray) -> List[Tuple[np.ndarray, int, int]]:
        """
        Scans segmentation for connected components per class.

        Returns:
            List of tuples: (binary_mask, class_id, volume)
            Sorted by volume descending (Global Sort).
        """
        components = []

        # Get all unique classes (excluding background 0)
        classes = np.unique(seg)
        classes = classes[classes > 0]

        for c_id in classes:
            # Create binary mask for this class
            class_mask = seg == c_id

            # Find connected components
            labeled_mask, num_features = ndimage.label(class_mask)

            for f_id in range(1, num_features + 1):
                component_mask = labeled_mask == f_id
                volume = int(np.sum(component_mask))

                if volume >= self.min_voxels:
                    # Store tuple: (mask, class_id, volume)
                    components.append((component_mask, int(c_id), volume))

        # Global Sort: Sort all components by volume (descending)
        # This means Lesion 1 is the largest abnormality in the scan,
        # regardless of whether it is a cyst or tumor.
        components.sort(key=lambda x: x[2], reverse=True)

        return components

    @abstractmethod
    def _extract_single_lesion(self, image: np.ndarray, lesion_mask: np.ndarray) -> Dict[str, Any]:
        """
        Internal implementation.
        Extract features for a SINGLE specific binary lesion mask.

        Args:
            image: Full preprocessed image (D, H, W)
            lesion_mask: Binary mask of the specific lesion (D, H, W)
        """
        pass

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        pass
