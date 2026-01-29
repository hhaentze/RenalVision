"""
Abstract base class for feature extraction.
Handles component analysis and orchestration.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np

from .base_preprocessor import BasePreprocessor, ImageLike


class BaseFeatureExtractor(ABC):
    """
    Interface for all feature extractors.

    Responsibilities:
    1. Preprocessing (via injected Preprocessor).
    2. Segmentation Parsing (finding/sorting connected components).
    3. Delegation (calling _extract_single_lesion for each component).
    """

    def __init__(self, preprocessor: BasePreprocessor, min_volume: int = 400) -> None:
        self.preprocessor = preprocessor
        self.min_volume = min_volume

    def extract(
        self,
        image: ImageLike,
        seg: ImageLike,
        augment: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Public interface: Preprocess -> Find Lesions -> Extract per Lesion.

        Returns:
            List of dictionaries (one per valid lesion found in the image).
        """

        results: List[Dict[str, Any]] = []

        # 1. Run coupled preprocessor
        img_processed, seg_processed = self.preprocessor(image, seg, augment=augment)

        # 2. Iterate over all lesions
        component_stream = self.preprocessor.stream_components(
            img_processed, seg_processed, self.min_volume
        )
        for lesion_id, (img_comp, seg_comp, meta) in enumerate(component_stream):
            feats = self._extract_single_lesion(img_comp, seg_comp)
            feats["lesion_id"] = lesion_id
            feats["class_id"] = meta["class_id"] - 1
            feats["volume_voxels"] = meta["volume"]
            feats["augmented"] = augment
            feats["aug_id"] = 0
            results.append(feats)

        return results

    def extract_multiple_augmentations(
        self, image: ImageLike, seg: ImageLike, n_augmentations: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Returns features for original image + n segmentations.
        Has better load management compare to running extract n+1 times
        """

        results: List[Dict[str, Any]] = []

        # 1. Run coupled preprocessor
        img_processed, seg_processed = self.preprocessor(image, seg, augment=False)

        # 2. Iterate n+1 times over the image
        apply_augment = [False] + [True] * n_augmentations
        for augmentation_id, is_augmented in enumerate(apply_augment):
            component_stream = self.preprocessor.stream_components(
                img_processed, seg_processed, self.min_volume, augment=is_augmented
            )

            # 3. Extract all lesions
            for lesion_id, (img_comp, seg_comp, meta) in enumerate(component_stream):
                feats = self._extract_single_lesion(img_comp, seg_comp)
                feats["lesion_id"] = lesion_id
                feats["class_id"] = meta["class_id"] - 1
                feats["volume_voxels"] = meta["volume"]
                feats["augmented"] = is_augmented
                feats["aug_id"] = augmentation_id
                results.append(feats)

        return results

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
