from typing import Any, Dict, List

import numpy as np

from renal_vision.features.base_extractor import BaseFeatureExtractor
from renal_vision.features.base_preprocessor import BasePreprocessor


class FakePreprocessor(BasePreprocessor):
    def __init__(self):
        pass

    def __call__(self, image, seg, augment=False):
        # Identity function: just return inputs as-is
        image = self._prepare_data_point(image)
        seg = self._prepare_data_point(seg)
        return image, seg

    def stream_components(self, image, seg, min_volume=400, augment=False):
        # Always yield exactly 2 dummy components
        # Yields: (image_np, mask_np, metadata)
        yield np.ones((10, 10, 10)), np.ones((10, 10, 10)), {"class_id": 1, "volume": 1000}
        yield np.ones((10, 10, 10)), np.ones((10, 10, 10)), {"class_id": 1, "volume": 1000}

    def get_config(self):
        return {"fake": True}


class DummyExtractor(BaseFeatureExtractor):
    """
    A lightweight concrete class to test BaseFeatureExtractor logic
    without relying on external libraries like PyRadiomics.
    """

    def _extract_single_lesion(self, image: np.ndarray, lesion_mask: np.ndarray) -> Dict[str, Any]:
        # Simply return the sum of pixels as a "feature" to prove we ran
        return {"dummy_feature": float(image[lesion_mask == 1].sum())}

    @property
    def feature_names(self) -> List[str]:
        return ["dummy_feature"]

    def get_config(self) -> Dict[str, Any]:
        return {"test_mode": True}


# ==============================================================================
# 2. Testing the Orchestration (BaseFeatureExtractor)
# ==============================================================================


class TestBaseFlow:
    """Tests the loop logic, error handling, and augmentation delegation."""

    def test_extract_orchestration(self, mock_two_lesions):
        """
        Verify 'extract' calls the preprocessor and iterates over all results.
        """
        img, seg = mock_two_lesions
        extractor = DummyExtractor(FakePreprocessor())

        # Act
        results = extractor.extract(img, seg)

        # Assert
        assert len(results) == 2, "Should have extracted features for 2 components"
        assert results[0]["dummy_feature"] > 0

    def test_extract_includes_class_id(self, mock_single_lesion):
        """Verify that class_id (-1) from the preprocessor is preserved in output."""
        img, seg = mock_single_lesion
        extractor = DummyExtractor(FakePreprocessor())
        results = extractor.extract(img, seg)
        assert results[0]["class_id"] == 0, (
            "Class Id in features should be one integer below preprocessing id"
        )

        assert "class_id" in results[0], "Output should inherit metadata from preprocessor"

    def test_augmentations_flow(self, mock_two_lesions):
        """
        Verify extract_multiple_augmentations loops over N augments + 1 original.
        """
        img, seg = mock_two_lesions
        extractor = DummyExtractor(FakePreprocessor())

        # Act
        results = extractor.extract_multiple_augmentations(img, seg, n_augmentations=2)

        # Assert
        # Logic: 3 streams * 2 components per stream = 6 total feature dicts
        assert len(results) == 6
