import numpy as np

from renal_vision.features.preprocessing import BasePreprocessor

# Adjust imports to match your actual package structure
from renal_vision.features.radiomics import RadiomicsExtractor


class FakePreprocessor(BasePreprocessor):
    """
    A dumb, reliable preprocessor that behaves exactly how we need
    for testing the Extractor's loop logic.
    """

    def __init__(self):
        super().__init__()

    def stream_components(self, image, seg, min_voxels=10):
        # Yields: (image_np, mask_np, metadata)
        img = np.zeros((50, 50, 50))
        seg = np.zeros((50, 50, 50))
        seg[10:20, 10:20, 10:20] = 1
        yield (img, seg, {"class_id": 1, "volume": 1000})

    def stream_augmented(self, image, seg, n_augmentations):
        # Always yield 1 original + n_augmentations
        image = self._prepare_data_point(image)
        seg = self._prepare_data_point(seg)
        yield image, seg, False
        for _ in range(n_augmentations):
            yield image, seg, True

    def get_config(self):
        return {"fake": True}


class TestRadiomicsExtractor:
    """Tests the specific PyRadiomics integration."""

    def test_radiomics_call_signature(self, mock_single_lesion):
        """
        Verify that our class correctly initializes the external library
        and calls it with the right data format.
        """
        img, seg = mock_single_lesion
        extractor = RadiomicsExtractor(FakePreprocessor())

        # Act
        results = extractor.extract(img, seg)

        # Assert
        assert len(results) == 1
        assert results[0]["original_shape_VoxelVolume"] == 1000
        assert results[0]["original_glszm_GrayLevelNonUniformityNormalized"] == 1

    def test_get_config(self):
        """Verify config returns serializable types."""
        extractor = RadiomicsExtractor()
        config = extractor.get_config()
        assert isinstance(config, dict)
        # Ensure it doesn't crash or return non-serializable objects
