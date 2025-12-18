import numpy as np
import pytest
from monai.data import MetaTensor

from renal_vision.features.preprocessing import (
    CropPreprocessor,
    CTPreprocessor,
    MevisCropPreprocessor,
    StaticCropPreprocessor,
)

# ==========================================
# 1. Happy Path Tests
# ==========================================


class TestCTPreprocessor:
    """Tests for the standard CT Preprocessor."""

    def test_call_structure_and_types(self, mock_single_lesion):
        """Verify __call__ returns MetaTensors of correct dimensionality."""
        image, seg = mock_single_lesion
        preprocessor = CTPreprocessor(normalize=True)

        # Act
        out_img, out_seg = preprocessor(image, seg)

        # Assert Types
        assert isinstance(out_img, MetaTensor)
        assert isinstance(out_seg, MetaTensor)

        # Assert Shapes (Should preserve channel dim)
        # Note: Exact spatial shape might change due to spacing, but dims should match
        assert out_img.ndim == 4  # (D, H, W)
        assert out_seg.ndim == 4

    def test_augmentation_stream_count(self, mock_single_lesion):
        """Verify stream_augmented yields original + N augmentations."""
        image, seg = mock_single_lesion
        preprocessor = CTPreprocessor()
        n_aug = 3

        # Act
        generator = preprocessor.stream_augmented(image, seg, n_augmentations=n_aug)
        results = list(generator)

        # Assert
        # Should yield: 1 original + 3 augmented = 4 total
        assert len(results) == n_aug + 1

        # Check first item is not augmented
        _, _, is_aug_first = results[0]
        assert is_aug_first is False

        # Check subsequent items are augmented
        _, _, is_aug_second = results[1]
        assert is_aug_second is True


class TestCropPreprocessors:
    """Tests for Crop and StaticCrop Preprocessors."""

    @pytest.mark.parametrize("processor_class", [CropPreprocessor, StaticCropPreprocessor])
    def test_stream_components_shape_contract_fmcib(self, processor_class, mock_two_lesions):
        """
        CRITICAL: Ensure stream_components yields numpy arrays of shape 50x50x50.
        This validates the specific requirement for feature extraction input.
        """
        image, seg = mock_two_lesions
        preprocessor = processor_class()

        # Act
        # We expect 2 components based on the fixture data
        components = list(preprocessor.stream_components(image, seg))

        # Assert
        assert len(components) == 2, "Should detect the 2 synthetic lesions created in fixture"

        for comp_img, comp_mask, metadata in components:
            # 1. Check Return Types
            assert isinstance(comp_img, np.ndarray), "Stream should return numpy array"
            assert isinstance(comp_mask, np.ndarray), "Stream should return numpy array"
            assert isinstance(metadata, dict), "Metadata should be a dictionary"

            # 2. Check Strict Shape Requirement (50x50x50)
            expected_shape = (50, 50, 50)
            assert comp_img.shape == expected_shape, (
                f"Expected image shape {expected_shape}, but got {comp_img.shape}"
            )
            assert comp_mask.shape == expected_shape, (
                f"Expected mask shape {expected_shape}, but got {comp_mask.shape}"
            )

            # 3. Check Metadata keys
            assert "class_id" in metadata
            assert "volume" in metadata

    @pytest.mark.parametrize("processor_class", [MevisCropPreprocessor])
    def test_stream_components_shape_contract_mevis(self, processor_class, mock_two_lesions):
        """
        CRITICAL: Ensure stream_components yields numpy arrays of shape 50x50x50.
        This validates the specific requirement for feature extraction input.
        """
        image, seg = mock_two_lesions
        preprocessor = processor_class()

        # Act
        # We expect 2 components based on the fixture data
        components = list(preprocessor.stream_components(image, seg))

        # Assert
        assert len(components) == 2, "Should detect the 2 synthetic lesions created in fixture"

        for comp_img, comp_mask, metadata in components:
            # 1. Check Return Types
            assert isinstance(comp_img, np.ndarray), "Stream should return numpy array"
            assert isinstance(comp_mask, np.ndarray), "Stream should return numpy array"
            assert isinstance(metadata, dict), "Metadata should be a dictionary"

            # 2. Check Strict Shape Requirement (224,224,z)
            expected_shape = 224
            assert (comp_img.shape[0] == expected_shape) and (
                comp_img.shape[1] == expected_shape
            ), f"Expected image x and y of size {expected_shape}, but got shape {comp_img.shape}"
            assert (comp_mask.shape[0] == expected_shape) and (
                comp_mask.shape[1] == expected_shape
            ), f"Expected mask x and y of size {expected_shape}, but got {comp_mask.shape}"

            # 3. Check Metadata keys
            assert "class_id" in metadata
            assert "volume" in metadata


# ==========================================
# 2. Edge Case Tests
# ==========================================


class TestEdgeCases:
    ALL_PREPROCESSORS = [CTPreprocessor, CropPreprocessor, StaticCropPreprocessor]

    @pytest.mark.parametrize("preprocessor_class", ALL_PREPROCESSORS)
    def test_empty_segmentation_handling(self, preprocessor_class, mock_no_lesion):
        """
        Edge Case: If segmentation is empty (all zeros),
        stream_components should yield nothing without crashing.
        """
        image, seg = mock_no_lesion
        preprocessor = preprocessor_class()

        # Act
        components = list(preprocessor.stream_components(image, seg))

        # Assert
        assert len(components) == 0, (
            f"{preprocessor_class.__name__} failed: Should yield 0 components for empty segmentation"
        )

    @pytest.mark.parametrize("preprocessor_class", ALL_PREPROCESSORS)
    def test_min_voxels_filtering(self, preprocessor_class, mock_two_lesions):
        """
        Edge Case: Components smaller than min_voxels should be ignored.
        """
        image, seg = mock_two_lesions

        # Create a tiny speck of noise (1 voxel) in the seg
        # Note: MetaTensor can be indexed like torch tensor
        seg[10, 10, 10] = 1

        preprocessor = preprocessor_class()

        # Set min_voxels higher than 1 so the noise speck is ignored
        # The fixture has actual lesions of size 1000 voxels (10x10x10)
        components_gen = preprocessor.stream_components(image, seg, min_voxels=10)
        components = list(components_gen)

        # We assume the implementation correctly calculates connected components.
        # We verify that we didn't get a 3rd component corresponding to the single voxel.
        # (We expect exactly 2 from the main fixture setup)
        assert len(components) == 2, (
            f"{preprocessor_class.__name__} failed: Noise filtering did not work as expected."
        )

    def test_config_generation(self):
        """Ensure get_config returns a dictionary and doesn't crash."""
        preprocessor = CTPreprocessor()
        config = preprocessor.get_config()
        assert isinstance(config, dict)
