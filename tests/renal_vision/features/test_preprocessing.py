import numpy as np
import pytest
from monai.data import MetaTensor

from renal_vision.features.preprocessing import (
    CTFMPreprocessor,
    CTPreprocessor,
    FMCIBPreprocessor,
    MevisPreprocessor,
)

# ==========================================
# 1. Happy Path Tests
# ==========================================


class TestCTPreprocessor:
    """Tests for the standard CT Preprocessor."""

    def test_call_structure_and_types(self, mock_single_lesion):
        """Verify __call__ returns MetaTensors of correct dimensionality."""
        image, seg = mock_single_lesion
        preprocessor = CTPreprocessor()

        # Act
        out_img, out_seg = preprocessor(image, seg)

        # Assert Types
        assert isinstance(out_img, MetaTensor)
        assert isinstance(out_seg, MetaTensor)

        # Assert Shapes (Should preserve channel dim)
        # Note: Exact spatial shape might change due to spacing, but dims should match
        assert out_img.ndim == 4  # (D, H, W)
        assert out_seg.ndim == 4

    def test_volume_filtering1(self, mock_two_lesions):
        """Verify that all lesions are found"""

        image, seg = mock_two_lesions
        preprocessor = CTPreprocessor()

        # Act
        _, out_seg = preprocessor(image, seg)
        valid_components, metadata_list = preprocessor.filter_components(out_seg, min_volume=1)

        # Assert
        assert len(metadata_list) == 2, "Should detect the 2 synthetic lesions created in fixture"
        assert len(np.unique(valid_components)) == 3, "Should include the classes [0,1,2]"
        assert metadata_list[0]["volume"] == 4000
        assert metadata_list[1]["volume"] == 1000
        assert metadata_list[0]["class_id"] == 2
        assert metadata_list[1]["class_id"] == 1

    def test_volume_filtering2(self, mock_two_lesions):
        """Verify that small lesions are excluded"""

        image, seg = mock_two_lesions
        preprocessor = CTPreprocessor()

        # Act
        _, out_seg = preprocessor(image, seg)
        valid_components, metadata_list = preprocessor.filter_components(out_seg, min_volume=1001)

        # Assert
        assert len(metadata_list) == 1, (
            "Should detect one of the two synthetic lesions created in fixture"
        )
        assert len(np.unique(valid_components)) == 2, "Should include the classes [0,2]"
        assert metadata_list[0]["volume"] == 4000
        assert metadata_list[0]["class_id"] == 2


class TestCropPreprocessors:
    """Tests for Crop and StaticCrop Preprocessors."""

    @pytest.mark.parametrize("processor_class", [FMCIBPreprocessor])
    def test_stream_components_shape_contract_fmcib(self, processor_class, mock_two_lesions):
        """
        CRITICAL: Ensure stream_components yields numpy arrays of shape 50x50x50.
        This validates the specific requirement for feature extraction input.
        """
        image, seg = mock_two_lesions
        preprocessor = processor_class()

        # Act
        # We expect 2 components based on the fixture data
        components = list(preprocessor.stream_components(image, seg, min_volume=1))

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

    @pytest.mark.parametrize("processor_class", [MevisPreprocessor])
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

            # 2. Check Strict Shape Requirement (64,64,z)
            expected_shape = 64
            assert (comp_img.shape[0] == expected_shape) and (
                comp_img.shape[1] == expected_shape
            ), f"Expected image x and y of size {expected_shape}, but got shape {comp_img.shape}"
            assert (comp_mask.shape[0] == expected_shape) and (
                comp_mask.shape[1] == expected_shape
            ), f"Expected mask x and y of size {expected_shape}, but got {comp_mask.shape}"

            # 3. Check Metadata keys
            assert "class_id" in metadata
            assert "volume" in metadata


class TestAugmentations:
    ALL_PREPROCESSORS = [CTPreprocessor, FMCIBPreprocessor, MevisPreprocessor, CTFMPreprocessor]
    HEAVY_AUG_PREPROCESSORS = [FMCIBPreprocessor, MevisPreprocessor, CTFMPreprocessor]

    @pytest.mark.parametrize("preprocessor_class", ALL_PREPROCESSORS)
    def test_no_augmentation1(self, preprocessor_class, mock_single_lesion):
        image, seg = mock_single_lesion
        preprocessor = preprocessor_class()

        component1 = next(preprocessor.stream_components(image, seg, min_volume=1))
        component2 = next(preprocessor.stream_components(image, seg, min_volume=1))

        assert np.array_equal(component1[0], component2[0]), "images should be equal"
        assert np.array_equal(component1[1], component2[1]), "segmentations should be equal"
        assert component1[2] == component2[2], "metadata should be equal"

    @pytest.mark.parametrize("preprocessor_class", ALL_PREPROCESSORS)
    def test_no_augmentation2(self, preprocessor_class, mock_single_lesion):
        image, seg = mock_single_lesion
        preprocessor = preprocessor_class()

        _img1, _seg1 = preprocessor(image, seg)
        _img2, _seg2 = preprocessor(image, seg)
        component1 = next(preprocessor.stream_components(_img1, _seg1, min_volume=1))
        component2 = next(preprocessor.stream_components(_img2, _seg2, min_volume=1))

        assert np.array_equal(component1[0], component2[0]), "images should be equal"
        assert np.array_equal(component1[1], component2[1]), "segmentations should be equal"
        assert component1[2] == component2[2], "metadata should be equal"

    @pytest.mark.parametrize("preprocessor_class", HEAVY_AUG_PREPROCESSORS)
    def test_augmentation(self, preprocessor_class, mock_single_lesion):
        image, seg = mock_single_lesion
        preprocessor = preprocessor_class()

        component1 = next(preprocessor.stream_components(image, seg, min_volume=1, augment=True))

        # first chance
        component2 = next(preprocessor.stream_components(image, seg, min_volume=1, augment=True))
        # second chance
        if np.array_equal(component1[0], component2[0]):
            component2 = next(
                preprocessor.stream_components(image, seg, min_volume=1, augment=True)
            )

        assert not np.array_equal(component1[0], component2[0]), "images should not be equal"
        assert component1[2] == component2[2], "metadata should be equal"


# ==========================================
# 2. Edge Case Tests
# ==========================================
class TestEdgeCases:
    ALL_PREPROCESSORS = [CTPreprocessor, FMCIBPreprocessor, MevisPreprocessor, CTFMPreprocessor]

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
    def test_min_volume_filtering(self, preprocessor_class, mock_two_lesions):
        """
        Edge Case: Components smaller than min_volume should be ignored.
        """
        image, seg = mock_two_lesions

        # Create a tiny speck of noise (1 voxel) in the seg
        # Note: MetaTensor can be indexed like torch tensor
        seg[10, 10, 10] = 1

        preprocessor = preprocessor_class()

        # Set min_volume higher than 1 so the noise speck is ignored
        # The fixture has actual lesions of size 1000 voxels (10x10x10)
        components_gen = preprocessor.stream_components(image, seg, min_volume=10)
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
