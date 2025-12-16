# conftest.py
from typing import Tuple

import pytest
import torch
from monai.data import MetaTensor


@pytest.fixture
def mock_no_lesion() -> Tuple[MetaTensor, MetaTensor]:
    """Generates data with an empty segmentation mask (no lesions)."""
    spatial_shape = (60, 60, 60)
    img_array = torch.rand(*spatial_shape)
    seg_array = torch.zeros(*spatial_shape)  # All zeros

    return MetaTensor(img_array), MetaTensor(seg_array)


@pytest.fixture
def mock_single_lesion() -> Tuple[MetaTensor, MetaTensor]:
    """Generates synthetic 3D CT image and segmentation."""
    spatial_shape = (60, 60, 60)  # Smaller for speed
    img_array = torch.rand(*spatial_shape)
    seg_array = torch.zeros(*spatial_shape)
    seg_array[10:20, 10:20, 10:20] = 1  # Lesion 1

    return MetaTensor(img_array), MetaTensor(seg_array)


@pytest.fixture
def mock_two_lesions() -> Tuple[MetaTensor, MetaTensor]:
    """
    Generates a synthetic 3D CT image and a corresponding segmentation mask.

    Returns:
        image (MetaTensor): Shape (1, 100, 100, 100) - Random noise simulating CT.
        seg (MetaTensor): Shape (1, 100, 100, 100) - Binary mask with 2 distinct cubes.
    """
    spatial_shape = (100, 100, 100)
    img_array = torch.rand(*spatial_shape)
    image = MetaTensor(img_array, affine=torch.eye(3))
    seg_array = torch.zeros(*spatial_shape)

    # Lesion 1: 10x10x10 cube at [20:30, 20:30, 20:30]
    seg_array[20:30, 20:30, 20:30] = 1
    # Lesion 2: 10x10x10 cube at [60:70, 60:70, 60:70]
    seg_array[60:70, 60:70, 60:70] = 1

    seg = MetaTensor(seg_array, affine=torch.eye(3))
    return image, seg
