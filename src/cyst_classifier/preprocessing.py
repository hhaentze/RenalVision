"""Preprocessing functions for CT images and segmentations."""

import warnings

import numpy as np
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Spacingd,
)
from scipy import ndimage


def validate_hu_range(image):
    """
    Validate that image values are in valid Hounsfield Unit range.

    Args:
        image: numpy array with intensity values

    Raises:
        ValueError: If values are outside valid HU range [-2048, 3071]
    """
    min_val, max_val = image.min(), image.max()
    if min_val < -2048 or max_val > 3071:
        raise ValueError(
            f"Image values [{min_val:.1f}, {max_val:.1f}] outside valid HU range "
            f"[-2048, 3071]. Ensure input is in Hounsfield Units."
        )


def load_and_preprocess(
    image_path,
    seg_path,
    target_spacing=(2.0, 2.0, 2.0),
    window_center=40,
    window_width=400,
    map_labels=True,
):
    """
    Load and preprocess CT image and segmentation.

    Preprocessing steps:
    1. Load image and segmentation
    2. Validate HU range
    3. Resample to target spacing (2mm isotropic)
    4. Window CT to [center-width/2, center+width/2] = [-160, 240] HU (clipped)
    5. Map segmentation labels: (1,2,3) -> (0,1,1) if map_labels=True
    6. Set kidney voxels (label=1 in original) to 0 in segmentation

    Args:
        image_path: Path to CT image (.nii.gz)
        seg_path: Path to segmentation (.nii.gz)
        target_spacing: Target voxel spacing in mm (default: 2mm isotropic)
        window_center: CT window center in HU (default: 40)
        window_width: CT window width in HU (default: 400)
        map_labels: If True, map (1,2,3) to (0,1,1) (default: True)

    Returns:
        image: Preprocessed CT image (numpy array, in HU within window)
        seg: Preprocessed segmentation (numpy array)
        affine: Affine transformation matrix

    Raises:
        ValueError: If image is not in HU range or no lesions found
    """
    # Load images using MONAI
    transforms = Compose(
        [
            LoadImaged(keys=["image", "seg"], image_only=False),
            EnsureChannelFirstd(keys=["image", "seg"]),
            Spacingd(keys=["image", "seg"], pixdim=target_spacing, mode=("bilinear", "nearest")),
            EnsureTyped(keys=["image", "seg"]),
        ]
    )

    data = transforms({"image": image_path, "seg": seg_path})

    # Extract numpy arrays
    image = data["image"].squeeze().cpu().numpy()
    seg = data["seg"].squeeze().cpu().numpy().astype(np.int32)
    affine = data["image"].meta["affine"]

    # Validate HU range
    validate_hu_range(image)

    # Apply CT windowing: clip to [center - width/2, center + width/2]
    window_min = window_center - window_width / 2  # -160 HU
    window_max = window_center + window_width / 2  # 240 HU
    image = np.clip(image, window_min, window_max)

    # Map labels if requested
    if map_labels:
        # Map: 1 (kidney) -> 0, 2 (tumor) -> 1, 3 (cyst) -> 1
        seg_mapped = np.zeros_like(seg)
        seg_mapped[seg == 2] = 1  # tumor
        seg_mapped[seg == 3] = 1  # cyst
        seg = seg_mapped
    else:
        # Still remove kidney label
        seg[seg == 1] = 0

    return image, seg, affine


def extract_lesions(image, seg, min_voxels=10, exclude_border=True):
    """
    Extract individual lesions from segmentation as separate samples.

    Uses connected component analysis to identify individual lesions.
    Each lesion is returned as a separate masked region.

    Args:
        image: CT image (numpy array)
        seg: Segmentation mask (numpy array, non-zero = lesion)
        min_voxels: Minimum lesion size in voxels (default: 10)
        exclude_border: If True, exclude lesions touching image boundaries (default: True)

    Returns:
        lesions: List of tuples (lesion_image, lesion_mask, label)
                 - lesion_image: Full image (for context)
                 - lesion_mask: Binary mask for this specific lesion
                 - label: Original label value (2=tumor, 3=cyst) or 1 if mapped

    Raises:
        ValueError: If no lesions found
    """
    # Find connected components
    labeled_seg, num_lesions = ndimage.label(seg > 0)

    if num_lesions == 0:
        raise ValueError("No lesions found in segmentation")

    lesions = []

    for lesion_id in range(1, num_lesions + 1):
        lesion_mask = labeled_seg == lesion_id

        # Check size
        lesion_size = np.sum(lesion_mask)
        if lesion_size < min_voxels:
            warnings.warn(
                f"Lesion {lesion_id} has only {lesion_size} voxels (< {min_voxels}). Skipping."
            )
            continue

        # Check if touching border
        if exclude_border:
            if touches_border(lesion_mask):
                continue

        # Get original label (for ground truth)
        # Take the most common non-zero label within this lesion
        original_labels = seg[lesion_mask]
        label = np.bincount(original_labels[original_labels > 0]).argmax()

        lesions.append((image, lesion_mask.astype(np.uint8), int(label)))

    if len(lesions) == 0:
        raise ValueError("No valid lesions found after filtering")

    return lesions


def touches_border(mask):
    """
    Check if a binary mask touches the image boundary.

    Args:
        mask: Binary mask (numpy array)

    Returns:
        bool: True if mask touches any face of the 3D volume
    """
    return (
        np.any(mask[0, :, :])
        or np.any(mask[-1, :, :])
        or np.any(mask[:, 0, :])
        or np.any(mask[:, -1, :])
        or np.any(mask[:, :, 0])
        or np.any(mask[:, :, -1])
    )
