"""Preprocessing functions for CT images and segmentations."""

import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from monai.data import MetaTensor
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapLabelValued,
    Spacingd,
)
from scipy import ndimage


class CTPreprocessor:
    """Handle loading and preprocessing of CT images and segmentations for both file paths and numpy arrays."""

    def __init__(
        self,
        target_spacing: Tuple[float, float, float] = (2.0, 2.0, 2.0),
        window_center: float = 40,
        window_width: float = 400,
        label_map: Dict[int, int] = {0: 0},
    ):
        self.target_spacing = target_spacing
        self.window_center = window_center
        self.window_width = window_width
        self.label_map = label_map

        # Define the shared pipeline (Transforms that apply to BOTH files and arrays)
        self.transforms = Compose(
            [
                EnsureChannelFirstd(keys=["image", "seg"], channel_dim="no_channel"),
                Spacingd(
                    keys=["image", "seg"], pixdim=self.target_spacing, mode=("bilinear", "nearest")
                ),
                EnsureTyped(keys=["image", "seg"]),
                MapLabelValued(
                    keys=["seg"],
                    orig_labels=list(label_map.keys()),
                    target_labels=list(label_map.values()),
                    dtype=np.int16,
                ),
            ]
        )

    def _finalize_result(self, data: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Internal shared method to extract arrays, window, and return.
        """
        # 1. Extract
        # .array converts MetaTensor back to pure numpy
        image = data["image"].array.squeeze()
        seg = data["seg"].array.squeeze().astype(np.int32)

        # Get the new affine from the MetaTensor (it was updated by Spacingd)
        new_affine = data["image"].affine.numpy()

        # 2. Validate
        validate_hu_range(image)

        # 3. Windowing (Shared logic)
        win_min = self.window_center - self.window_width / 2
        win_max = self.window_center + self.window_width / 2
        image = np.clip(image, win_min, win_max)

        return image, seg, new_affine

    def process_files(
        self, image_path: str | Path, seg_path: str | Path
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Entry point for File Paths.
        """
        # Load files specifically
        loader = LoadImaged(keys=["image", "seg"], image_only=False)
        data = loader({"image": image_path, "seg": seg_path})

        # Pass to shared transforms
        data = self.transforms(data)

        return self._finalize_result(data)

    def process_arrays(
        self, image_arr: np.ndarray, seg_arr: np.ndarray, original_affine: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Entry point for Numpy Arrays.
        """
        # Wrap numpy in MetaTensor to "fake" a loaded file
        # This injects the metadata required for Spacingd to work
        data = {
            "image": MetaTensor(torch.tensor(image_arr), affine=torch.tensor(original_affine)),
            "seg": MetaTensor(torch.tensor(seg_arr), affine=torch.tensor(original_affine)),
        }

        # Pass to shared transforms
        data = self.transforms(data)

        return self._finalize_result(data)


def validate_hu_range(image):
    """
    Validate that image values are in valid Hounsfield Unit range.
    """
    min_val, max_val = float(image.min()), float(image.max())
    if min_val < -2048 or max_val > 3071:
        raise ValueError(
            f"Image values [{min_val:.1f}, {max_val:.1f}] outside valid HU range "
            f"[-2048, 3071]. Ensure input is in Hounsfield Units."
        )


def create_affine_from_spacing(spacing: Tuple[float, float, float]) -> np.ndarray:
    """Creates a diagonal 4x4 affine matrix from spacing."""
    affine = np.eye(4)
    affine[0, 0] = spacing[0]
    affine[1, 1] = spacing[1]
    affine[2, 2] = spacing[2]
    return affine


def extract_lesions(
    image: np.ndarray, seg: np.ndarray, min_voxels: int = 10, exclude_border: bool = True
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """
    Extract individual lesions from segmentation as separate samples.
    """
    # Find connected components
    labeled_seg, num_lesions = ndimage.label(seg > 0)

    if num_lesions == 0:
        raise ValueError("No lesions found in segmentation")

    lesions: List[Tuple[np.ndarray, np.ndarray, int]] = []

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
        original_labels = seg[lesion_mask]
        label = np.bincount(original_labels[original_labels > 0]).argmax()

        lesions.append((image, lesion_mask.astype(np.uint8), int(label)))

    if len(lesions) == 0:
        raise ValueError("No valid lesions found after filtering")

    return lesions


def touches_border(mask: np.ndarray) -> bool:
    """
    Check if a binary mask touches the image boundary.
    """
    return bool(
        np.any(mask[0, :, :])
        or np.any(mask[-1, :, :])
        or np.any(mask[:, 0, :])
        or np.any(mask[:, -1, :])
        or np.any(mask[:, :, 0])
        or np.any(mask[:, :, -1])
    )
