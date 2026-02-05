"""
Preprocessing logic using MONAI for CT images.
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from monai.transforms import (
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    MapLabelValued,
    NormalizeIntensityd,
    Orientationd,
    RandAdjustContrastd,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandShiftIntensityd,
    Resized,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
)

from .base_preprocessor import BasePreprocessor
from .transforms import ConditionalAddChanneld, MinimumCropForegroundd


class CTPreprocessor(BasePreprocessor):
    """
    Standard CT Preprocessor using MONAI.
    """

    def __init__(
        self,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        window_center: float = 40,
        window_width: float = 400,
        label_map: Optional[Dict[int, int]] = None,
        orientation: str = "RAS",
        heavy_augmentations: bool = False,
        rough_crop_margin: int | Sequence[int] = 20,
    ):
        self.target_spacing = target_spacing
        self.window_center = window_center
        self.window_width = window_width
        self.label_map = label_map or {0: 0}
        self.orientation = orientation

        if isinstance(rough_crop_margin, int):
            rough_crop_margin = [rough_crop_margin] * 3
        rough_crop_margin = [
            int(rcm // target_spacing[i]) for i, rcm in enumerate(rough_crop_margin)
        ]

        # 1. Base Transforms
        base_transforms = [
            EnsureChannelFirstd(keys=["image", "seg"], channel_dim="no_channel"),
            ConditionalAddChanneld(keys=["image", "seg"]),
            EnsureTyped(keys=["image", "seg"]),
            MapLabelValued(
                keys=["seg"],
                orig_labels=list(self.label_map.keys()),
                target_labels=list(self.label_map.values()),
                dtype=np.int16,
            ),
            Orientationd(keys=["image", "seg"], axcodes=self.orientation, lazy=True),
            Spacingd(
                keys=["image", "seg"],
                pixdim=self.target_spacing,
                mode=("bilinear", "nearest"),
                lazy=True,
            ),
            # rough crop to non-zero seg region to speed up subsequent processing
            CropForegroundd(
                keys=["image", "seg"],
                source_key="seg",
                margin=rough_crop_margin,
                lazy=True,
                allow_smaller=True,
            ),
        ]

        # 2. Cropping
        # (lazy transforms at the end)
        crop_transforms = [
            ConditionalAddChanneld(keys=["image", "seg"]),
            CropForegroundd(
                keys=["image", "seg"], source_key="seg", margin=10, allow_smaller=True, lazy=True
            ),
        ]

        # 3. Augmentation
        # (lazy transforms at the beginning)
        if heavy_augmentations:
            aug_transforms = [
                RandAffined(
                    keys=["image", "seg"],
                    prob=0.5,
                    rotate_range=(np.pi / 12, np.pi / 12, np.pi / 12),  # +/- 15 degrees
                    scale_range=(0.1, 0.1, 0.1),  # +/- 10% zoom
                    translate_range=(3, 3, 3),  # Shift center by +/- 3 mm
                    mode=("bilinear", "nearest"),  # Bilinear for image, Nearest for mask
                    padding_mode="border",
                    lazy=True,
                ),
                # Simulate slightly incorrect segmentations
                RandAffined(keys=["seg"], prob=0.5, translate_range=2, mode="nearest", lazy=True),
                # Intensity Transforms (Scanner Variation)
                RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
                # Simulate different reconstruction kernels (Blur)
                RandGaussianSmoothd(
                    keys=["image"],
                    prob=0.2,
                    sigma_x=(0.5, 1.0),
                    sigma_y=(0.5, 1.0),
                    sigma_z=(0.5, 1.0),
                ),
                # Simulate subtle density variations (Gamma); retains the general HU relationships but stretches contrast
                RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
            ]
        else:
            aug_transforms = [
                # Simulate slightly incorrect segmentations
                RandAffined(keys=["seg"], prob=0.5, translate_range=2, mode="nearest", lazy=True),
                # Intensity Variation
                RandShiftIntensityd(keys=["image"], offsets=10, prob=0.3),
            ]

        # 4. Intensity Transforms (should be applied after augmentation)
        win_min = self.window_center - self.window_width / 2
        win_max = self.window_center + self.window_width / 2

        intensity_transforms = [
            ScaleIntensityRanged(
                keys=["image"],
                a_min=win_min,
                a_max=win_max,
                b_min=win_min,
                b_max=win_max,
                clip=True,
            ),
        ]

        # 5. Initialize Base Class
        super().__init__(
            base_transforms=base_transforms,
            crop_transforms=crop_transforms,
            aug_transforms=aug_transforms,
            intensity_transforms=intensity_transforms,
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": "CTPreprocessor",
            "target_spacing": self.target_spacing,
            "window_center": self.window_center,
            "window_width": self.window_width,
            "orientation": self.orientation,
        }


class FMCIBPreprocessor(CTPreprocessor):
    """
    Preprocessor using FMCIB preprocessing settings:
    - Crops around lesions to a target size of [50,50,50]
    - Normalizes intensity by subtracting -1024 and dividing by 3072
    """

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["name"] = "FMCIBPreprocessor"
        config["target_shape"] = self.target_shape
        return config

    def __init__(self, target_shape: Optional[Tuple[int, int, int]] = (50, 50, 50), **kwargs):
        super().__init__(heavy_augmentations=True, **kwargs)
        self.target_shape = target_shape

        # Cropping
        self.crop_transforms = [
            ConditionalAddChanneld(keys=["image", "seg"]),
            MinimumCropForegroundd(
                keys=["image", "seg"],
                source_key="seg",
                margin=10,
                min_shape=self.target_shape if self.target_shape is not None else (50, 50, 50),
                allow_smaller=True,
                lazy=True,
            ),
        ]
        if target_shape is not None:
            self.crop_transforms.append(
                Resized(
                    keys=["image", "seg"],
                    mode=["area", "nearest"],
                    spatial_size=target_shape,
                    lazy=True,
                )
            )

        # Intensity Normalization
        self.intensity_transforms = [
            NormalizeIntensityd(keys=["image"], subtrahend=-1024, divisor=3072)  # type: ignore [arg-type]
        ]


class MevisPreprocessor(CTPreprocessor):
    """
    Preprocessor using MevisLab-inspired cropping and normalization:
    - Crops around lesions to a target size of [64,64,variable]
    - Normalizes intensity to a specified window and scales to [0,1]
    """

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["name"] = "MevisPreprocessor"
        return config

    def __init__(self, target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0), **kwargs):
        super().__init__(target_spacing=target_spacing, heavy_augmentations=True, **kwargs)

        # Cropping Transforms
        self.crop_transforms = [
            ConditionalAddChanneld(keys=["image", "seg"]),
            MinimumCropForegroundd(
                keys=["image", "seg"],
                source_key="seg",
                min_shape=(64, 64, 3),
                margin=10,
                allow_smaller=True,
                lazy=True,
            ),
            # If the patient's body doesn't fill the 64,64 frame (e.g. at the edge), pad with 0.
            SpatialPadd(keys=["image", "seg"], spatial_size=(64, 64, -1), lazy=True),
            Resized(
                keys=["image", "seg"],
                mode=["area", "nearest"],
                spatial_size=(64, 64, -1),
                lazy=True,
            ),
        ]

        # intensity transforms
        win_min = self.window_center - self.window_width / 2
        win_max = self.window_center + self.window_width / 2
        self.intensity_transforms = [
            ScaleIntensityRanged(
                keys=["image"],
                a_min=win_min,
                a_max=win_max,
                b_min=0,
                b_max=1,
                clip=True,
            ),
        ]


class CTFMPreprocessor(CTPreprocessor):
    """
    Preprocessor using CTFM inspired cropping and normalization:
    - load images in SPL orientation
    - crop without resizing
    """

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["name"] = "CTFMPreprocessor"
        return config

    def __init__(
        self,
        target_spacing: Tuple[float, float, float] = (3.0, 1.0, 1.0),
        orientation="SPL",
        **kwargs,
    ):
        super().__init__(
            orientation=orientation,
            target_spacing=target_spacing,
            heavy_augmentations=True,
            rough_crop_margin=60,
            **kwargs,
        )

        self.crop_transforms = [
            ConditionalAddChanneld(keys=["image", "seg"]),
            MinimumCropForegroundd(
                keys=["image", "seg"],
                source_key="seg",
                min_shape=(24, 64, 64),
                margin=10,
                allow_smaller=True,
                lazy=True,
            ),
            SpatialPadd(keys=["image", "seg"], spatial_size=(24, 64, 64), lazy=True),
        ]

        # intensity transforms
        win_min = self.window_center - self.window_width / 2
        win_max = self.window_center + self.window_width / 2
        self.intensity_transforms = [
            ScaleIntensityRanged(
                keys=["image"],
                a_min=win_min,
                a_max=win_max,
                b_min=0,
                b_max=1,
                clip=True,
            ),
        ]
