"""
Preprocessing logic using MONAI for CT images.
"""

import copy
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch
from monai.data import MetaTensor
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImage,
    MapLabelValued,
    NormalizeIntensityd,
    Orientationd,
    RandAdjustContrastd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandShiftIntensityd,
    Resized,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
)
from scipy import ndimage

from renal_vision.features.transforms import ConditionalAddChanneld, MinimumCropForegroundd

ImageLike = Union[str, Path, MetaTensor]


class BasePreprocessor(ABC):
    """Abstract base class for all preprocessors."""

    def _prepare_data_point(self, image: ImageLike) -> MetaTensor:
        if isinstance(image, MetaTensor):
            return image
        else:
            return LoadImage()(image)

    def _prepare_data(self, image: ImageLike, seg: ImageLike) -> Dict[str, MetaTensor]:
        """Helper method to normalize inputs into a MONAI-compatible dictionary."""
        return {
            "image": self._prepare_data_point(image),
            "seg": self._prepare_data_point(seg),
        }

    def _finalize_numpy(self, data: dict[str, MetaTensor]) -> Tuple[np.ndarray, np.ndarray]:
        """Helper to convert dictionary to numpy return format."""
        image_np = data["image"].detach().cpu().numpy().squeeze()
        seg_np = data["seg"].detach().cpu().numpy().squeeze().astype(np.int16)
        return image_np, seg_np

    def __init__(
        self,
        base_transforms: List[Any] = [],
        aug_transforms: List[Any] = [],
        intensity_transforms: List[Any] = [],
        crop_transforms: List[Any] = [],
    ):
        """Initializes the preprocessor with given transform lists."""
        self.base_transforms = base_transforms
        self.aug_transforms = aug_transforms
        self.intensity_transforms = intensity_transforms
        self.crop_transforms = crop_transforms

    def __call__(
        self,
        image: ImageLike,
        seg: ImageLike,
        augment: bool = False,
    ) -> Tuple[MetaTensor, MetaTensor]:
        """
        Applies the preprocessing pipeline to the image and segmentation.
        Load -> [optional Augment] -> Intensity Normalize -> Return

        Returns: image, seg
        """

        # Load Inputs
        data = self._prepare_data(image, seg)

        # 2. Build Pipeline (cast strictly for MyPy)
        transforms = list(self.base_transforms)
        if augment:
            transforms.extend(cast(List[Any], self.aug_transforms))
        transforms.extend(cast(List[Any], self.intensity_transforms))
        pipeline = Compose(transforms)

        # 3. Apply Transforms
        data = pipeline(data)

        return data["image"], data["seg"]

    def stream_augmented(
        self, image: ImageLike, seg: ImageLike, n_augmentations: int
    ) -> Generator[Tuple[MetaTensor, MetaTensor, bool], None, None]:
        """
        Generator that loads once, then yields:
        1. The original (non-augmented) image/seg
        2. n_augmentations versions of augmented image/seg

        Returns: image, seg, is_augmented
        """

        if n_augmentations < 0:
            raise ValueError(f"Number of augmentations cannot be negative. Is: {n_augmentations}")

        # 1. HEAVY LIFTING (Done Once)
        base_data = self._prepare_data(image, seg)
        base_data = Compose(self.base_transforms)(base_data)

        # 2. Pipeline for intensity (always applied) and augmentation
        # We need separate pipelines because we apply them at different stages
        aug_pipeline = Compose(cast(List[Any], self.aug_transforms))
        intensity_pipeline = Compose(cast(List[Any], self.intensity_transforms))

        # 3. Yield Original
        data_orig = copy.deepcopy(base_data)
        data_orig = intensity_pipeline(data_orig)
        yield data_orig["image"], data_orig["seg"], False

        # 4. Yield Augmentations
        for _ in range(n_augmentations):
            data_aug = copy.deepcopy(base_data)
            data_aug = aug_pipeline(data_aug)
            data_aug = intensity_pipeline(data_aug)

            yield data_aug["image"], data_aug["seg"], True
            del data_aug

    def stream_components(
        self,
        image: MetaTensor,
        seg: MetaTensor,
        min_volume: int = 100,
    ) -> Generator[Tuple[np.ndarray, np.ndarray, dict[str, int]], None, None]:
        """
        Returns generator over individual connected components in the segmentation.
        Crops image and seg around each component, if specified.

        Returns: component_image, component_mask, metadata(class_id, class_intern_id, volume)
        """

        pipeline = Compose(cast(List[Any], self.crop_transforms))

        # Get all unique classes (excluding background 0)
        classes = np.unique(seg)
        classes = classes[classes > 0]

        # calculate voxel_volume
        spacing = np.linalg.norm(image.meta["affine"], axis=0)
        voxel_volume = np.prod(spacing)

        for c_id in classes:
            # Create binary mask for this class
            class_mask = seg == c_id

            # Find connected components
            labeled_mask, num_comp = ndimage.label(class_mask)
            labeled_mask = torch.tensor(labeled_mask)
            for comp_id in range(1, num_comp + 1):
                component_mask = seg * 0  # new empty meta tensor
                component_mask[labeled_mask == comp_id] = 1

                volume = int(np.sum(component_mask) * voxel_volume)
                if volume < min_volume:
                    continue  # skip small components

                # Apply cropping transforms
                data = {"image": image, "seg": component_mask}
                data = pipeline(data)
                cropped_image, cropped_mask = self._finalize_numpy(data)
                metadata = {
                    "class_id": int(c_id),
                    "volume": volume,
                }

                yield cropped_image, cropped_mask, metadata
                del cropped_image, cropped_mask, component_mask

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        pass


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
        normalize: bool = True,
        orientation: str = "RAS",
        heavy_augmentations: bool = False,
        rough_crop_margin: int | Sequence[int] = 20,
    ):
        self.target_spacing = target_spacing
        self.window_center = window_center
        self.window_width = window_width
        self.label_map = label_map or {0: 0}
        self.normalize = normalize
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
                keys=["image", "seg"], source_key="seg", margin=rough_crop_margin, lazy=True
            ),
        ]

        # 2. Augmentation
        aug_transforms = [
            # Spatial Transforms (Anatomy & Positioning)
            RandFlipd(keys=["image", "seg"], prob=0.3, spatial_axis=0, lazy=True),
            RandFlipd(keys=["image", "seg"], prob=0.3, spatial_axis=1, lazy=True),
            RandFlipd(keys=["image", "seg"], prob=0.3, spatial_axis=2, lazy=True),
            RandRotate90d(keys=["image", "seg"], prob=0.3, max_k=3, spatial_axes=(0, 1), lazy=True),
            RandRotate90d(keys=["image", "seg"], prob=0.3, max_k=3, spatial_axes=(0, 2), lazy=True),
            RandRotate90d(keys=["image", "seg"], prob=0.3, max_k=3, spatial_axes=(1, 2), lazy=True),
            # Intensity Variation
            RandShiftIntensityd(keys=["image"], offsets=10, prob=0.3),
            # Simulate slightly incorrect segmentations
            RandAffined(keys=["seg"], prob=0.5, translate_range=2, mode="nearest", lazy=True),
        ]

        if heavy_augmentations:
            aug_transforms += [
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
                # 3. Intensity Transforms (Scanner Variation)
                RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.1),
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

        # 3. Intensity Transforms (should be applied after augmentation)
        # Calculate window boundaries and normalize to [0, 1] if specified
        win_min = self.window_center - self.window_width / 2
        win_max = self.window_center + self.window_width / 2
        b_min = 0.0 if self.normalize else win_min
        b_max = 1.0 if self.normalize else win_max

        intensity_transforms = [
            ScaleIntensityRanged(
                keys=["image"],
                a_min=win_min,
                a_max=win_max,
                b_min=b_min,
                b_max=b_max,
                clip=True,
            ),
        ]

        # 4. Cropping
        crop_transforms = [
            ConditionalAddChanneld(keys=["image", "seg"]),
            CropForegroundd(keys=["image", "seg"], source_key="seg", margin=10),
        ]

        # 5. Initialize Base Class
        super().__init__(
            base_transforms=base_transforms,
            aug_transforms=aug_transforms,
            intensity_transforms=intensity_transforms,
            crop_transforms=crop_transforms,
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": "CTPreprocessor",
            "target_spacing": self.target_spacing,
            "window_center": self.window_center,
            "window_width": self.window_width,
            "normalize": self.normalize,
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

        self.intensity_transforms = (
            [NormalizeIntensityd(keys=["image"], subtrahend=-1024, divisor=3072)]  # type: ignore [arg-type]
            if self.normalize
            else []
        )

        self.crop_transforms = [
            ConditionalAddChanneld(keys=["image", "seg"]),
            MinimumCropForegroundd(
                keys=["image", "seg"],
                source_key="seg",
                margin=10,
                min_shape=self.target_shape if self.target_shape is not None else (50, 50, 50),
                allow_smaller=True,
            ),
        ]
        if target_shape is not None:
            self.crop_transforms.append(
                Resized(keys=["image", "seg"], mode=["area", "nearest"], spatial_size=target_shape)
            )


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

    def __init__(self, target_spacing: Tuple[float, float, float] = (1.0, 1.0, 3.0), **kwargs):
        super().__init__(target_spacing=target_spacing, heavy_augmentations=True, **kwargs)

        # 4. Cropping Transforms
        self.crop_transforms = [
            ConditionalAddChanneld(keys=["image", "seg"]),
            MinimumCropForegroundd(
                keys=["image", "seg"],
                source_key="seg",
                min_shape=(64, 64, 3),
                margin=10,
            ),
            # If the patient's body doesn't fill the 64,64 frame (e.g. at the edge), pad with 0.
            SpatialPadd(keys=["image", "seg"], spatial_size=(64, 64, -1)),
            Resized(keys=["image", "seg"], mode=["area", "nearest"], spatial_size=(64, 64, -1)),
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

    def __init__(self, target_spacing: Tuple[float, float, float] = (3.0, 1.0, 1.0), **kwargs):
        super().__init__(
            orientation="SPL",
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
                min_shape=(24, 128, 128),
                margin=10,
            ),
            SpatialPadd(keys=["image", "seg"], spatial_size=(24, 128, 128)),
        ]
