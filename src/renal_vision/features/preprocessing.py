"""
Preprocessing logic using MONAI for CT images.
"""

import copy
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union, cast

import numpy as np
from monai.data import MetaTensor
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImage,
    MapLabelValued,
    RandAffined,
    RandGaussianNoised,
    ScaleIntensityRanged,
    Spacingd,
)

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

    @abstractmethod
    def __call__(
        self,
        image: ImageLike,
        seg: ImageLike,
        augment: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def stream_augmented(
        self, image: ImageLike, seg: ImageLike, n_augmentations: int
    ) -> Generator[Tuple[np.ndarray, np.ndarray, bool], None, None]:
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        pass


class CTPreprocessor(BasePreprocessor):
    """
    Standard CT Preprocessor using MONAI.
    """

    def _finalize_numpy(self, data: dict[str, MetaTensor]) -> Tuple[np.ndarray, np.ndarray]:
        """Helper to convert dictionary to numpy return format."""
        image_np = data["image"].detach().cpu().numpy().squeeze()
        seg_np = data["seg"].detach().cpu().numpy().squeeze().astype(np.int16)
        return image_np, seg_np

    def __init__(
        self,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        window_center: float = 40,
        window_width: float = 400,
        label_map: Optional[Dict[int, int]] = None,
        normalize: bool = True,
    ):
        self.target_spacing = target_spacing
        self.window_center = window_center
        self.window_width = window_width
        self.label_map = label_map or {0: 0}
        self.normalize = normalize

        # 1. Base Transforms
        self.base_transforms = [
            EnsureChannelFirstd(keys=["image", "seg"], channel_dim="no_channel"),
            Spacingd(
                keys=["image", "seg"],
                pixdim=self.target_spacing,
                mode=("bilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "seg"]),
            MapLabelValued(
                keys=["seg"],
                orig_labels=list(self.label_map.keys()),
                target_labels=list(self.label_map.values()),
                dtype=np.int16,
            ),
        ]

        # 2. Intensity Transforms (should be applied after augmentation)

        # Calculate window boundaries and normalize to [0, 1] if specified
        win_min = self.window_center - self.window_width / 2
        win_max = self.window_center + self.window_width / 2
        b_min = 0.0 if self.normalize else win_min
        b_max = 1.0 if self.normalize else win_max

        self.intensity_transforms = [
            EnsureChannelFirstd(keys=["image", "seg"], channel_dim="no_channel"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=win_min,
                a_max=win_max,
                b_min=b_min,
                b_max=b_max,
                clip=True,
            ),
        ]

        # 3. Augmentation
        self.aug_transforms = [
            RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.1),
            RandAffined(
                keys=["image", "seg"],
                prob=0.5,
                rotate_range=(np.pi / 12, np.pi / 12, np.pi / 12),
                scale_range=(0.1, 0.1, 0.1),
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            ),
        ]

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": "CTPreprocessor",
            "target_spacing": self.target_spacing,
            "window_center": self.window_center,
            "window_width": self.window_width,
            "label_map": self.label_map,
            "normalize": self.normalize,
        }

    def __call__(
        self,
        image: ImageLike,
        seg: ImageLike,
        augment: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # 1. Normalize Inputs
        data = self._prepare_data(image, seg)

        # 2. Build Pipeline (cast strictly for MyPy)
        transforms = list(self.base_transforms)
        if augment:
            transforms.extend(cast(List[Any], self.aug_transforms))
        transforms.extend(cast(List[Any], self.intensity_transforms))
        pipeline = Compose(transforms)

        # 3. Apply Transforms
        data = pipeline(data)

        # 5. Extract & Return Numpy
        return self._finalize_numpy(data)

    def stream_augmented(
        self, image: ImageLike, seg: ImageLike, n_augmentations: int
    ) -> Generator[Tuple[np.ndarray, np.ndarray, bool], None, None]:
        """
        Generator that loads once, then yields:
        1. The original (non-augmented) image/seg
        2. n_augmentations versions of augmented image/seg

        Returns generator object with:
         - transformed image: np.ndarray
         - transformed segmentation: np.ndarray
         - is_augmented: bool
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
        yield *self._finalize_numpy(data_orig), False  # (data, is_augmented)

        # 4. Yield Augmentations
        for _ in range(n_augmentations):
            data_aug = copy.deepcopy(base_data)
            data_aug = aug_pipeline(data_aug)
            data_aug = intensity_pipeline(data_aug)

            yield *self._finalize_numpy(data_aug), True
