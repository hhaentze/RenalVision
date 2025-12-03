"""
Preprocessing logic using MONAI for CT images.
"""

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch
from monai.data import MetaTensor
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapLabelValued,
    RandAffined,
    RandGaussianNoised,
    ScaleIntensityRange,
    Spacingd,
)


class BasePreprocessor(ABC):
    """Abstract base class for all preprocessors."""

    def _prepare_data(
        self,
        image: Union[str, Path, np.ndarray],
        seg: Union[str, Path, np.ndarray],
        affine: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Helper method to normalize inputs into a MONAI-compatible dictionary.
        """
        if isinstance(image, (str, Path)) and isinstance(seg, (str, Path)):
            loader = LoadImaged(keys=["image", "seg"], image_only=False)
            return loader({"image": image, "seg": seg})

        elif isinstance(image, np.ndarray) and isinstance(seg, np.ndarray):
            if affine is None:
                affine = np.eye(4)
                warnings.warn("No affine provided for numpy inputs; defaulting to identity matrix.")
            return {
                "image": MetaTensor(torch.tensor(image), affine=torch.tensor(affine)),
                "seg": MetaTensor(torch.tensor(seg), affine=torch.tensor(affine)),
            }
        else:
            raise ValueError(
                "Inputs for 'image' and 'seg' must be both file paths or both numpy arrays."
            )

    @abstractmethod
    def __call__(
        self,
        image: Union[str, Path, np.ndarray],
        seg: Union[str, Path, np.ndarray],
        augment: bool = False,
        affine: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass

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

        # 2. Windowing (Intensity Clipping & Optional Scaling)
        # Calculate window boundaries
        win_min = self.window_center - self.window_width / 2
        win_max = self.window_center + self.window_width / 2

        # Normalize to [0, 1] if specified
        b_min = 0.0 if self.normalize else win_min
        b_max = 1.0 if self.normalize else win_max

        self.intensity_transform = ScaleIntensityRange(
            a_min=win_min,
            a_max=win_max,
            b_min=b_min,
            b_max=b_max,
            clip=True,
        )

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
        image: Union[str, Path, np.ndarray],
        seg: Union[str, Path, np.ndarray],
        augment: bool = False,
        affine: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # 1. Normalize Inputs
        data = self._prepare_data(image, seg, affine)

        # 2. Build Pipeline
        transforms = list(self.base_transforms)
        if augment:
            # Cast strictly for MyPy
            transforms.extend(cast(List[Any], self.aug_transforms))

        pipeline = Compose(transforms)

        # 3. Apply Transforms
        data = pipeline(data)

        # 4. Apply Windowing (on image only)
        img_tensor = data["image"]
        img_tensor = self.intensity_transform(img_tensor)

        # 5. Extract & Return Numpy
        if isinstance(img_tensor, MetaTensor):
            image_np = img_tensor.array.squeeze()
        elif isinstance(img_tensor, torch.Tensor):
            image_np = img_tensor.detach().cpu().numpy().squeeze()
        else:
            image_np = np.asarray(img_tensor).squeeze()

        seg_np = data["seg"].array.squeeze().astype(np.int32)

        return image_np, seg_np
