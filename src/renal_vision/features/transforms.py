# Copyright 2025 Hartmut Häntze

from collections.abc import Callable, Sequence
from typing import Dict, Hashable, Mapping, Union

import numpy as np
import torch
from monai.config import IndexSelection, KeysCollection, SequenceStr
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms import CropForeground, CropForegroundd, MapTransform, Transform
from monai.transforms.utils import (
    compute_divisible_spatial_size,
    generate_spatial_bounding_box,
    is_positive,
)
from monai.utils import PytorchPadMode, convert_data_type


class MinimumCropForeground(CropForeground):
    """Similar to MONAI's CropForeground but ensures that the cropped region has at least a minimum shape."""

    def __init__(self, min_shape: Sequence[int], **kwargs):
        super().__init__(**kwargs)
        self.min_shape = min_shape

    def compute_bounding_box(self, img: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the start points and end points of bounding box to crop.
        Ensure the bounding box has at least the shape specified in `self.min_shape`.
        Adjust bounding box coords to be divisible by `k`.

        """
        box_start, box_end = generate_spatial_bounding_box(
            img, self.select_fn, self.channel_indices, self.margin, self.allow_smaller
        )
        box_start_, *_ = convert_data_type(
            box_start, output_type=np.ndarray, dtype=np.int16, wrap_sequence=True
        )
        box_end_, *_ = convert_data_type(
            box_end, output_type=np.ndarray, dtype=np.int16, wrap_sequence=True
        )
        img_shape = np.asarray(img.shape[1:], dtype=np.int16)
        orig_spatial_size = box_end_ - box_start_

        # Ensure the spatial size is at least `self.min_shape`
        spatial_size = np.maximum(orig_spatial_size, np.asarray(self.min_shape))

        # Make the spatial size divisible by `k`
        spatial_size = np.asarray(
            compute_divisible_spatial_size(spatial_size.tolist(), k=self.k_divisible)
        )

        # Update box_start and box_end
        box_start_ = box_start_ - np.floor_divide(np.asarray(spatial_size) - orig_spatial_size, 2)
        box_end_ = box_start_ + spatial_size

        # Clip end so that it doesnt go over image boundaries
        box_start_ = np.maximum(box_start_, 0)
        box_end_ = np.minimum(box_end_, img_shape)

        return box_start_, box_end_


class MinimumCropForegroundd(CropForegroundd):
    """Similar to MONAI's CropForegroundd but ensures that the cropped region has at least a minimum shape."""

    def __init__(
        self,
        min_shape: Sequence[int],
        keys: KeysCollection,
        source_key: str,
        select_fn: Callable = is_positive,
        channel_indices: Union[IndexSelection, None] = None,
        margin: Union[Sequence[int], int] = 0,
        allow_smaller: bool = True,
        k_divisible: Union[Sequence[int], int] = 1,
        mode: SequenceStr = PytorchPadMode.CONSTANT,
        start_coord_key: Union[str, None] = "foreground_start_coord",
        end_coord_key: Union[str, None] = "foreground_end_coord",
        allow_missing_keys: bool = False,
        lazy: bool = False,
        **pad_kwargs,
    ):
        super().__init__(
            keys=keys,
            source_key=source_key,
            select_fn=select_fn,
            channel_indices=channel_indices,
            margin=margin,
            allow_smaller=allow_smaller,
            k_divisible=k_divisible,
            mode=mode,
            start_coord_key=start_coord_key,
            end_coord_key=end_coord_key,
            allow_missing_keys=allow_missing_keys,
            lazy=lazy,
            **pad_kwargs,
        )

        self.cropper = MinimumCropForeground(
            select_fn=select_fn,
            channel_indices=channel_indices,
            margin=margin,
            allow_smaller=allow_smaller,
            k_divisible=k_divisible,
            lazy=lazy,
            min_shape=min_shape,
            **pad_kwargs,
        )


class ConditionalAddChannel(Transform):
    """
    Adds a channel dimension to the input if it does not have one (ndim=3).
    """

    def __call__(self, data: NdarrayOrTensor) -> NdarrayOrTensor:
        # Check for 3 dimensions (Spatial only)
        if data.ndim == 3:
            # Add channel dimension at index 0
            # Slicing with None is valid for both Tensor and ndarray
            return data[None, ...]
        return data


class ConditionalAddChanneld(MapTransform):
    """
    Dictionary-based transform to conditionally add a channel dimension.
    """

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            if d[key].ndim == 3:
                d[key] = d[key][None, ...]
        return d
