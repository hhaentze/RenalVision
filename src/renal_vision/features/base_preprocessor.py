"""
Base preprocessing logic using MONAI.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple, Union, cast

import numpy as np
import torch
from monai.data import MetaTensor
from monai.transforms import (
    Compose,
    LoadImage,
    MapLabelValue,
    MapTransform,
    Randomizable,
)
from scipy import ndimage

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
        crop_transforms: List[Any] = [],
        aug_transforms: List[Any] = [],
        intensity_transforms: List[Any] = [],
    ):
        """Initializes the preprocessor with given transform lists.

        Note:
        - lazy transforms in crop_transform should be at the end
        - lazy transforms in aug_transforms should be at the start
        """
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
        Load -> (augment)-> Intensity Normalize -> Return

        Returns: image, seg
        """

        # Load Inputs
        data = self._prepare_data(image, seg)

        # 2. Build Pipeline (cast strictly for MyPy)
        transforms = list(self.base_transforms)
        if augment:
            transforms.extend(cast(List[Any], self.aug_transforms))
        pipeline = Compose(transforms)

        # 3. Apply Transforms
        data = pipeline(data)

        return data["image"], data["seg"]

    def filter_components(
        self,
        seg: MetaTensor,
        min_volume: int,
    ) -> Tuple[MetaTensor, List[Dict[str, int]]]:
        """
        Filter for all components larger than min_volume

        Returns:
        - MetaTensor with all valid components, each with a unique class id>0
        - List with metadata for each component
        """

        # Get all unique classes (excluding background 0)
        classes = np.unique(seg)
        classes = classes[classes > 0]

        # calculate voxel_volume
        affine = seg.meta["affine"]
        voxel_volume = torch.abs(torch.linalg.det(affine[:3, :3]))

        # define output variables
        component_mask = MetaTensor(torch.zeros_like(seg), meta=seg.meta.copy())
        metadata_list = []

        comp_counter = 0
        for c_id in classes:
            # Create binary mask for this class
            class_mask = seg == c_id

            # Find connected components
            labeled_mask, num_comp = ndimage.label(class_mask)
            labeled_mask = torch.tensor(labeled_mask)

            for comp_id in range(1, num_comp + 1):
                volume = int((labeled_mask == comp_id).sum() * voxel_volume)
                if volume < min_volume:
                    continue  # skip small components

                comp_counter += 1
                component_mask[labeled_mask == comp_id] = comp_counter

                metadata = {
                    "class_id": int(c_id),
                    "volume": volume,
                }
                metadata_list.append(metadata)

        # Sort entries by volume
        # (this makes it to constiently compare different preprocesser that might use different image orientations)
        sorted_indices = sorted(
            range(len(metadata_list)), key=lambda i: metadata_list[i]["volume"], reverse=True
        )
        sorted_metadata_list = [metadata_list[i] for i in sorted_indices]

        # Create ID mapping (+1 becase the list starts indexing at 0 but the array at 1)
        id_mapping = {old_idx + 1: new_idx + 1 for new_idx, old_idx in enumerate(sorted_indices)}
        mapper = MapLabelValue(orig_labels=id_mapping.keys(), target_labels=id_mapping.values())  # type: ignore [arg-type]
        sorted_component_mask = mapper(component_mask)

        return sorted_component_mask, sorted_metadata_list

    def _crop_pipeline(self, augment: bool = False, normalize: bool = True) -> List[Any]:
        """crop -> (augment) -> normalize"""
        transforms = self.crop_transforms.copy()
        if augment:
            transforms += self.aug_transforms
        if normalize:
            transforms += self.intensity_transforms
        return transforms

    def stream_components(
        self,
        image: MetaTensor,
        seg: MetaTensor,
        min_volume: int = 100,
        augment: bool = False,
        normalize: bool = True,
    ) -> Generator[Tuple[np.ndarray, np.ndarray, dict[str, int]], None, None]:
        """
        Returns generator over individual connected components in the segmentation.

        Returns: component_image, component_mask, metadata(class_id, class_intern_id, volume)
        """

        # filter components
        valid_component_mask, metadata_list = self.filter_components(seg, min_volume)

        # define pipeline
        pipeline = Compose(self._crop_pipeline(augment=augment, normalize=normalize))

        for i, metadata in enumerate(metadata_list, start=1):
            # select component
            component_mask = seg * 0
            component_mask[valid_component_mask == i] = 1

            # Apply cropping transforms
            data = {"image": image, "seg": component_mask}
            data = pipeline(data)
            cropped_image, cropped_mask = self._finalize_numpy(data)

            yield cropped_image, cropped_mask, metadata
            del cropped_image, cropped_mask, component_mask

    def select_component(
        self,
        image: MetaTensor,
        seg: MetaTensor,
        i: int,
        min_volume: int = 100,
        augment: bool = False,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, dict[str, int]]:
        """
        Return i:th component

        Returns: component_image, component_mask, metadata(class_id, class_intern_id, volume)
        """

        # filter components
        valid_component_mask, metadata_list = self.filter_components(seg, min_volume)
        if i >= len(metadata_list):
            raise IndexError(
                f"{i}th component out of bounds for number of components: {len(metadata_list)}"
            )

        # define pipeline
        pipeline = Compose(self._crop_pipeline(augment=augment, normalize=normalize))

        # select component and metadata
        component_mask = seg * 0
        component_mask[valid_component_mask == i + 1] = 1
        metadata = metadata_list[i]

        # Apply cropping transforms
        data = {"image": image, "seg": component_mask}
        data = pipeline(data)
        cropped_image, cropped_mask = self._finalize_numpy(data)

        return cropped_image, cropped_mask, metadata

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        pass

    ###### For Compatibility with Monai Datasets ######

    def LoadCase(self):
        """Returns a configured LoadCase transform."""
        return _LoadCaseTransform(parent=self)

    def SelectComponent(self, min_volume: int, augment: bool = False):
        """
        Returns a configured SelectComponent transform.
        Allows overriding defaults via arguments.
        """
        return _SelectComponentTransform(parent=self, min_volume=min_volume, augment=augment)


######Factory Classes For Compatibility with Monai Datasets ######


class _LoadCaseTransform(MapTransform):
    """Deterministic: extract specific lesion from cached volume"""

    def __init__(self, parent):
        super().__init__(keys=["image_path", "seg_path"])
        self.parent = parent

    def __call__(self, data):
        image, seg = self.parent(data["image_path"], data["seg_path"])
        data["image"] = image
        data["seg"] = seg
        return data


class _SelectComponentTransform(MapTransform, Randomizable):
    """Random: augment lesion"""

    def __init__(self, parent, min_volume, augment):
        super().__init__(keys=["image", "seg", "lesion_id"])
        self.parent = parent
        self.min_volume = min_volume
        self.augment = augment

    def __call__(self, data):
        lesion, lesion_seg, _ = self.parent.select_component(
            data["image"],
            data["seg"],
            i=data["lesion_id"],
            augment=self.augment,
            min_volume=self.min_volume,
        )

        data["image"] = lesion
        data["seg"] = lesion_seg
        data["label"] = data["class_id"]

        return data
