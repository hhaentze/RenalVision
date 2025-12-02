import warnings
from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np

from .features import Feature, extract_features
from .models import ModelBundle, predict_proba
from .preprocessing import CTPreprocessor, extract_lesions
from .utils import apply_uncertainty_threshold, get_all_lesion_components


class Predictor:
    def __init__(self, model_path: str | Path):
        self.model_bundle: ModelBundle = ModelBundle.load(model_path)
        self.feature_list = [Feature(fname) for fname in self.model_bundle.feature_names]

    def _load(
        self,
        image: str | Path | np.ndarray,
        seg: str | Path | np.ndarray,
        affine: Optional[np.ndarray] = None,
        label_map: Optional[dict[int, int]] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Wrapper to load and preprocess data.

        If the inputs are file paths, load and preprocess them.
        If they are numpy arrays, assume they are already preprocessed.
        """

        preprocessor = CTPreprocessor(label_map=label_map if label_map else {0: 0})

        if isinstance(image, (str, Path)) and isinstance(seg, (str, Path)):
            return preprocessor.process_files(image, seg)

        elif isinstance(image, np.ndarray) and isinstance(seg, np.ndarray):
            if affine is None:
                raise ValueError("Affine must be provided when passing numpy arrays.")
            return preprocessor.process_arrays(image, seg, affine)

        raise ValueError("image and seg must both be either file paths or numpy arrays.")

    def infer_lesion(
        self,
        image: str | Path | np.ndarray,
        seg: str | Path | np.ndarray,
        affine: Optional[np.ndarray] = None,
        certainty_threshold: float = 0.5,
        return_probability: bool = False,
        label_map: Optional[dict[int, int]] = None,
    ) -> int | float:
        """
        Infer the class of a single lesion in the provided image and segmentation.

        Args:
            image: Path to CT image (.nii.gz) or preprocessed numpy array
            seg: Path to segmentation (.nii.gz) or preprocessed numpy array
            affine: Affine matrix
            certainty_threshold: Probability threshold below which prediction is "unsure"
            return_probability: If True, return probability of cyst instead of class

        Returns:
            int: Predicted class (0=tumor, 1=cyst, -1=unsure) if return_probability is False
            float: Probability of cyst if return_probability is True
        """

        # load and preprocess
        image, seg, affine = self._load(image, seg, affine, label_map=label_map)
        lesions = extract_lesions(image, seg)
        if len(lesions) == 0:
            raise ValueError("No lesions found in the provided segmentation.")
        elif len(lesions) > 1:
            raise ValueError(
                "Multiple lesions found in the provided segmentation. Use infer_mask instead."
            )
        lesion_img, lesion_mask, _ = lesions[0]

        # Extract features
        features = extract_features(lesion_img, lesion_mask, self.feature_list)
        feature_vector = np.array([[features[f.value] for f in self.feature_list]])

        # Get probabilities
        pred_proba = predict_proba(self.model_bundle, feature_vector)[0]

        # Apply uncertainty threshold
        pred_with_unsure, max_prob = apply_uncertainty_threshold(
            pred_proba.reshape(1, -1), certainty_threshold
        )
        pred_class = pred_with_unsure[0]

        if return_probability:
            return pred_proba[1]  # probability of cyst
        else:
            return pred_class

    def infer_mask(
        self,
        image: str | Path | np.ndarray,
        seg: str | Path | np.ndarray,
        affine: Optional[np.ndarray] = None,
        label_map: Optional[dict[int, int]] = None,
        certainty_threshold: float = 0.5,
        output: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Infer classes for all lesions in the provided image and segmentation mask.

        Args:
            image: Path to CT image (.nii.gz) or preprocessed numpy array
            seg: Path to segmentation (.nii.gz) or preprocessed numpy array
            affine: Affine matrix
            certainty_threshold: Probability threshold below which prediction is "unsure"
            output: Path to save output segmentation (.nii.gz). If None, do not save.

        Returns:
            image: Preprocessed CT image (numpy array)
            output_seg: Segmentation with predicted classes (numpy array)
            affine: Affine matrix
        """

        # load and preprocess
        image, seg, affine = self._load(image, seg, affine, label_map=label_map)
        lesions = extract_lesions(image, seg)
        if len(lesions) == 0:
            raise ValueError("No lesions found in the provided segmentation.")
        print(f"Found {len(lesions)} valid lesions")

        # Get labeled components
        labeled_seg, num_components = get_all_lesion_components(seg)

        # Create output segmentation (0=background, 1=tumor, 2=cyst, -1=unsure)
        output_seg = np.zeros_like(seg, dtype=np.int16)

        component_id = 1
        for lesion_img, lesion_mask, _ in lesions:
            # Extract features
            features = extract_features(lesion_img, lesion_mask, self.feature_list)
            feature_vector = np.array([[features[f.value] for f in self.feature_list]])

            # Get probabilities
            pred_proba = predict_proba(self.model_bundle, feature_vector)[0]

            # Apply uncertainty threshold
            pred_with_unsure, max_prob = apply_uncertainty_threshold(
                pred_proba.reshape(1, -1), certainty_threshold
            )
            pred_class = pred_with_unsure[0]

            # Update output segmentation for this component
            output_seg[labeled_seg == component_id] = pred_class
            component_id += 1

        if output:
            if not affine:
                warnings.warn("Affine not provided, using identity matrix for saving NIfTI.")
                affine = np.eye(4)

            output_path = Path(output)
            output_nii = nib.Nifti1Image(output_seg, affine)
            nib.save(output_nii, output_path)
            print(f"Results saved to {output_path}")

        return image, output_seg, affine
