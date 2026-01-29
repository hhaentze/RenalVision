"""
Inference logic.
Reconstructs the feature pipeline from the trained model configuration
and generates predictions for new images.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from monai.data import MetaTensor
from monai.transforms import SaveImage
from scipy import ndimage

from renal_vision.bundles import ImplementedModels, load_model_bundle, suggest_similar_enum
from renal_vision.features.base_preprocessor import ImageLike
from renal_vision.features.embeddings_radiomics import RadiomicsExtractor
from renal_vision.features.preprocessing import (
    CTPreprocessor,
    FMCIBPreprocessor,
)

from .models import ModelBundle, predict, predict_proba


class LesionPredictor:
    """
    End-to-end predictor that wraps:
    1. Feature Extraction (recreated from model config)
    2. Classification (using trained model)
    """

    def __init__(self, model_identifier: Union[str, Path, ImplementedModels]):
        # Load model from bundle zoo
        if isinstance(model_identifier, ImplementedModels) or (
            model_identifier in ImplementedModels.__members__
        ):
            self.bundle = load_model_bundle(model_identifier)

        # Load model from custom path
        elif Path(model_identifier).is_file():
            self.bundle = ModelBundle.load(model_identifier)

        else:
            # Try to suggest a similar model identifier
            similar_model = suggest_similar_enum(model_identifier, ImplementedModels)
            if similar_model:
                raise ValueError(
                    f"Unknown model identifier: '{model_identifier}'. "
                    f"Did you mean '{similar_model.name}'?"
                )
            else:
                raise ValueError(f"Model not found: '{model_identifier}'.")

        # Reconstruct the feature extraction pipeline used during training
        self.extractor = self._reconstruct_extractor(self.bundle.extractor_config)

    def _reconstruct_extractor(self, config: Dict[str, Any]) -> Any:
        """
        Factory method to instantiate the correct extractor from config dictionary.
        """

        # 1. Reconstruct Preprocessor
        prep_config = config["preprocessor"]
        args = {k: v for k, v in prep_config.items() if k != "name"}
        if prep_config["name"] == "CTPreprocessor":
            preprocessor = CTPreprocessor(**args)
        elif prep_config["name"] == "FMCIBPreprocessor":
            preprocessor = FMCIBPreprocessor(**args)
        elif prep_config["name"] == "StaticCropPreprocessor":
            preprocessor = FMCIBPreprocessor(**args)
        else:
            raise ValueError(f"Unknown preprocessort type in model config: {prep_config['name']}")

        # 2. Instantiate Extractor
        extractor_type = config["type"]
        if extractor_type == "radiomics":
            # Filter out keys that aren't arguments to __init__
            valid_keys = {"feature_names", "min_volume"}
            ext_kwargs = {k: v for k, v in config.items() if k in valid_keys}
            return RadiomicsExtractor(preprocessor=preprocessor, **ext_kwargs)
        else:
            raise ValueError(f"Unknown extractor type in model config: {extractor_type}")

    def _find_components(self, seg: ImageLike) -> Tuple[MetaTensor, int]:
        """
        Utility method to find connected components in a segmentation mask.

        Returns:
        - a mask where each connected component has a unique integer ID.
        - number of connected components found
        """

        seg_mask = self.extractor.preprocessor._prepare_data_point(seg)
        comp_mask = seg_mask * 0  # new empty meta tensor

        # Get all unique classes (excluding background 0)
        classes = np.unique(seg_mask)
        classes = classes[classes > 0]

        comp_count = 0
        for class_id in classes:
            # Create binary mask for this class
            class_mask = seg == class_id

            # Find connected components
            labeled_mask, num_comp = ndimage.label(class_mask)
            for class_comp_id in range(1, num_comp + 1):
                comp_count += 1
                comp_mask[labeled_mask == class_comp_id] = comp_count

        return comp_mask, comp_count

    def infer_lesion(
        self,
        image: ImageLike,
        seg: ImageLike,
    ) -> Dict[str, Any]:
        """
        Predict the class of a single lesion (or the largest lesion in the mask).
        [Important] Class IDs start at 0

        Returns a dictionary with the prediction and probability.
        """

        # 1. Check Number of Lesions
        _, num_lesions = self._find_components(seg)
        if num_lesions == 0:
            raise ValueError("No lesions found in input segmentation.")
        if num_lesions > 1:
            raise ValueError(
                f"Found {num_lesions} different lesions.",
                "For predicting more than one lesion please use infer_mask.",
            )

        # 2. Extract Features
        lesion_features = self.extractor.extract(image, seg, augment=False)
        if num_lesions == 0:
            raise ValueError(
                "Extractor could not handle lesion. Volume might be too small.",
                f"Supported min_volume: {self.extractor.min_volume}",
            )
        if num_lesions > 1:
            raise Exception("This should never happen.")

        target_lesion = lesion_features[0]

        # 2. Prepare Feature Vector
        # Extract columns in the exact order the model expects
        try:
            X = np.array([target_lesion[f] for f in self.bundle.feature_names])
            X = X[None]
        except KeyError as e:
            raise KeyError(f"Feature extraction mismatch. Missing feature: {e}")

        # 3. Predict
        pred_proba = predict_proba(self.bundle, X)
        pred_class = int(np.argmax(pred_proba))
        class_name = self.bundle.class_names.get(pred_class, f"Class {pred_class}")

        return {
            "class_id": pred_class,
            "class_name": class_name,
            "confidence": float(np.max(pred_proba)),
            "probability": pred_proba.tolist(),
            "volume_voxels": target_lesion.get("volume_voxels"),
        }

    def infer_mask(
        self,
        image: ImageLike,
        seg: ImageLike,
        output_path: Optional[str | Path] = None,
    ) -> MetaTensor:
        """
        Predict classes for all lesions in a segmentation mask.
        [Important] Class IDs start at 1 (0 is background class)

        Args:
            image: Path to CT image or Monai MetaTensor.
            seg: Path to segmentation mask or Monai MetaTensor.
            output_path: Optional path to save the result.

        Returns:
            np.ndarray: The predicted segmentation mask (same shape as input).
        """
        # 1. Load/Cast segmentation of to be classified target lesions
        seg_obj = self.extractor.preprocessor._prepare_data_point(seg)

        # 2. separate connected components in source mask and asign unique ids
        # (we need those for step 4)
        labeled_mask, num_comp = self._find_components(seg_obj)
        print(f"Extracting features for {num_comp} lesions")

        # 3. extract features
        lesion_features = self.extractor.extract(image, labeled_mask, augment=False)

        # 4. predict and map predicitons to output mask
        prediction_mask = seg_obj * 0  # new empty meta tensor

        for features in lesion_features:
            # Extract columns in the exact order the model expects
            try:
                X = np.array([features[f] for f in self.bundle.feature_names])
                X = X[None]
            except KeyError as e:
                raise KeyError(f"Feature extraction mismatch. Missing feature: {e}")
            prediction = predict(self.bundle, X)[0]

            # the lesion ids that we created in (2) are now stored in features["class_id"]
            # attention: indexing of features and prediction start at 0, not 1
            if features["class_id"] in range(0, num_comp):
                prediction_mask[labeled_mask == features["class_id"] + 1] = (
                    prediction + 1
                )  # +1 to avoid 0 background
            else:
                raise ValueError(f"Unknown class id: {features['class_id']}")

        # 5. Optional Save
        if output_path:
            out_p = Path(output_path)

            # Robust extension handling (for cases like .nii.gz)
            extensions = "".join(out_p.suffixes)
            stem = out_p.name.replace(extensions, "")
            prediction_mask.meta["filename_or_obj"] = stem

            saver = SaveImage(
                output_dir=out_p.parent,
                output_postfix="",  # Don't append "_trans" etc.
                output_ext=extensions,  # Force exact extension (.mha, .nii.gz)
                separate_folder=False,  # Don't create a subfolder named 'result'
                resample=False,  # We are providing the grid
                print_log=True,
            )
            saver(prediction_mask)

        return prediction_mask
