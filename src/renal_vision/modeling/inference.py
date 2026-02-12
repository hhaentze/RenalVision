"""
Inference logic.
Reconstructs the feature pipeline from the trained model configuration
and generates predictions for new images.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from monai.data import MetaTensor
from monai.transforms import SaveImage

from renal_vision.bundles import ImplementedModels, load_model_bundle, suggest_similar_enum
from renal_vision.features.base_preprocessor import ImageLike
from renal_vision.features.preprocessing import (
    CTFMPreprocessor,
    CTPreprocessor,
    FMCIBPreprocessor,
    MevisPreprocessor,
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
        kwargs = {k: v for k, v in prep_config.items() if k != "name"}
        if prep_config["name"] == "CTPreprocessor":
            preprocessor = CTPreprocessor(**kwargs)
        elif prep_config["name"] == "FMCIBPreprocessor":
            preprocessor = FMCIBPreprocessor(**kwargs)
        elif prep_config["name"] == "MevisPreprocessor":
            preprocessor = MevisPreprocessor(**kwargs)
        elif prep_config["name"] == "CTFMPreprocessor":
            preprocessor = CTFMPreprocessor(**kwargs)
        else:
            raise ValueError(f"Unknown preprocessor type in model config: {prep_config['name']}")

        # 2. Instantiate Extractor
        # Filter out keys that aren't arguments to __init__
        valid_keys = {"feature_names", "min_volume"}
        ext_kwargs = {k: v for k, v in config.items() if k in valid_keys}
        extractor_type = config["type"]
        if extractor_type == "radiomics":
            from renal_vision.features.embeddings_radiomics import RadiomicsExtractor

            return RadiomicsExtractor(preprocessor=preprocessor, **ext_kwargs)
        elif extractor_type == "MevisEmbeddings":
            from renal_vision.features.embeddings_mevis import MevisExtractor

            return MevisExtractor(preprocessor=preprocessor, **ext_kwargs)
        elif extractor_type == "CTFM_embeddings":
            from renal_vision.features.embeddings_ctfm import CTFMExtractor

            return CTFMExtractor(preprocessor=preprocessor, **ext_kwargs)
        elif extractor_type == "fmcib":
            from renal_vision.features.embeddings_fmcib import FMCIBExtractor

            return FMCIBExtractor(preprocessor=preprocessor, **ext_kwargs)
        elif extractor_type == "ImageExtractor":
            from renal_vision.features.base_extractor import ImageExtractor

            return ImageExtractor(preprocessor=preprocessor, **ext_kwargs)
        else:
            raise ValueError(f"Unknown extractor type in model config: {extractor_type}")

    def filter_components(
        self,
        seg: ImageLike,
        min_volume: Optional[int] = None,
        strict: bool = True,
    ) -> Tuple[MetaTensor, List[Dict[str, Any]]]:
        """
        Filter for connected components with a minimum target volume. By default min_volume will be loaded from the extractor config

        args:
        - strict: due to rounding erros the volume of lesions may slightly change before/after transformation (e.g. 403->399).
                This cause logic errors as a previously filtered and accepted lesion may be rejected in the extraction step.
                To avoid this, if strict is true, we increase the min_volume used for filtering by 5%

        Returns:
        - MetaTensor with all valid components, each with a unique class id>0
        - List with metadata for each component
        """

        seg_mask = self.extractor.preprocessor._prepare_data_point(seg)
        if min_volume is None:
            min_volume = self.extractor.min_volume
        if strict:
            min_volume = int(min_volume * 1.05)

        return self.extractor.preprocessor.filter_components(seg_mask, min_volume)

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
        seg_obj, metadata_list = self.filter_components(seg, min_volume=0)
        num_lesions = len(metadata_list)

        if num_lesions == 0:
            raise ValueError("No lesions found in input segmentation.")
        if num_lesions > 1:
            raise ValueError(
                f"Found {num_lesions} different lesions.",
                "For predicting more than one lesion please use infer_mask.",
            )
        metadata = metadata_list[0]

        min_volume = self.extractor.min_volume
        if metadata["volume"] < min_volume:
            raise ValueError(
                f"Lesion Volume ({metadata['volume']}) mm^3 too small.",
                f"Supported min_volume: {min_volume} mm^3.",
            )

        # 2. Extract Features
        lesion_features = self.extractor.extract(image, seg_obj, augment=False)
        if len(lesion_features) == 0:
            if metadata["volume"] * 0.8 < min_volume:
                raise ValueError(
                    f"Lesion Volume ({metadata['volume']}) mm^3 too close to threshold.",
                    "(Transformation may change calculated volume by around 5%)"
                    f"Supported min_volume: {min_volume} mm^3.",
                )
            else:
                raise Exception("This should never happen.")
        if len(lesion_features) > 1:
            raise Exception("This should never happen.")
        target_lesion = lesion_features[0]

        # 3. Prepare Feature Vector
        # Extract columns in the exact order the model expects
        try:
            X = np.array([target_lesion[f] for f in self.bundle.feature_names])
            X = X[None]
        except KeyError as e:
            raise KeyError(f"Feature extraction mismatch. Missing feature: {e}")

        # 4. Predict
        pred_proba = predict_proba(self.bundle, X)
        pred_class = int(np.argmax(pred_proba))
        class_name = self.bundle.class_names.get(pred_class, f"Class {pred_class}")

        return {
            "class_id": pred_class,
            "class_name": class_name,
            "confidence": float(np.max(pred_proba)),
            "probability": pred_proba.squeeze().tolist(),
            "volume": target_lesion["volume"],
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

        # 1. Filter for lesions, assign unique ids
        seg_obj, metadata_list = self.filter_components(seg, min_volume=0)

        # 2. Check volume
        min_volume = self.extractor.min_volume
        num_comp = len(metadata_list)
        n_valid = sum([meta["volume"] >= min_volume for meta in metadata_list])
        print(f"Found {num_comp} lesions")
        if n_valid < num_comp:
            print(
                f"Warning: {num_comp - n_valid} lesions are smaller than the minimal required volume of {min_volume} mm^3. They will be ignored"
            )
        if n_valid == 0:
            print(
                print(
                    f"Warning: All detected lesions are smaller than the minimal required volume of {min_volume} mm^3. Returned empty mask."
                )
            )
            return MetaTensor(torch.zeros_like(seg_obj), meta=seg_obj.meta.copy())

        # 3. extract features
        lesion_features = self.extractor.extract(image, seg_obj, augment=False)

        # 4. predict and map predicitons to output mask
        prediction_mask = MetaTensor(torch.zeros_like(seg_obj), meta=seg_obj.meta.copy())

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
                prediction_mask[seg_obj == features["class_id"] + 1] = (
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
