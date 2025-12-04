"""
Inference logic.
Reconstructs the feature pipeline from the trained model configuration
and generates predictions for new images.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from monai.transforms import LoadImage, SaveImage
from scipy import ndimage

from features.preprocessing import CTPreprocessor
from features.radiomics import RadiomicsExtractor
from modeling.models import ModelBundle, predict, predict_proba


class LesionPredictor:
    """
    End-to-end predictor that wraps:
    1. Feature Extraction (recreated from model config)
    2. Classification (using trained model)
    """

    def __init__(self, model_path: Union[str, Path]):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"Loading model bundle from {model_path}...")
        self.bundle = ModelBundle.load(self.model_path)

        # Reconstruct the feature extraction pipeline used during training
        self.extractor = self._reconstruct_extractor(self.bundle.extractor_config)

    def _reconstruct_extractor(self, config: Dict[str, Any]) -> Any:
        """
        Factory method to instantiate the correct extractor from config dictionary.
        Handles JSON type conversion (e.g., string keys back to int).
        """
        extractor_type = config.get("type")

        # 1. Reconstruct Preprocessor
        # The extractor config contains the preprocessor config
        prep_config = config.get("preprocessor", {})

        # JSON converts dict keys to strings. We must convert label_map keys back to int.
        if "label_map" in prep_config and prep_config["label_map"]:
            raw_map = prep_config["label_map"]
            prep_config["label_map"] = {int(k): int(v) for k, v in raw_map.items()}

        # Instantiate Preprocessor (currently only CTPreprocessor supported)
        preprocessor = CTPreprocessor(**prep_config)

        # 2. Instantiate Extractor
        if extractor_type == "radiomics":
            # Filter out keys that aren't arguments to __init__
            valid_keys = {"feature_names", "min_voxels"}
            ext_kwargs = {k: v for k, v in config.items() if k in valid_keys}

            return RadiomicsExtractor(preprocessor=preprocessor, **ext_kwargs)

        else:
            raise ValueError(f"Unknown extractor type in model config: {extractor_type}")

    def infer_lesion(
        self,
        image: Union[str, Path, np.ndarray],
        seg: Union[str, Path, np.ndarray],
        affine: Union[None, np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Predict the class of a single lesion (or the largest lesion in the mask).
        Returns a dictionary with the prediction and probability.
        """
        # 1. Extract Features (returns list of dicts, one per lesion)
        # Note: We disable augmentation during inference
        lesion_features = self.extractor.extract(
            image,
            seg,
            affine=affine,
            augment=False,
        )

        if not lesion_features:
            raise ValueError("No valid lesions found in input segmentation.")

        # We assume the user wants the primary lesion (ID 1), which is sorted first
        target_lesion = lesion_features[0]

        # 2. Prepare Feature Vector
        # Extract columns in the exact order the model expects
        try:
            vector = [target_lesion[f] for f in self.bundle.feature_names]
        except KeyError as e:
            raise KeyError(f"Feature extraction mismatch. Missing feature: {e}")

        X = np.array([vector])

        # 3. Predict
        class_idx = predict(self.bundle, X)[0]
        proba = predict_proba(self.bundle, X)[0]

        class_name = self.bundle.class_names.get(class_idx, f"Class {class_idx}")

        return {
            "lesion_id": target_lesion.get("lesion_id"),
            "class_id": int(class_idx),
            "class_name": class_name,
            "probability": proba.tolist(),
            "confidence": float(np.max(proba)),
            "volume_voxels": target_lesion.get("volume_voxels"),
        }

    def infer_mask(
        self,
        image: Union[str, Path, np.ndarray],
        seg: Union[str, Path, np.ndarray],
        affine: Optional[np.ndarray] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """
        Predict classes for all lesions in a segmentation mask.

        Args:
            image: Path to CT image or numpy array.
            seg: Path to segmentation mask or numpy array.
            affine: Affine matrix (required if inputs are numpy arrays).
            output_path: Optional path to save the result as NIfTI.

        Returns:
            np.ndarray: The predicted segmentation mask (same shape as input).
        """
        # 1. Load Original Header/Data
        if isinstance(seg, (str, Path)):
            loader = LoadImage(image_only=True, ensure_channel_first=True)
            seg_obj = loader(seg)
            # Use .numpy() to convert MetaTensor to numpy
            orig_affine = seg_obj.affine.numpy()
            orig_data = seg_obj.array.squeeze().astype(np.int32)
        else:
            if affine is None:
                raise ValueError("Affine required when inputs are numpy arrays.")
            orig_data = seg.astype(np.int32)
            orig_affine = affine

        # 2. Extract Features
        lesion_features = self.extractor.extract(image, seg, augment=False, affine=affine)

        prediction_mask = np.zeros_like(orig_data, dtype=np.int16)

        if not lesion_features:
            return prediction_mask

        # 3. Reconstruction Loop (Match World Coordinates)
        unique_classes = np.unique(orig_data)
        unique_classes = unique_classes[unique_classes > 0]

        for c_id in unique_classes:
            labeled_mask, num_components = ndimage.label(orig_data == c_id)

            for i in range(1, num_components + 1):
                component_mask = labeled_mask == i

                # A. Calculate Original Centroid (World Coords)
                cz, cy, cx = ndimage.center_of_mass(component_mask)
                voxel_coord = np.array([cx, cy, cz, 1.0])
                orig_center = (orig_affine @ voxel_coord)[:3]

                # B. Find Nearest Feature Match
                best_match = None
                min_dist = float("inf")

                for feat in lesion_features:
                    feat_center = np.array(
                        [
                            feat["centroid_world_x"],
                            feat["centroid_world_y"],
                            feat["centroid_world_z"],
                        ]
                    )
                    dist = float(np.linalg.norm(orig_center - feat_center))

                    if dist < min_dist:
                        min_dist = dist
                        best_match = feat

                # C. Predict & Paint
                match_threshold = 5.0  # mm
                if best_match and min_dist <= match_threshold:
                    vector = np.array([[best_match[f] for f in self.bundle.feature_names]])
                    pred_class = predict(self.bundle, vector)[0]
                    prediction_mask[component_mask] = int(pred_class)
                else:
                    # No match found within threshold; assign background
                    prediction_mask[component_mask] = -1
                    print(f"No match found for component {i} of class {c_id}; assigned -1.")

        # 4. Optional Save (using MONAI SaveImage)
        if output_path:
            out_p = Path(output_path)

            # Robust extension handling (for cases like .nii.gz)
            extensions = "".join(out_p.suffixes)
            stem = out_p.name.replace(extensions, "")

            saver = SaveImage(
                output_dir=out_p.parent,
                output_postfix="",  # Don't append "_trans" etc.
                output_ext=extensions,  # Force exact extension (.mha, .nii.gz)
                separate_folder=False,  # Don't create a subfolder named 'result'
                resample=False,  # We are providing the grid
                print_log=True,
            )

            # Add singleton channel dim: [D, H, W] -> [1, D, H, W]
            pred_with_channel = prediction_mask[None, ...]

            # Mock the metadata
            meta_map = {"filename_or_obj": stem, "affine": orig_affine}

            saver(pred_with_channel, meta_data=meta_map)
            print(f"Prediction mask saved to {out_p}")

        return prediction_mask
