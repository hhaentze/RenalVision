"""Main entry point for Cyst classifier."""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .explainability import (
    explain_decision_tree,
    explain_logistic_regression,
)
from .features import Feature, extract_features
from .inference import Predictor
from .models import ModelBundle, predict_proba, train_model
from .parser import create_parser, validate_args
from .preprocessing import CTPreprocessor, extract_lesions
from .utils import (
    apply_uncertainty_threshold,
    compute_metrics,
    compute_metrics_with_uncertainty,
    plot_confusion_matrix,
    plot_multiclass_roc,
    print_metrics_report,
)


def load_label_map_and_names(json_path: Optional[str]) -> Tuple[Dict[int, int], Dict[int, str]]:
    """
    Load label mapping and optional class names from JSON file.

    Supported formats:
    1. Flat map: {"0": 0, "1": 1}
    2. Structured: {
         "map": {"2": 0, "3": 1},
         "names": {"0": "Tumor", "1": "Cyst"}
       }
    """
    if not json_path:
        return {}, {}

    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Label map file not found: {json_path}")

    with open(path, "r") as f:
        data = json.load(f)

    # Check if structured
    if "map" in data or "names" in data:
        mapping = {int(k): int(v) for k, v in data.get("map", {}).items()}
        names = {int(k): str(v) for k, v in data.get("names", {}).items()}
        return mapping, names
    else:
        # Fallback to simple flat map
        return {int(k): int(v) for k, v in data.items()}, {}


def is_feature_csv(df: pd.DataFrame) -> bool:
    """Check if CSV contains pre-extracted features."""
    feature_cols = ["mean_hu", "std_hu"]
    has_features = all(col in df.columns for col in feature_cols)
    has_metadata = "label" in df.columns
    return has_features and has_metadata


def load_feature_data(df: pd.DataFrame, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Load feature matrix X and label vector y."""
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features in CSV: {missing}")

    X = df[feature_names].values
    y = df["label"].values.astype(int)
    return X, y


def train_mode(args: Any) -> None:
    """Execute training workflow."""
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    label_map, json_class_names = load_label_map_and_names(args.label_map)

    # Features
    if args.features:
        feature_list = [Feature[f.upper()] for f in args.features]
    else:
        feature_list = list(Feature)
    feature_names = [f.value for f in feature_list]

    X: np.ndarray
    y: np.ndarray
    n_classes: int
    final_class_names: List[str] = []

    # --- 1. Load Data & Determine Classes ---
    if is_feature_csv(df):
        print("\n✓ Detected pre-extracted feature CSV")

        # Apply label map to feature CSV if provided
        if label_map:
            print("Applying label mapping to CSV labels...")
            # We map the 'label' column. Any label not in map is left as is (or should we drop?)
            # Assuming strictly mapped or identity if not present is risky.
            # Usually strict mapping is better for cleaning.
            # Here: if label in map, replace. Else keep.
            df["label"] = df["label"].map(lambda x: label_map.get(x, x))

        X, y = load_feature_data(df, feature_names)

        # Validate Labels
        unique_labels = sorted(np.unique(y))
        if unique_labels != list(range(len(unique_labels))):
            raise ValueError(
                f"Invalid labels in CSV after mapping: {unique_labels}. "
                "Labels must be contiguous integers starting at 0 (e.g. 0, 1, 2)."
            )

        n_classes = len(unique_labels)
        print(f"Detected {n_classes} classes: {unique_labels}")

        # Resolve Class Names
        # Priority 1: JSON Names
        if json_class_names:
            print("Using class names from JSON.")
            # Verify we have names for all classes
            if not all(i in json_class_names for i in unique_labels):
                warnings.warn(
                    "JSON class names provided but missing some classes. Filling with defaults."
                )
            final_class_names = [json_class_names.get(i, f"Class {i}") for i in range(n_classes)]

        # Priority 2: CSV 'class_name' column
        elif "class_name" in df.columns:
            print("Inferring class names from CSV...")
            inferred_names = {}
            for label in unique_labels:
                # Get names associated with this (potentially new) label
                names_for_label = df[df["label"] == label]["class_name"].unique()
                if len(names_for_label) > 1:
                    warnings.warn(
                        f"Label {label} has multiple names in CSV: {names_for_label}. "
                        f"This often happens when merging classes. Using '{names_for_label[0]}'. "
                        "Please specify explicit class names in the label map JSON to avoid this."
                    )
                    inferred_names[label] = str(names_for_label[0])
                elif len(names_for_label) == 1:
                    inferred_names[label] = str(names_for_label[0])
                else:
                    inferred_names[label] = f"Class {label}"

            final_class_names = [inferred_names[i] for i in range(n_classes)]

    else:
        print("\n⊙ Detected image path CSV (extracting features...)")
        # NOTE: load_and_preprocess applies the label_map internally for images
        all_features: List[List[float]] = []
        all_labels: List[int] = []
        preprocessor = CTPreprocessor(label_map=label_map)
        for _, row in tqdm(df.iterrows(), total=len(df)):
            try:
                image, seg, _ = preprocessor.process_files(
                    row["image_path"],
                    row["seg_path"],
                )
                lesions = extract_lesions(image, seg, min_voxels=args.min_voxels)

                for lesion_img, lesion_mask, label in lesions:
                    feats = extract_features(lesion_img, lesion_mask, feature_list)
                    all_features.append([float(feats[f.value]) for f in feature_list])
                    all_labels.append(label)

            except Exception as e:
                print(f"Warning: {e}")
                continue

        if not all_features:
            raise ValueError("No valid lesions found in the dataset.")

        X = np.array(all_features)
        y = np.array(all_labels)

        # Recalculate n_classes based on actual data found after mapping
        unique_found = sorted(np.unique(y))
        if unique_found != list(range(len(unique_found))):
            raise ValueError(f"Extracted labels are not contiguous starting at 0: {unique_found}")

        n_classes = len(unique_found)
        print(f"Extracted {n_classes} classes: {unique_found}")

        # Resolve names for image mode
        if json_class_names:
            final_class_names = [json_class_names.get(i, f"Class {i}") for i in range(n_classes)]

    # Priority 3: User args or Default
    if not final_class_names:
        if args.class_names:
            if len(args.class_names) != n_classes:
                raise ValueError(
                    f"Provided {len(args.class_names)} class names via CLI, but found {n_classes} classes."
                )
            final_class_names = args.class_names
        else:
            final_class_names = [f"Class {i}" for i in range(n_classes)]

    # --- 3. Train ---
    print(f"\nTraining {args.model} for {n_classes} classes...")
    print(f"Class mapping: {dict(enumerate(final_class_names))}")

    model_bundle = train_model(
        X,
        y,
        model_type=args.model,
        n_classes=n_classes,
        feature_names=feature_names,
        class_names=final_class_names,
        tree_max_depth=args.tree_depth,
    )

    # --- 4. Save ---
    output_path = Path(args.output_dir)
    model_path = output_path / "model.pkl"
    model_bundle.save(model_path)
    print(f"Model and metadata saved to {args.output_dir}")

    # --- 5. Explain (Legacy support for binary) ---
    if args.explain and n_classes == 2 and args.model in ["logistic", "tree"]:
        print("\nGenerating explanations (Binary Legacy Mode)...")
        explanation_dir = output_path / "explanations"
        if args.model == "logistic":
            explain_logistic_regression(model_bundle, X, y, output_dir=explanation_dir)
        elif args.model == "tree":
            explain_decision_tree(
                model_bundle, output_dir=explanation_dir, rule_format=args.rule_format
            )


def infer_mode(args):
    """
    Run inference on new data.

    Two modes:
    1. Single lesion: Return predicted class
    2. Multi-lesion: Return updated segmentation with predictions

    Note: Inference always requires actual images (not feature CSV).
    """
    print(f"Loading model from {args.model}...")
    predictor = Predictor(args.model)

    if args.multi_lesion:
        # Process all lesions
        print("Processing multiple lesions...")

        predictor.infer_mask(
            image=args.image,
            seg=args.seg,
            certainty_threshold=args.uncertainty_threshold,
            output=args.output,
        )

    else:
        # Single lesion mode
        print("Processing single lesion...")
        prediction = predictor.infer_lesion(
            image=args.image,
            seg=args.seg,
            certainty_threshold=args.uncertainty_threshold,
        )

        # Print result
        if prediction == -1:
            class_name = "Unsure"
        elif prediction == 0:
            class_name = "Tumor"
        elif prediction == 1:
            class_name = "Cyst"
        else:
            raise ValueError(f"Unknown predicted class: {prediction}")

        print(f"\nPrediction: {class_name}")


def eval_mode(args: Any) -> None:
    """Execute evaluation workflow."""
    model_bundle = ModelBundle.load(Path(args.model))
    df = pd.read_csv(args.data)
    feature_names = model_bundle.feature_names
    label_map, _ = load_label_map_and_names(args.label_map)

    # 1. Get Data
    if is_feature_csv(df):
        # Apply mapping here as well for consistency in eval
        if label_map:
            df["label"] = df["label"].map(lambda x: label_map.get(x, x))
        X, y_true = load_feature_data(df, feature_names)
    else:
        print("Extracting features from images...")
        all_feats: List[List[float]] = []
        all_lbls: List[int] = []
        feature_list = [Feature(f) for f in feature_names]

        preprocessor = CTPreprocessor(label_map=label_map)
        for _, row in tqdm(df.iterrows(), total=len(df)):
            try:
                img, seg, _ = preprocessor.process_files(
                    row["image_path"],
                    row["seg_path"],
                )
                for l_img, l_mask, lbl in extract_lesions(img, seg, min_voxels=args.min_voxels):
                    d = extract_features(l_img, l_mask, feature_list)
                    all_feats.append([float(d[f]) for f in feature_names])
                    all_lbls.append(lbl)
            except Exception as e:
                print(f"Warning: {e}")
                continue

        X = np.array(all_feats)
        y_true = np.array(all_lbls)

    if len(y_true) == 0:
        print("No valid data found for evaluation.")
        return

    # 2. Predict
    y_proba = predict_proba(model_bundle, X)

    # 3. Metrics
    threshold = args.uncertainty_threshold
    y_pred, _ = apply_uncertainty_threshold(y_proba, threshold)

    use_uncertainty = threshold > 0.5
    output_dir = Path(args.output_dir)

    if use_uncertainty:
        metrics = compute_metrics_with_uncertainty(
            y_true, y_pred, y_proba, model_bundle.class_names
        )
        print_metrics_report(metrics, model_bundle.class_names)
        if "confusion_matrix_with_unsure" in metrics:
            plot_confusion_matrix(
                metrics["confusion_matrix_with_unsure"],  # type: ignore
                model_bundle.class_names,
                str(output_dir / "confusion_matrix.png"),
            )
    else:
        y_pred_hard = y_proba.argmax(axis=1)
        metrics = compute_metrics(y_true, y_pred_hard, y_proba, model_bundle.class_names)
        print_metrics_report(metrics, model_bundle.class_names)
        plot_confusion_matrix(
            metrics["confusion_matrix"],  # type: ignore
            model_bundle.class_names,
            str(output_dir / "confusion_matrix.png"),
        )

    # 4. ROC Curves (Multi-class)
    # We always plot certain prediction ROCs if probabilities are available
    if y_proba is not None:
        if use_uncertainty:
            # Filter unsure for ROC calculation?
            # Standard practice: ROC uses raw probabilities. Unsure threshold is a post-processing decision step.
            # However, y_true must match.
            # We can plot ROC on the full set using raw probabilities vs true labels.
            pass

        plot_multiclass_roc(
            y_true,
            y_proba,
            n_classes=model_bundle.n_classes,
            class_names=model_bundle.class_names,
            output_path=str(output_dir / "roc_curves.png"),
        )

    # Save raw metrics
    with open(output_dir / "metrics.txt", "w") as f:
        f.write(str(metrics))


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    validate_args(args)

    if args.mode == "train":
        train_mode(args)
    elif args.mode == "infer":
        infer_mode(args)
    elif args.mode == "eval":
        eval_mode(args)


if __name__ == "__main__":
    main()
