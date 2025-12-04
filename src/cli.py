import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from features.dataset import FeatureDatasetProcessor
from features.preprocessing import CTPreprocessor
from features.radiomics import RadiomicsExtractor
from modeling import eval, inference, train


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RenalVision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to run")

    # ================= EXTRACT =================
    extract_parser = subparsers.add_parser("extract", help="Extract features from a dataset")
    extract_parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to input CSV (must contain 'image_path' and 'seg_path' columns)",
    )
    extract_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save output Parquet file (e.g., features.parquet)",
    )
    extract_parser.add_argument(
        "--extractor",
        type=str,
        default="radiomics",
        choices=["radiomics"],
        help="Type of feature extractor to use.",
    )
    extract_parser.add_argument(
        "--label-map",
        type=str,
        help="Path to JSON file containing label mapping (e.g. {'2': 1, '3': 2})",
    )
    extract_parser.add_argument(
        "--min-voxels", type=int, default=10, help="Minimum lesion size in voxels to process."
    )
    extract_parser.add_argument(
        "--augment",
        type=int,
        default=0,
        help="Number of augmented copies to generate per sample (default: 0).",
    )
    extract_parser.add_argument(
        "--normalize",
        action="store_true",
        help="If set, normalize image intensities to [0, 1]. Default is False (preserve HU).",
    )

    # ================= TRAIN =================
    train_parser = subparsers.add_parser("train", help="Train a model on extracted features")
    train_parser.add_argument("--data", required=True, help="Path to training features (Parquet)")
    train_parser.add_argument("--output-dir", required=True, help="Folder to save model.pkl")
    train_parser.add_argument(
        "--model", choices=["logistic", "tree", "xgboost"], default="logistic"
    )
    train_parser.add_argument(
        "--class-config", help="JSON map of class names (e.g. {'0': 'Tumor'})"
    )
    train_parser.add_argument("--extractor-config", help="JSON config from features (optional)")
    train_parser.add_argument("--depth", type=int, default=5, help="Tree depth")

    # ================= EVAL =================
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained model")
    eval_parser.add_argument("--data", required=True, help="Path to test features (Parquet)")
    eval_parser.add_argument("--model", required=True, help="Path to trained model.pkl")
    eval_parser.add_argument("--output-dir", required=True, help="Folder to save results")

    # ================= INFER =================
    infer_parser = subparsers.add_parser("infer", help="Run inference on a new image")
    infer_parser.add_argument("--image", required=True, help="Path to CT image")
    infer_parser.add_argument("--seg", required=True, help="Path to segmentation mask")
    infer_parser.add_argument("--model", required=True, help="Path to trained model.pkl")
    infer_parser.add_argument("--output", help="Path to save results CSV (optional)")

    return parser


def load_label_map(json_path: Optional[str]) -> Dict[int, int]:
    """Load label mapping from JSON file."""
    if not json_path:
        return {0: 0}

    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Label map file not found: {json_path}")

    with open(path, "r") as f:
        data = json.load(f)
        # Ensure keys/values are integers
        return {int(k): int(v) for k, v in data.items()}


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    if not Path(args.data).exists():
        raise FileNotFoundError(f"Input CSV not found: {args.data}")

    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} rows from {args.data}")

    # 2. Configure Preprocessor
    label_map = load_label_map(args.label_map)
    preprocessor = CTPreprocessor(label_map=label_map, normalize=args.normalize)

    # 3. Instantiate Extractor
    if args.extractor == "radiomics":
        extractor = RadiomicsExtractor(
            preprocessor=preprocessor,
            min_voxels=args.min_voxels,
            feature_names=None,  # Default to all
        )
    else:
        # Placeholder for future extractors
        raise ValueError(f"Unknown extractor type: {args.extractor}")

    # 4. Run Batch Processing
    processor = FeatureDatasetProcessor(extractor)

    processor.process_dataset(input_df=df, output_path=args.output, augment_count=args.augment)

    if args.command == "train":
        train.run_training(
            data_path=args.data,
            output_dir=args.output_dir,
            model_type=args.model,
            class_config_path=args.class_config,
            extractor_config_path=args.extractor_config,
            tree_max_depth=args.depth,
        )

    elif args.command == "eval":
        eval.run_evaluation(
            data_path=args.data,
            model_path=args.model,
            output_dir=args.output_dir,
        )

    elif args.command == "infer":
        predictor = inference.LesionPredictor(model_path=args.model)

        if args.output:
            # Multi-lesion / file output mode
            predictor.infer_mask(args.image, args.seg, args.output)
        else:
            # Single lesion / print mode
            try:
                result = predictor.infer_lesion(args.image, args.seg)
                print("\n Prediction Result:")
                print("-" * 30)
                print(f" Class:      {result['class_name']} (ID: {result['class_id']})")
                print(f" Confidence: {result['confidence']:.2%}")
                print(f" Volume:     {result['volume_voxels']} voxels")
                print("-" * 30)
            except ValueError as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()
