"""Command-line argument parser for KITS classifier."""

import argparse
from pathlib import Path
from typing import Any


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for train/infer/eval modes."""
    parser = argparse.ArgumentParser(
        description="Renal Vision: Dynamic Multi-class Lesion Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="mode", help="Operation mode", required=True)

    # ========== TRAIN MODE ==========
    train_parser = subparsers.add_parser("train", help="Train a classifier")
    train_parser.add_argument(
        "--data", type=str, required=True, help="Path to CSV (image paths or feature CSV)"
    )
    train_parser.add_argument(
        "--model",
        type=str,
        choices=["logistic", "tree", "xgboost"],
        required=True,
        help="Model architecture to use",
    )
    train_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory (will contain model.pkl and metadata)",
    )
    train_parser.add_argument(
        "--features",
        type=str,
        nargs="+",
        default=None,
        help="List of features to use (default: all)",
    )
    train_parser.add_argument(
        "--min-voxels", type=int, default=10, help="Minimum lesion size in voxels (default: 10)"
    )
    train_parser.add_argument(
        "--tree-depth", type=int, default=5, help="Max depth for decision tree/xgboost (default: 5)"
    )
    # Dynamic class arguments
    train_parser.add_argument(
        "--n-classes",
        type=int,
        default=2,
        help="Number of classes (used only if training from images directly, ignored for feature CSV)",
    )
    train_parser.add_argument(
        "--class-names",
        type=str,
        nargs="+",
        help="Optional list of class names (e.g. 'Tumor' 'Cyst')",
    )
    train_parser.add_argument(
        "--label-map",
        type=str,
        help="Path to JSON file containing label mapping (e.g. {'2': 0, '3': 1})",
    )
    train_parser.add_argument(
        "--explain", action="store_true", help="Generate detailed model explanations"
    )
    train_parser.add_argument(
        "--rule-format",
        type=str,
        choices=["nested", "flat"],
        default="nested",
        help="Format for decision tree rules (default: nested)",
    )

    # ========== INFER MODE ==========
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--image", type=str, required=True, help="Path to CT image (.nii.gz)")
    infer_parser.add_argument(
        "--seg", type=str, required=True, help="Path to segmentation mask (.nii.gz)"
    )
    infer_parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model (.pkl)"
    )
    infer_parser.add_argument(
        "--multi-lesion",
        action="store_true",
        help="Process all lesions and output updated segmentation",
    )
    infer_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for result (required for --multi-lesion)",
    )
    infer_parser.add_argument(
        "--label-map",
        type=str,
        help="Path to JSON file containing label mapping for preprocessing",
    )
    infer_parser.add_argument(
        "--min-voxels", type=int, default=10, help="Minimum lesion size in voxels"
    )
    infer_parser.add_argument(
        "--uncertainty-threshold",
        type=float,
        default=0.5,
        help="Probability threshold. Unsure predictions are labeled as -1.",
    )

    # ========== EVAL MODE ==========
    eval_parser = subparsers.add_parser("eval", help="Evaluate model")
    eval_parser.add_argument(
        "--data", type=str, required=True, help="Path to CSV (image paths or feature CSV)"
    )
    eval_parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model (.pkl)"
    )
    eval_parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory for results"
    )
    eval_parser.add_argument(
        "--label-map",
        type=str,
        help="Path to JSON file containing label mapping for preprocessing",
    )
    eval_parser.add_argument(
        "--min-voxels", type=int, default=10, help="Minimum lesion size in voxels"
    )

    # Uncertainty handling
    uncertainty_group = eval_parser.add_mutually_exclusive_group()
    uncertainty_group.add_argument(
        "--find-threshold",
        action="store_true",
        help="Analyze uncertainty thresholds (for validation set only)",
    )
    uncertainty_group.add_argument(
        "--uncertainty-threshold",
        type=float,
        default=0.5,
        help="Probability threshold. Unsure predictions are labeled as -1.",
    )
    eval_parser.add_argument(
        "--explain",
        action="store_true",
        help="Generate detailed model explanations",
    )

    return parser


def validate_args(args: Any) -> None:
    """Validate parsed arguments."""
    if args.mode == "infer" and args.multi_lesion and args.output is None:
        raise ValueError("--output is required when using --multi-lesion")

    if args.mode in ["train", "eval"]:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if hasattr(args, "label_map") and args.label_map:
        if not Path(args.label_map).exists():
            raise FileNotFoundError(f"Label map file not found: {args.label_map}")
