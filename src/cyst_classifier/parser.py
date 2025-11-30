"""Command-line argument parser for KITS classifier."""

import argparse
from pathlib import Path


def create_parser():
    """Create argument parser for train/infer/eval modes."""
    parser = argparse.ArgumentParser(
        description="Cyst vs Tumor Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="mode", help="Operation mode", required=True)

    # ========== TRAIN MODE ==========
    train_parser = subparsers.add_parser("train", help="Train a classifier")
    train_parser.add_argument(
        "--data", type=str, required=True, help="Path to CSV with columns: seg_path, image_path"
    )
    train_parser.add_argument(
        "--model",
        type=str,
        choices=["logistic", "tree"],
        required=True,
        help="Model type: logistic regression or decision tree",
    )
    train_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory (will contain model.pkl and explanations/)",
    )
    train_parser.add_argument(
        "--features",
        type=str,
        nargs="+",
        default=None,
        help="List of features to use (default: all). Options: mean_hu, std_hu, cov, p10, p90, "
        "entropy, glcm_contrast, gradient_mag, sphericity, frac_below_20hu",
    )
    train_parser.add_argument(
        "--min-voxels", type=int, default=10, help="Minimum lesion size in voxels (default: 10)"
    )
    train_parser.add_argument(
        "--tree-depth", type=int, default=5, help="Max depth for decision tree (default: 5)"
    )
    train_parser.add_argument(
        "--explain", action="store_true", help="Generate detailed model explanations"
    )
    train_parser.add_argument(
        "--rule-format",
        type=str,
        choices=["nested", "flat"],
        default="nested",
        help="Format for decision tree rules: nested (if-else) or flat (list of paths) (default: nested)",
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
        "--no-label-mapping",
        action="store_true",
        help="Disable mapping of labels (1,2,3) to (0,1,1)",
    )
    infer_parser.add_argument(
        "--min-voxels", type=int, default=10, help="Minimum lesion size in voxels (default: 10)"
    )
    infer_parser.add_argument(
        "--uncertainty-threshold",
        type=float,
        default=0.5,
        help="Probability threshold for uncertain predictions (default: 0.5 = no unsure class, label=4)",
    )

    # ========== EVAL MODE ==========
    eval_parser = subparsers.add_parser("eval", help="Evaluate model")
    eval_parser.add_argument(
        "--data", type=str, required=True, help="Path to CSV with columns: seg_path, image_path"
    )
    eval_parser.add_argument(
        "--model", type=str, required=True, help="Path to trained model (.pkl)"
    )
    eval_parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory for results"
    )
    eval_parser.add_argument(
        "--min-voxels", type=int, default=10, help="Minimum lesion size in voxels (default: 10)"
    )

    # Uncertainty handling (mutually exclusive)
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
        help="Probability threshold for uncertain predictions (default: 0.5 = no unsure class)",
    )
    eval_parser.add_argument(
        "--explain",
        action="store_true",
        help="Generate detailed model explanations (includes uncertainty if threshold > 0.5)",
    )

    return parser


def validate_args(args):
    """Validate parsed arguments."""
    if args.mode == "infer" and args.multi_lesion and args.output is None:
        raise ValueError("--output is required when using --multi-lesion")

    if args.mode == "train":
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "eval":
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
