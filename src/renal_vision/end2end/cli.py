"""
Command-line interface for the features module.
Acts as the entry point for batch feature extraction.
"""

import argparse

from renal_vision.shared.parser_config import get_base_parser


# ================= Logic =================
def run_train(args: argparse.Namespace) -> None:
    import json
    from pathlib import Path
    from typing import Dict, Optional

    from renal_vision.end2end.train import train
    from renal_vision.features.preprocessing import FMCIBPreprocessor

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

    label_map = load_label_map(args.label_map)
    preprocessor = FMCIBPreprocessor(label_map=label_map)

    lr = 1e-4

    train(
        csv_path=Path(args.data),
        output_dir=Path(args.output_dir),
        cache_dir=Path(args.cache_dir),
        preprocessor=preprocessor,
        min_volume=args.min_volume,
        batch_size=args.batch_size,
        lr=lr,
        epochs=args.epochs,
        num_workers=args.num_workers,
        pretrained=args.pretrained,
        cache_rate=args.cache_rate,
        no_image_caching=args.no_image_caching,
        validate=args.validate,
    )


# ================= Configuration =================
def config_extract(parser: argparse.ArgumentParser) -> None:
    """Configures an existing parser for 'extract' and binds the logic."""

    # I/O Arguments
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to input CSV (must contain 'image_path' and 'seg_path' columns)",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help="Path to cache directory",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save output weights",
    )

    # Flags
    parser.add_argument(
        "--no_image_caching", action="store_true", help="Disable caching of images during training"
    )

    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Initialise with pretrained weights and use reduced learning rate",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Use a sbuset of the data for validation",
    )

    # Configuration
    parser.add_argument("--batch-size", type=int, default=8, help="Batchsize for training")

    parser.add_argument(
        "--preprocessor",
        type=str,
        default="fmcib",
        choices=["fmcib"],
        help="Type of feature extractor to use (only supports fmcib at the moment).",
    )

    parser.add_argument(
        "--label-map",
        type=str,
        help="Path to JSON file containing label mapping (e.g. {'2': 1, '3': 2})",
    )
    parser.add_argument(
        "--min-volume",
        type=int,
        default=400,
        help="Minimum lesion size in mm^3 to process. Default: 400 mm^3",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs used for trainig",
    )

    # Multiprocessing
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers.",
    )

    # Memory
    parser.add_argument(
        "--cache_rate",
        type=float,
        default=0,
        help="Fraction of the data that can be cached in memory during traing. Value between 0 and 1. Default: 0",
    )

    parser.set_defaults(func=run_train)


def add_subparsers(subparsers: argparse._SubParsersAction) -> None:
    extract_parser = subparsers.add_parser(
        "train_end2end", help="Train a ResNet directly on the images."
    )
    config_extract(extract_parser)


# ================= Standalone CLI =================


def main() -> None:
    parser = get_base_parser("End2End Trainer")
    config_extract(parser)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
