"""
Command-line interface for the features module.
Acts as the entry point for batch feature extraction.
"""

import argparse

from renal_vision.shared.parser_config import get_base_parser


# ================= Logic =================
def run_extract(args: argparse.Namespace) -> None:
    import json
    from pathlib import Path
    from typing import Dict, Optional

    import pandas as pd

    from .base_extractor import BaseFeatureExtractor
    from .dataset import FeatureDatasetProcessor
    from .preprocessing import BasePreprocessor

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

    # 1. Load Data
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Input CSV not found: {args.data}")

    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} rows from {args.data}")

    # 2. Configure Preprocessor
    # We construct this explicitly to pass CLI arguments
    label_map = load_label_map(args.label_map)

    # 3. Instantiate Extractor
    preprocessor: BasePreprocessor
    extractor: BaseFeatureExtractor
    if args.extractor == "radiomics":
        from .embeddings_radiomics import RadiomicsExtractor
        from .preprocessing import CTPreprocessor

        preprocessor = CTPreprocessor(label_map=label_map, normalize=False)
        extractor = RadiomicsExtractor(
            preprocessor=preprocessor,
            min_volume=args.min_volume,
            feature_names=None,  # Default to all
        )

    elif args.extractor == "embeddings_fmcib":
        from .embeddings_fmcib import FMCIBExtractor
        from .preprocessing import FMCIBPreprocessor

        preprocessor = FMCIBPreprocessor(label_map=label_map, normalize=True)
        extractor = FMCIBExtractor(
            preprocessor=preprocessor,
            min_volume=args.min_volume,
        )

    elif args.extractor == "embeddings_mevis":
        from .embeddings_mevis import MevisExtractor
        from .preprocessing import MevisPreprocessor

        preprocessor = MevisPreprocessor(label_map=label_map, normalize=True)
        extractor = MevisExtractor(
            preprocessor=preprocessor,
            min_volume=args.min_volume,
        )
    elif args.extractor == "embeddings_ctfm":
        from .embeddings_ctfm import CTFMExtractor
        from .preprocessing import CTFMPreprocessor

        preprocessor = CTFMPreprocessor(label_map=label_map, normalize=True)
        extractor = CTFMExtractor(
            preprocessor=preprocessor,
            min_volume=args.min_volume,
        )
    else:
        # Placeholder for future extractors
        raise ValueError(f"Unknown extractor type: {args.extractor}")

    # 4. Run Batch Processing
    processor = FeatureDatasetProcessor(extractor)

    processor.process_dataset(
        input_df=df,
        output_path=args.output,
        augment_count=args.augment,
        num_jobs=args.num_jobs,
        cores_per_job=args.cores_per_job,
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
        "--output",
        type=str,
        required=True,
        help="Path to save output Parquet file (e.g., features.parquet)",
    )

    # Configuration
    parser.add_argument(
        "--extractor",
        type=str,
        default="radiomics",
        choices=["radiomics", "embeddings_fmcib", "embeddings_mevis"],
        help="Type of feature extractor to use.",
    )
    parser.add_argument(
        "--label-map",
        type=str,
        help="Path to JSON file containing label mapping (e.g. {'2': 1, '3': 2})",
    )
    parser.add_argument(
        "--min-volume", type=int, default=100, help="Minimum lesion size in voxels to process."
    )

    # Augmentation
    parser.add_argument(
        "--augment",
        type=int,
        default=0,
        help="Number of augmented copies to generate per sample (default: 0).",
    )

    # Preprocessing Overrides
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="If set, normalize image intensities to [0, 1]. Default is False (preserve HU).",
    )
    parser.set_defaults(func=run_extract)

    # Multiprocessing
    parser.add_argument(
        "--num_jobs",
        type=int,
        default=4,
        help="Number of extraction jobs that can be run in parallel.",
    )
    parser.add_argument("--cores_per_job", type=int, default=1, help="Number of cores per job.")


def add_subparsers(subparsers: argparse._SubParsersAction) -> None:
    extract_parser = subparsers.add_parser(
        "extract", help="Extract features from a dataset into a parquet file."
    )
    config_extract(extract_parser)


# ================= Standalone CLI =================


def main() -> None:
    parser = get_base_parser("Feature Extraction Module")
    config_extract(parser)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
