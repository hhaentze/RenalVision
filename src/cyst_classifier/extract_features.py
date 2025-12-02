"""Script to extract and cache features from CT scans."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .features import Feature, extract_features
from .preprocessing import CTPreprocessor, extract_lesions


def create_extraction_parser() -> argparse.ArgumentParser:
    """Create argument parser for feature extraction."""
    parser = argparse.ArgumentParser(
        description="Extract and cache radiomics features from data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to CSV with columns: seg_path, image_path"
    )
    parser.add_argument("--output", type=str, required=True, help="Output path for feature CSV")
    parser.add_argument(
        "--min-voxels", type=int, default=10, help="Minimum lesion size in voxels (default: 10)"
    )
    parser.add_argument(
        "--features",
        type=str,
        nargs="+",
        default=None,
        help="List of features to extract",
    )
    parser.add_argument(
        "--label-map",
        type=str,
        help="Path to JSON file containing label mapping",
    )

    return parser


def load_label_map(json_path: Optional[str]) -> Dict[int, int]:
    if not json_path:
        return {0: 0}
    with open(json_path, "r") as f:
        data = json.load(f)
        return {int(k): int(v) for k, v in data.items()}


def extract_and_cache_features(
    data_csv: str,
    output_csv: str,
    min_voxels: int = 10,
    feature_list: Optional[List[Feature]] = None,
    label_map: Optional[Dict[int, int]] = None,
) -> None:
    """
    Extract features from all lesions and save to CSV.
    """
    print(f"Loading data from {data_csv}...")
    df = pd.read_csv(data_csv)

    # Defaults
    if feature_list is None:
        feature_list = list(Feature)
    if label_map is None:
        label_map = {0: 0}

    feature_names = [f.value for f in feature_list]
    all_rows: List[Dict[str, Any]] = []

    print("Processing scans and extracting lesions...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        case = row.get("case", f"case_{idx}")

        try:
            # Load and preprocess
            preprocessor = CTPreprocessor(label_map=label_map)
            image, seg, _ = preprocessor.process_files(
                row["image_path"],
                row["seg_path"],
            )

            lesions = extract_lesions(image, seg, min_voxels=min_voxels)

            # Sort by volume
            lesions_with_volume = []
            for l_img, l_mask, label in lesions:
                vol = np.sum(l_mask)
                lesions_with_volume.append((l_img, l_mask, label, vol))
            lesions_with_volume.sort(key=lambda x: x[3], reverse=True)

            for l_id, (l_img, l_mask, label, vol) in enumerate(lesions_with_volume, start=1):
                features = extract_features(l_img, l_mask, feature_list)

                row_dict = {
                    "case": case,
                    "lesion_id": l_id,
                    "label": label,
                    "volume_voxels": vol,
                }
                for fname in feature_names:
                    row_dict[fname] = features[fname]

                all_rows.append(row_dict)

        except Exception as e:
            print(f"Error: {e}")
            continue

    if not all_rows:
        print("No features extracted. Exiting.")
        return

    feature_df = pd.DataFrame(all_rows)
    feature_df.to_csv(output_csv, index=False)
    print(f"Saved {len(feature_df)} lesions to {output_csv}")


def main() -> None:
    parser = create_extraction_parser()
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    feature_list = None
    if args.features:
        feature_list = [Feature[f.upper()] for f in args.features]

    label_map = load_label_map(args.label_map)

    extract_and_cache_features(
        data_csv=args.data,
        output_csv=args.output,
        min_voxels=args.min_voxels,
        feature_list=feature_list,
        label_map=label_map,
    )


if __name__ == "__main__":
    main()
