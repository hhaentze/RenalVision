"""Script to extract and cache features from CT scans."""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from .preprocessing import load_and_preprocess, extract_lesions
from .features import Feature, extract_features


def create_extraction_parser():
    """Create argument parser for feature extraction."""
    parser = argparse.ArgumentParser(
        description="Extract and cache radiomics features from data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to CSV with columns: seg_path, image_path"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output path for feature CSV"
    )
    parser.add_argument(
        "--min-voxels", type=int, default=10,
        help="Minimum lesion size in voxels (default: 10)"
    )
    parser.add_argument(
        "--features", type=str, nargs="+", default=None,
        help="List of features to extract (default: all). Options: mean_hu, std_hu, cov, "
             "p10, p90, entropy, glcm_contrast, gradient_mag, sphericity, frac_below_20hu"
    )
    
    return parser


def extract_and_cache_features(data_csv, output_csv, min_voxels=10, feature_list=None):
    """
    Extract features from all lesions and save to CSV.
    
    Args:
        data_csv: Path to input CSV with seg_path, image_path columns
        output_csv: Path to output feature CSV
        min_voxels: Minimum lesion size in voxels
        feature_list: List of Feature enums to extract (default: all)
        
    Output CSV columns:
        - All feature values (mean_hu, std_hu, etc.)
        - label: Original label (2=tumor, 3=cyst)
        - lesion_id: ID within scan, ordered by volume (1=largest)
        - case: Case ID of original CT scan
    """
    print(f"Loading data from {data_csv}...")
    df = pd.read_csv(data_csv)
    
    # Determine which features to extract
    if feature_list is None:
        feature_list = list(Feature)
    
    feature_names = [f.value for f in feature_list]
    print(f"Extracting {len(feature_names)} features: {feature_names}")
    
    # Collect all lesion features
    all_rows = []
    
    print("Processing scans and extracting lesions...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        case = row["case"]
        
        try:
            # Load and preprocess
            image, seg, _ = load_and_preprocess(
                row['image_path'], 
                row['seg_path'],
                map_labels=False  # Keep original labels (2, 3)
            )
            
            # Extract individual lesions
            lesions = extract_lesions(image, seg, min_voxels=min_voxels)
            
            # Sort lesions by volume (largest first) and assign IDs
            lesions_with_volume = []
            for lesion_img, lesion_mask, label in lesions:
                volume = np.sum(lesion_mask)
                lesions_with_volume.append((lesion_img, lesion_mask, label, volume))
            
            # Sort by volume descending
            lesions_with_volume.sort(key=lambda x: x[3], reverse=True)
            
            # Extract features for each lesion
            for lesion_id, (lesion_img, lesion_mask, label, volume) in enumerate(lesions_with_volume, start=1):
                features = extract_features(lesion_img, lesion_mask, feature_list)
                
                # Create row dict
                row_dict = {
                    'case': case,
                    'lesion_id': lesion_id,
                    'label': label,
                    'volume_voxels': volume
                }
                
                # Add all feature values
                for fname in feature_names:
                    row_dict[fname] = features[fname]
                
                all_rows.append(row_dict)
                
        except Exception as e:
            print(f"\nWarning: Failed to process case {case}: {e}")
            continue
    
    if len(all_rows) == 0:
        print("Error: No valid lesions found in dataset!")
        sys.exit(1)
    
    # Create DataFrame
    feature_df = pd.DataFrame(all_rows)
    
    # Reorder columns for readability: metadata first, then features
    metadata_cols = ['case', 'lesion_id', 'label', 'volume_voxels']
    feature_cols = feature_names
    column_order = metadata_cols + feature_cols
    feature_df = feature_df[column_order]
    
    # Save to CSV
    feature_df.to_csv(output_csv, index=False)
    
    print(f"\n{'='*60}")
    print(f"Feature extraction complete!")
    print(f"{'='*60}")
    print(f"Total lesions processed: {len(feature_df)}")
    print(f"  Tumors (label=2): {np.sum(feature_df['label'] == 2)}")
    print(f"  Cysts (label=3): {np.sum(feature_df['label'] == 3)}")
    print(f"\nOutput saved to: {output_csv}")
    
    # Print size comparison
    csv_size_mb = Path(output_csv).stat().st_size / (1024 * 1024)
    print(f"CSV size: {csv_size_mb:.2f} MB")
    print(f"Bytes per lesion: {Path(output_csv).stat().st_size / len(feature_df):.0f}")
    
    return feature_df


def main():
    """Main entry point for feature extraction script."""
    parser = create_extraction_parser()
    args = parser.parse_args()
    
    # Create output directory if needed
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Determine feature list
    if args.features:
        feature_list = [Feature[f.upper()] for f in args.features]
    else:
        feature_list = None
    
    # Extract features
    extract_and_cache_features(
        data_csv=args.data,
        output_csv=args.output,
        min_voxels=args.min_voxels,
        feature_list=feature_list
    )


if __name__ == "__main__":
    main()
