"""
Dataset-level processing logic.
Handles batch extraction, augmentation loops, and data storage.
"""

from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
from tqdm import tqdm

from .base import BaseFeatureExtractor


class FeatureDatasetProcessor:
    """
    Orchestrates feature extraction over an entire dataset.
    """

    def __init__(self, extractor: BaseFeatureExtractor):
        self.extractor = extractor

    def process_dataset(
        self,
        input_df: pd.DataFrame,
        output_path: Union[str, Path],
        image_col: str = "image_path",
        seg_col: str = "seg_path",
        augment_count: int = 0,
    ) -> None:
        """
        Run extraction on all rows in input_df.
        Saves the result to a Parquet file.
        """
        results: List[Dict[str, Any]] = []

        print(f"Starting extraction with {type(self.extractor).__name__}...")

        for idx, row in tqdm(input_df.iterrows(), total=len(input_df)):
            image_path = row.get(image_col)
            seg_path = row.get(seg_col)

            if not image_path or not seg_path:
                continue

            # Base metadata from the CSV row (case_id, etc.)
            row_meta = row.to_dict()

            # Helper to run extraction and append to results
            def _run_and_collect(is_aug: bool, aug_id: int):
                try:
                    # extract() returns a LIST of dicts (one per lesion)
                    lesion_features_list = self.extractor.extract(
                        image_path, seg_path, augment=is_aug
                    )

                    for lesion_feats in lesion_features_list:
                        # Merge row metadata + lesion features
                        entry = row_meta.copy()
                        entry.update(lesion_feats)
                        entry["augmented"] = is_aug
                        entry["aug_id"] = aug_id
                        results.append(entry)

                except Exception as e:
                    print(f"Error processing {image_path} (aug={is_aug}): {e}")

            # 1. Original
            _run_and_collect(is_aug=False, aug_id=0)

            # 2. Augmentations
            for i in range(augment_count):
                _run_and_collect(is_aug=True, aug_id=i + 1)

        # 3. Save
        self._save_results(results, output_path)

    @staticmethod
    def load_features(path: Union[str, Path]) -> pd.DataFrame:
        """
        Load features from a file (Parquet preferred).
        Abstraction for training scripts.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Feature file not found: {path}")

        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        elif path.suffix == ".csv":
            return pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def _save_results(self, results: List[Dict[str, Any]], output_path: Union[str, Path]) -> None:
        if not results:
            print("No features extracted. Nothing to save.")
            return

        df = pd.DataFrame(results)

        # Enforce Parquet extension
        path_obj = Path(output_path)
        if path_obj.suffix != ".parquet":
            path_obj = path_obj.with_suffix(".parquet")
            print(f"Note: Enforcing .parquet extension. Output will be: {path_obj}")

        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Save
        df.to_parquet(path_obj, index=False)
        print(f"Saved {len(df)} lesions to {path_obj}")
