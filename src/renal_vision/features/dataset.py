"""
Dataset-level processing logic.
Handles batch extraction, augmentation loops, and data storage.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from .base_extractor import BaseFeatureExtractor


def worker_init(cores_per_worker: int):
    """
    Precision resource control for the worker process.
    """
    # 1. Set environment variables for BLAS/OpenMP libraries
    os.environ["OMP_NUM_THREADS"] = str(cores_per_worker)
    os.environ["MKL_NUM_THREADS"] = str(cores_per_worker)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cores_per_worker)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cores_per_worker)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cores_per_worker)

    # 2. Set Torch-specific thread limits
    torch.set_num_threads(cores_per_worker)


def _process_single_row(args):
    """
    Standalone helper function to process one row.
    Needs to be outside the class or a static method to be picklable.
    """
    extractor, row, image_col, seg_col, augment_count = args
    image_path = row.get(image_col)
    seg_path = row.get(seg_col)

    if not image_path or not seg_path:
        return []

    row_meta = row.to_dict()
    local_results = []

    try:
        if augment_count > 0:
            lesion_features_list = extractor.extract_multiple_augmentations(
                image_path, seg_path, augment_count
            )
        else:
            lesion_features_list = extractor.extract(image_path, seg_path)

        for lesion_feats in lesion_features_list:
            entry = row_meta.copy()
            entry.update(lesion_feats)
            local_results.append(entry)

    except Exception as e:
        print(f"Error processing {image_path}: {e}", flush=True)

    return local_results


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
        num_jobs: int = 4,
        cores_per_job: int = 1,
        batch_size: int = 200,
    ) -> None:
        # Setup Paths & Clean Up
        final_path = Path(output_path).with_suffix(".parquet")
        tmp_path = final_path.with_suffix(".parquet.tmp")
        config_path = final_path.with_name(final_path.stem + ".config.json")
        for p in [final_path, tmp_path, config_path]:
            if p.exists():
                p.unlink()
        final_path.parent.mkdir(parents=True, exist_ok=True)

        # Design schema for parquet writer
        # active features: float
        # all other: string
        schema_blueprint = {f: pa.float64() for f in self.extractor.feature_names}
        schema_blueprint.update(
            {
                "lesion_id": pa.int16(),
                "class_id": pa.int16(),
                "volumne_voxels": pa.float16(),
                "augmented": pa.bool_(),
                "aug_id": pa.int16(),
            }
        )
        current_batch: List[Dict[str, Any]] = []
        writer: Optional[pq.ParquetWriter] = None

        print(f"Starting extraction with {num_jobs * cores_per_job} workers...")

        # Prepare arguments for the pool
        # Note: self.extractor must be picklable.
        tasks = [
            (self.extractor, row, image_col, seg_col, augment_count)
            for _, row in input_df.iterrows()
        ]

        def handle_results(res_list: List[Dict]) -> None:
            nonlocal writer, current_batch

            if res_list:
                current_batch.extend(res_list)

            if len(current_batch) > batch_size:
                writer = self._write_batch_pyarrow(
                    current_batch, tmp_path, writer, schema_blueprint
                )
                current_batch.clear()

        if num_jobs > 1:
            # --- Multiprocessing Path ---
            with ProcessPoolExecutor(
                max_workers=num_jobs, initializer=worker_init, initargs=(cores_per_job,)
            ) as executor:
                futures = [executor.submit(_process_single_row, task) for task in tasks]
                for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing"):
                    handle_results(future.result())

        else:
            # --- Single Process / Debug Path ---
            worker_init(cores_per_job)
            for task in tqdm(tasks, desc="Processing (Single)"):
                single_res: List[Any] = _process_single_row(task)
                handle_results(single_res)

        # Final flush
        if current_batch:
            writer = self._write_batch_pyarrow(current_batch, tmp_path, writer, schema_blueprint)

        # Finalize the file
        if writer is not None:
            writer.close()
            tmp_path.rename(final_path)
            with open(config_path, "w") as f:
                json.dump(self.extractor.get_config(), f, indent=4)
            print(f"Extraction complete. Saved to: {final_path}")
        else:
            print("No data was processed.")

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

    def _write_batch_pyarrow(
        self,
        data_list: List[Dict[str, Any]],
        path: Path,
        writer: Optional[pq.ParquetWriter],
        schema_blueprint: Optional[Dict[str, Any]],
    ) -> pq.ParquetWriter:
        """
        Converts a list of dicts to a PyArrow Table and writes/appends to Parquet.
        """
        if writer is None:
            if schema_blueprint is None:
                raise ValueError("Schema must be defined if writer is None")  #

            new_blueprint = []
            for k in data_list[0].keys():
                if k in schema_blueprint:
                    new_blueprint.append((k, schema_blueprint[k]))
                else:
                    new_blueprint.append((k, pa.string()))
            schema = pa.schema(new_blueprint)
            writer = pq.ParquetWriter(str(path), schema, compression="snappy")

        df = pd.DataFrame(data_list)
        df = df[writer.schema.names]

        # Enforce types defined in pyarrow schema
        for field in writer.schema:
            col = field.name
            a_type = field.type

            if pa.types.is_floating(a_type):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(a_type.to_pandas_dtype())
            elif pa.types.is_integer(a_type):
                df[col] = (
                    pd.to_numeric(df[col], errors="coerce").round().astype(a_type.to_pandas_dtype())
                )
            elif pa.types.is_boolean(a_type):
                df[col] = df[col].map({True: True, False: False, "True": True, "False": False})
            elif pa.types.is_string(a_type) or pa.types.is_binary(a_type):
                df[col] = df[col].astype(str).replace(["None", "nan", "<NA>"], None)

        table = pa.Table.from_pandas(df, schema=writer.schema, preserve_index=False)
        writer.write_table(table)

        return writer
