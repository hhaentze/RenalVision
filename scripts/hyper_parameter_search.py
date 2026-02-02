"""Hyperparameter selection based on cross-fold validation"""

import argparse
import json
import tempfile
from os.path import join
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm

from renal_vision.modeling.eval import run_evaluation
from renal_vision.modeling.train import run_training
from renal_vision.shared.utils import describe_data, generate_patient_fold_mapping

# --- 1. Configuration & Search Space ---
N_FOLDS = 10  # Reporting performance

xgb_params = {
    "max_depth": [4, 6],
    "learning_rate": [0.01, 0.05, 0.1],
    "gamma": [0, 1, 5],
    "min_child_weight": [4, 8, 12],
    "subsample": [0.7, 0.9],
    "colsample_bytree": [0.7, 0.9],
}
mlp_params = {
    "hidden_layer_sizes": [(100,), (256,), (256, 128)],  # (100,) is standard
    "alpha": [0.0001, 0.001, 0.01],  # L2 regularization term
    "learning_rate_init": [0.001, 0.01],
    "batch_size": [64, 128],
}

configs = [("mlp", list(ParameterGrid(mlp_params))), ("xgboost", list(ParameterGrid(xgb_params)))]
tempfile.tempdir = "/tmp"


def main(
    feature_path: str,
    extractor_config_path: str,
    class_config_path: str,
    output_csv: str,
    group_col: str = "case",
    stratify_col: str = "class_id",
    aug_number: int = -1,
) -> None:
    # load features
    features = pd.read_parquet(feature_path)

    # decide on how many augmentations to include
    if aug_number > -1:
        features = features[features["aug_id"] <= aug_number].reset_index(drop=True)
    describe_data(features, target_column=stratify_col)

    # assign partitions to features
    fold_map = generate_patient_fold_mapping(
        features, group_col=group_col, stratify_col=stratify_col, n_folds=N_FOLDS
    )
    features["fold"] = features[group_col].astype(str).map(fold_map)

    config_results: Dict[str, List[float]] = {}
    param_lookup: Dict[str, Dict[str, Any]] = {}

    n_loops = sum([N_FOLDS * len(c[1]) for c in configs])
    with tqdm(total=n_loops, desc="Evaluating hyperparameters") as pbar:
        for model_name, param_grid in configs:
            for f in range(N_FOLDS):
                with tempfile.TemporaryDirectory() as tmpdirname:
                    # Split
                    it_path = join(tmpdirname, "tmp_it.parquet")
                    iv_path = join(tmpdirname, "tmp_iv.parquet")
                    features[features["fold"] != f].to_parquet(it_path, index=False)
                    features[(features["fold"] == f) & (~features["augmented"])].to_parquet(
                        iv_path, index=False
                    )

                    for _, params in enumerate(param_grid):
                        param_key = json.dumps(params | {"model": model_name}, sort_keys=True)
                        if param_key not in param_lookup:
                            param_lookup[param_key] = params
                            config_results[param_key] = []

                        # Train and Eval
                        run_training(
                            it_path,
                            tmpdirname,
                            model_type=model_name,
                            class_config_path=class_config_path,  # "names_binary.json",
                            extractor_config_path=extractor_config_path,  # "/sc-scratch/sc-scratch-cc06-ag-ki-radiologie/kidney/embeddings/new/kits_radiomics.config.json",
                            verbose=False,
                            class_column=stratify_col,
                            **params,
                        )
                        m = run_evaluation(
                            iv_path,
                            join(tmpdirname, "model.pkl"),
                            tmpdirname,
                            class_column=stratify_col,
                            verbose=False,
                        )
                        config_results[param_key].append(m["f1"])
                        pbar.update(1)

        # --- 3. Analysis and Ranking ---
        ranking_data = []

        for param_key, scores in config_results.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)

            ranking_data.append(
                {
                    "params_json": param_key,
                    "mean_f1": mean_score,
                    "std_f1": std_score,
                    "min_f1": np.min(scores),  # Check for "worst-case" scenario
                    "cv_scores": scores,
                }
            )

    # Convert to DataFrame for easy viewing
    report_df = pd.DataFrame(ranking_data).sort_values(by="mean_f1", ascending=False)

    # --- 4. Print the Top 3 Configurations ---
    print("\n--- TOP 3 CONFIGURATIONS BY MEAN F1 ---")
    for idx, row in report_df.head(3).iterrows():
        print(f"\nRank {idx + 1} | Mean F1: {row['mean_f1']:.4f} (±{row['std_f1']:.4f})")
        print(f"Parameters: {row['params_json']}")

    # --- 5. Save the configurations ---
    report_df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hyperparameter search")
    parser.add_argument("feature_path", help="Path to feature file")
    parser.add_argument("extractor_config_path", help="Path to extractor config")
    parser.add_argument("class_config_path", help="Path to class config")
    parser.add_argument("output_csv", help="Output CSV path")
    parser.add_argument("--group_col", default="case", help="Group column name (default: case)")
    parser.add_argument(
        "--stratify_col", default="class_id", help="Stratify column name (default: class_id)"
    )
    parser.add_argument(
        "--aug_number",
        type=int,
        default=-1,
        help="How many augmentations should be included? (default: all)",
    )

    args = parser.parse_args()

    main(
        feature_path=args.feature_path,
        extractor_config_path=args.extractor_config_path,
        class_config_path=args.class_config_path,
        output_csv=args.output_csv,
        group_col=args.group_col,
        stratify_col=args.stratify_col,
        aug_number=args.aug_number,
    )
