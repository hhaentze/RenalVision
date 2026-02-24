"""Hyperparameter selection based on cross-fold validation"""

import argparse
import json
import os
import tempfile
from os.path import join
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm

from renal_vision.modeling.eval import run_evaluation
from renal_vision.modeling.train import run_training
from renal_vision.shared.metrics import ModelEvaluator
from renal_vision.shared.utils import describe_data, generate_patient_fold_mapping

# --- 1. Configuration & Search Space ---
N_FOLDS = 10  # Reporting performance

xgb_params = {
    "max_depth": [4, 6],
    "colsample_bytree": [0.3, 0.7, 1],
    "min_child_weight": [1, 6, 12],
    "n_estimators": [100, 200],
    "learning_rate": [0.1, 0.3],
}

# mlp_params: D = {}
# configs = [("mlp", list(ParameterGrid(mlp_params))), ("xgboost", list(ParameterGrid(xgb_params)))]

configs = [("xgboost", list(ParameterGrid(xgb_params)))]
tempfile.tempdir = "/tmp"


def main(
    feature_path: str,
    extractor_config_path: str,
    class_config_path: str,
    output_csv: str,
    group_col: str = "case",
    stratify_col: str = "class_id",
    aug_number: int = -1,
    nparafolds: int = -1,
    parafold: int = 0,
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

    config_results: Dict[str, float] = {}
    param_lookup: Dict[str, Dict[str, Any]] = {}

    # Check for parallelism
    global configs
    if nparafolds >= 0:
        new_configs = []
        for c in configs:
            params = c[1]
            fold_length = len(params) // nparafolds
            start = fold_length * parafold
            end = fold_length * (parafold + 1)
            if parafold == (nparafolds - 1):
                end = len(params)
            params = params[start:end]
            new_configs.append((c[0], params))
        configs = new_configs
        output_csv = output_csv.replace(".csv", f"_fold_{parafold}.csv")

    n_loops = sum([N_FOLDS * len(c[1]) for c in configs])

    with tempfile.TemporaryDirectory() as tmpdirname:
        # prepare data
        for f in range(N_FOLDS):
            path = Path(tmpdirname) / f"fold_{f}"
            os.makedirs(path)
            it_path = path / "tmp_it.parquet"
            iv_path = path / "tmp_iv.parquet"
            features[features["fold"] != f].to_parquet(it_path, index=False)
            features[(features["fold"] == f) & (~features["augmented"])].to_parquet(
                iv_path, index=False
            )

        # run cross_val
        with tqdm(total=n_loops, desc="Evaluating hyperparameters") as pbar:
            for model_name, param_grid in configs:
                for _, params in enumerate(param_grid):
                    param_key = json.dumps(params | {"model": model_name}, sort_keys=True)
                    if param_key not in param_lookup:
                        param_lookup[param_key] = params

                    results = []
                    for f in range(N_FOLDS):
                        path = Path(tmpdirname) / f"fold_{f}"
                        it_path = path / "tmp_it.parquet"
                        iv_path = path / "tmp_iv.parquet"

                        # Train and Eval
                        run_training(
                            it_path,
                            tmpdirname,
                            model_type=model_name,
                            class_config_path=class_config_path,  # "names_binary.json",
                            extractor_config_path=extractor_config_path,  # "features.config.json",
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
                            return_preds=True,
                        )
                        results.append(m)
                        pbar.update(1)

                    # calculate avg AUC across folds
                    y_true_list = []
                    y_proba_list = []
                    for fold_res in results:
                        df = fold_res["pred_df"]
                        y_true_list.append(df["y_true"].values)
                        y_proba_list.append(df["y_proba"].values)

                    evaluator = ModelEvaluator(y_true_list, y_proba_list)
                    auc = evaluator.get_auc(with_ci=False).loc["MACRO AVERAGE"]["Mean"]
                    config_results[param_key] = auc

        # --- 3. Analysis and Ranking ---
        ranking_data = []

        for param_key, auc in config_results.items():
            ranking_data.append({"auc": auc, "params_json": param_key})

    # Convert to DataFrame for easy viewing
    report_df = pd.DataFrame(ranking_data).sort_values(by="auc", ascending=False)

    # --- 4. Print the Top 3 Configurations ---
    print("\n--- TOP 3 CONFIGURATIONS BY MEAN AUC ---")
    for idx, row in report_df.head(3).iterrows():
        print(f"\nRank {idx + 1} | Mean AUC: {row['auc']:.3f}")
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

    # Arguments for Parallelisation with Slurm
    parser.add_argument(
        "--nfolds", type=int, default=-1, help="Parallelise processing across n folds"
    )
    parser.add_argument("--fold", type=int, default=0, help="Parameter fold to use")

    args = parser.parse_args()

    main(
        feature_path=args.feature_path,
        extractor_config_path=args.extractor_config_path,
        class_config_path=args.class_config_path,
        output_csv=args.output_csv,
        group_col=args.group_col,
        stratify_col=args.stratify_col,
        aug_number=args.aug_number,
        nparafolds=args.nfolds,
        parafold=args.fold,
    )
