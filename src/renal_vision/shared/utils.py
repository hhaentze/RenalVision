import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def generate_patient_fold_mapping(
    df: pd.DataFrame,
    group_col: str,
    stratify_col: str,
    n_folds: int = 5,
    min_test_samples_per_class: int = 2,
    random_state: int = 42,
) -> Dict[str, int]:
    """
    Assigns patients to folds using a priority-based greedy approach.
    Handles patients with rows in multiple classes.
    """
    # 1. Prepare Metadata (Patient-level class counts)
    if "augmented" in df.columns:
        clean_df = df.loc[~df["augmented"], [group_col, stratify_col]]
    else:
        clean_df = df[[group_col, stratify_col]]

    # Rows: patients, Cols: counts of each class
    patient_meta = clean_df.groupby([group_col, stratify_col]).size().unstack(fill_value=0)

    class_names = list(patient_meta.columns)
    # Priority: Which classes are rarest? (We assign these first)
    class_priority = patient_meta.sum(axis=0).sort_values().index.tolist()

    # Convert dataframe rows into a list of dictionaries for iteration
    # Using .to_dict('index') directly is cleaner
    patient_data = patient_meta.to_dict(orient="index")
    patients = list(patient_data.keys())

    rng = np.random.default_rng(random_state)
    rng.shuffle(patients)

    # 2. Greedy Assignment Initialization
    # folds_counts tracks total samples of each class currently in each fold
    folds_counts = np.zeros((n_folds, len(class_names)))
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    mapping: Dict[str, int] = {}

    for p_id in patients:
        counts_dict = patient_data[p_id]
        p_counts_arr = np.array([counts_dict.get(c, 0) for c in class_names])

        # Determine the rarest class this patient actually has
        patient_classes = [c for c in class_priority if counts_dict.get(c, 0) > 0]

        if not patient_classes:
            # Patient has no clean samples (only augmented?) -> assign randomly
            best_fold = rng.integers(0, n_folds)
        else:
            # Look at the rarest class this patient possesses
            r_cls = patient_classes[0]
            r_idx = class_to_idx[r_cls]

            # Find which folds have the fewest samples of that specific rare class
            # We add a tiny bit of random noise (1e-6) to break ties randomly
            current_shares = folds_counts[:, r_idx] + rng.uniform(0, 1e-6, size=n_folds)
            best_fold = int(np.argmin(current_shares))

        # Update mapping and the global fold count tracker
        mapping[str(p_id)] = best_fold
        folds_counts[best_fold] += p_counts_arr

    # 3. Validation & Reporting
    _validate_distribution(folds_counts, class_names, min_test_samples_per_class)

    return mapping


def _validate_distribution(counts: np.ndarray, class_names: List[Any], min_samples: int) -> None:
    """Internal helper to warn if constraints aren't met."""
    for f_idx, row in enumerate(counts):
        for c_idx, class_name in enumerate(class_names):
            if row[c_idx] < min_samples:
                warnings.warn(
                    f"Fold {f_idx} is under-represented: "
                    f"Class '{class_name}' has only {int(row[c_idx])} samples."
                )


def describe_data(features, target_column="class_id"):
    n_cases = features["case"].nunique()
    n_lesions = len(features[~features["augmented"]])
    print(f"Found entries for {n_lesions} lesions from {n_cases} cases.")

    classes = features[target_column].unique()
    classes.sort()
    print(f"Found {len(classes)} classes: {classes}")

    for cl in classes:
        n_cl_lesions = len(features[(features[target_column] == cl) & (~features["augmented"])])
        print(f"  Class {cl}: {n_cl_lesions} lesions")

    if "source" in features.columns:
        sources = features["source"].unique()
        print(f"Data sources: {[str(src) for src in sources]}")
        for src in sources:
            n_src_lesions = len(features[(features["source"] == src) & (~features["augmented"])])
            print(f"  {src}: {n_src_lesions} lesions")

    oversampling_factor = (len(features) - n_lesions) / n_lesions
    print(f"Each lesion was augmented {oversampling_factor:.1f} times on average.")
