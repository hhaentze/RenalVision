"""
General utility functions (splitting, logging, etc.).
"""

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def generate_stratified_group_split(
    df: pd.DataFrame,
    group_col: str,
    stratify_col: Optional[str] = None,
    test_size: float = 0.2,
    min_test_samples_per_class: int = 5,
    random_state: int = 42,
    n_attempts: int = 100,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into Train and Test sets while ensuring:
    1. Group Integrity: All rows with the same group_col value stay together.
    2. Stratification: Approximates the class distribution of stratify_col.
    3. Min Samples: Ensures test set has at least N samples of each class.

    Args:
        df: Input DataFrame.
        group_col: Column name for grouping (e.g., 'case_id').
        stratify_col: Column name for stratification (e.g., 'class_id').
        n_attempts: Number of random splits to try to find the best balance.

    Returns:
        (train_df, test_df)
    """
    groups = df[group_col].unique()
    n_groups = len(groups)

    # If no stratification needed, do simple group shuffle
    if not stratify_col:
        np.random.seed(random_state)
        shuffled_groups = np.random.permutation(groups)
        n_test = int(n_groups * test_size)
        test_groups = set(shuffled_groups[:n_test])

        is_test = df[group_col].isin(test_groups)
        return df[~is_test], df[is_test]

    # --- Robust Stratified Group Split ---
    # We try 'n_attempts' random group splits and pick the one with
    # the lowest distribution error (KL divergence proxy).

    best_split = None
    min_error = float("inf")

    # Target distribution
    target_dist = df[stratify_col].value_counts(normalize=True)
    rng = np.random.default_rng(random_state)

    for _ in range(n_attempts):
        # 1. Randomly split groups
        shuffled = rng.permutation(groups)
        n_test_groups = max(1, int(n_groups * test_size))

        candidate_test_groups = set(shuffled[:n_test_groups])

        # 2. Check mask
        is_test = df[group_col].isin(candidate_test_groups)
        test_subset = df[is_test]

        if len(test_subset) == 0:
            continue

        # 3. Check Constraint: Minimum samples per class
        counts = test_subset[stratify_col].value_counts()
        if any(counts.get(cls, 0) < min_test_samples_per_class for cls in target_dist.index):
            # This split is invalid, skip
            continue

        # 4. Check Stratification Error (Sum of squared diffs from target distribution)
        test_dist = test_subset[stratify_col].value_counts(normalize=True)
        # Fill missing classes with 0
        test_dist = test_dist.reindex(target_dist.index, fill_value=0)

        error = np.sum((target_dist - test_dist) ** 2)

        if error < min_error:
            min_error = error
            best_split = (df[~is_test].copy(), df[is_test].copy())

    if best_split is None:
        warnings.warn(
            f"Could not find a split satisfying min_samples={min_test_samples_per_class} "
            "after {n_attempts} attempts. Falling back to simple random group split."
        )
        # Fallback logic could be repeated here or just raise Error
        return generate_stratified_group_split(df, group_col, None, test_size, 0, random_state)

    return best_split
