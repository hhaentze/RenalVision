"""Data utilities for train/test splitting."""

from typing import List, Tuple

import numpy as np


def generate_train_test_split(
    df, test_size: float = 0.2, random_state: int = 42
) -> Tuple[List[int], List[int]]:
    """
    Generate random train/test split indices for a DataFrame.
    If DataFrame contains a 'case' column, performs case/patient-stratified splitting

    Args:
        df: pandas DataFrame (typically with seg_path, image_path columns)
            If 'case' column exists, will split by case/patient ID
        test_size: Fraction of data to use for testing (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        train_indices: List of indices for training set
        test_indices: List of indices for testing set

    Example:
        >>> df = pd.read_csv("data.csv")
        >>> train_idx, test_idx = generate_train_test_split(df, test_size=0.2)
        >>> train_df = df.iloc[train_idx]
        >>> test_df = df.iloc[test_idx]
    """
    np.random.seed(random_state)

    # Check if 'case' column exists for patient-stratified splitting
    if "case" in df.columns:
        # Get unique case IDs
        unique_cases = df["case"].unique()
        n_cases = len(unique_cases)
        n_test_cases = int(n_cases * test_size)

        # Randomly shuffle cases
        shuffled_cases = np.random.permutation(unique_cases)

        # Split cases into train/test
        test_cases = shuffled_cases[:n_test_cases]
        train_cases = shuffled_cases[n_test_cases:]

        # Get indices for each split
        test_indices = df[df["case"].isin(test_cases)].index.tolist()
        train_indices = df[df["case"].isin(train_cases)].index.tolist()
        print("Found 'case' column in dataframe. Performed patient-stratified splitting.")

    else:
        # Simple random splitting (original behavior)
        n_samples = len(df)
        n_test = int(n_samples * test_size)

        # Generate random permutation
        indices = np.random.permutation(n_samples)

        test_indices = indices[:n_test].tolist()
        train_indices = indices[n_test:].tolist()
        print("No 'case' column in dataframe found. Performed random splitting.")

    return train_indices, test_indices
