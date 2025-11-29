"""Data utilities for train/test splitting."""

import numpy as np
from typing import Tuple, List


def generate_train_test_split(
    df, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[List[int], List[int]]:
    """
    Generate random train/test split indices for a DataFrame.
    
    Args:
        df: pandas DataFrame (typically with seg_path, image_path columns)
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
    
    n_samples = len(df)
    n_test = int(n_samples * test_size)
    
    # Generate random permutation
    indices = np.random.permutation(n_samples)
    
    test_indices = indices[:n_test].tolist()
    train_indices = indices[n_test:].tolist()
    
    return train_indices, test_indices
