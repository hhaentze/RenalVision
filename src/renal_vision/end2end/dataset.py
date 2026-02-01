from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from monai.data import CacheDataset, Dataset, PersistentDataset
from monai.transforms import Compose, Randomizable


def split_compose(compose: Compose) -> Tuple[Optional[Compose], Optional[Compose]]:
    """Splits a compose at the first position of a randomizable transform"""
    transforms = compose.transforms
    for idx, transform in enumerate(transforms):
        if isinstance(transform, Randomizable):
            det = Compose(transforms[:idx]) if idx > 0 else None
            tail = Compose(transforms[idx:])
            return det, tail
    return compose, None


def create_smart_dataset(
    data: Sequence[Dict[str, Any]],
    transform: Compose,
    cache_dir: Optional[str | Path] = None,
    cache_rate: int = 0,
    verbose: bool = True,
) -> Dataset:
    # --- Case 1: No Memory Caching Limit ---
    if cache_rate <= 0:
        if cache_dir is None:
            message = "SmartDataset: Using standard Dataset."
            dset = Dataset(data=data, transform=transform)
        else:
            message = f"SmartDataset: Using PersistentDataset at {cache_dir}."
            dset = PersistentDataset(data=data, transform=transform, cache_dir=str(cache_dir))

    # --- Case 2: Memory Caching Specified ---
    elif cache_rate > 0 and not cache_dir:
        message = f"SmartDataset: RAM Cache Mode ({round(cache_rate * 100, 1)}%)."
        dset = CacheDataset(data=data, transform=transform, cache_rate=cache_rate)

    # --- Case 3: Memory & Persistent Caching Specified ---
    else:
        message = f"SmartDataset: Hybrid Mode (Persistent + {round(cache_rate * 100, 1)}% Cache)."
        deterministic_transform, nondeterministic_transform = split_compose(transform)
        persistent_ds = PersistentDataset(
            data=data,
            transform=deterministic_transform,  # type: ignore [arg-type]
            cache_dir=str(cache_dir),
        )
        dset = CacheDataset(
            data=persistent_ds,  # type: ignore [arg-type]
            transform=nondeterministic_transform,  # type: ignore [arg-type]
            cache_rate=cache_rate,
        )

    if verbose:
        print(message)
    return dset
