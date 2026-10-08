"""Validate feature headers before allocating embeddings or quadratic KNN graphs."""
from pathlib import Path
import numpy as np


def validate_feature_file(path, expected_items):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing required feature matrix: {path}')
    matrix = np.load(path, mmap_mode='r', allow_pickle=False)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError(f'{path.name}: expected a nonempty 2D feature matrix, got {matrix.shape}')
    if matrix.shape[0] != expected_items:
        raise ValueError(
            f'{path.name}: {matrix.shape[0]} feature rows but {expected_items} interaction items. '
            'Provide features aligned to itemID; expanded-catalog features require a verified '
            'mapping and Stage-1 preprocessing. Do not truncate rows without verifying alignment.'
        )
    if not np.issubdtype(matrix.dtype, np.number) or np.issubdtype(matrix.dtype, np.complexfloating):
        raise ValueError(f'{path.name}: expected real numeric features, got {matrix.dtype}')
    return matrix.shape
