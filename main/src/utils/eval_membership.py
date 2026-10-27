"""Order-preserving replacement for the evaluator's membership loop only.

This module does not cache across calls, move metrics to another device, change
positive counts, change ranking, or alter metric reduction order.
"""
import numpy as np


def legacy_hit_matrix(pos_items, topk_index):
    """Exact original loop, retained as an independent comparison oracle."""
    bool_rec_matrix = []
    for m, n in zip(pos_items, topk_index):
        bool_rec_matrix.append([True if i in m else False for i in n])
    return np.asarray(bool_rec_matrix)


def set_hit_matrix(pos_items, topk_index):
    """Use integer sets where membership has identical equality semantics.

    Rows are interpreted in the supplied order every call. For integer arrays
    of identical dtype, equality is exact, so replacing repeated scans with a
    set preserves every boolean even with duplicates, empty rows and large IDs.
    Heterogeneous/noninteger dtypes retain the literal original NumPy membership
    expression; this also preserves dtype-specific comparisons and NaN behavior.
    """
    rows = []
    for positives, recommendations in zip(pos_items, topk_index):
        if (isinstance(positives, np.ndarray) and positives.ndim == 1
                and isinstance(recommendations, np.ndarray)
                and positives.dtype == recommendations.dtype
                and np.issubdtype(positives.dtype, np.integer)):
            positive_set = set(positives.tolist())
            rows.append([item in positive_set for item in recommendations.tolist()])
        else:
            rows.append([True if item in positives else False for item in recommendations])
    return np.asarray(rows)
