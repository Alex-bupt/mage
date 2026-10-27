"""CPU regression tests for the only fast-evaluation behavior change."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'main/src'))
from utils.eval_membership import legacy_hit_matrix, set_hit_matrix
from utils.metrics import metrics_dict


class MembershipEquivalence(unittest.TestCase):
    def compare(self, positives, recommendations):
        old = legacy_hit_matrix(positives, recommendations)
        new = set_hit_matrix(positives, recommendations)
        self.assertEqual(old.dtype, new.dtype)
        np.testing.assert_array_equal(old, new)
        lengths = np.asarray([len(row) for row in positives])
        if lengths.size and (lengths > 0).all() and old.shape[1]:
            for name, function in metrics_dict.items():
                self.assertEqual(function(old, lengths).tobytes(), function(new, lengths).tobytes(), name)

    def test_random_integer_rows_and_reordering(self):
        rng = np.random.default_rng(13)
        for dtype in (np.int8, np.uint8, np.int32, np.int64, np.uint64):
            for count in (1, 17, 127):
                positives = [rng.integers(0, 100, size=rng.integers(1, 20), dtype=dtype) for _ in range(count)]
                recommendations = rng.integers(0, 100, size=(count, 50), dtype=dtype)
                self.compare(positives, recommendations)
                order = rng.permutation(count)
                self.compare([positives[i] for i in order], recommendations[order])
                positives.reverse()
                self.compare(positives, recommendations)

    def test_duplicates_empty_and_noncontinuous_users(self):
        user_ids = [100004, 2, 9003]
        truth = {100004: [7, 7, 5], 2: [3], 9003: [8, 8]}
        positives = [np.asarray(truth[user], dtype=np.int64) for user in user_ids]
        self.compare(positives, np.array([[7, 7, 1], [3, 2, 0], [8, 7, 0]], dtype=np.int64))
        self.compare([np.array([], dtype=np.int64)], np.array([[1, 2, 3]], dtype=np.int64))
        self.compare([], np.empty((0, 50), dtype=np.int64))
        self.compare([np.array([1])], np.empty((1, 0), dtype=np.int64))

    def test_large_ids_and_heterogeneous_fallback(self):
        for dtype, values in ((np.int64, [-2**63, 0, 2**53, 2**53+1, 2**63-1]),
                              (np.uint64, [0, 2**53, 2**53+1, 2**63, 2**64-1])):
            row = np.asarray(values, dtype=dtype)
            self.compare([row[:2], row[2:]], np.vstack([row, row[::-1]]))
        self.compare([np.array([2**63], dtype=np.uint64)], np.array([[2**63-1]], dtype=np.int64))
        self.compare([np.array([2**63-1], dtype=np.int64)], np.array([[2**63]], dtype=np.uint64))
        self.compare([[2**80, 2**80+1]], np.array([[2**80, 2**80+2]], dtype=object))
        self.compare([np.array([np.nan, 1., 2.])], np.array([[np.nan, 1., 3.]]))


if __name__ == '__main__':
    unittest.main()
