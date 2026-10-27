"""Regression checks for graph construction and experiment configuration hooks."""
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import scipy.sparse as sp
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'main' / 'src'))
from main import load_yaml_overrides
from models.mage import MAGE


def original_normalization(n_users, n_items, interactions, item_adj):
    adj = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32).tolil()
    R = interactions.tolil()
    adj[:n_users, n_users:] = R
    adj[n_users:, :n_users] = R.T
    adj[n_users:, n_users:] = item_adj.tolil()
    adj = adj.todok()
    rowsum = np.array(adj.sum(1))
    rowsum[rowsum <= 1e-9] = 1e-9
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    D = sp.diags(d_inv)
    return D.dot(adj).dot(D).tocsr()


class GraphEquivalence(unittest.TestCase):
    def test_duplicate_directed_edges_and_isolated_nodes(self):
        interactions = sp.coo_matrix(
            (np.array([1., 1., 1., 1.], dtype=np.float32), ([0, 0, 1, 1], [0, 0, 1, 2])),
            shape=(4, 5),
        )
        visual = sp.coo_matrix(
            (np.array([1., 1., 1.], dtype=np.float32), ([0, 0, 2], [1, 1, 1])), shape=(5, 5),
        )
        text = visual + sp.coo_matrix(
            (np.array([0.7, 0.7], dtype=np.float32), ([1, 3], [2, 0])), shape=(5, 5),
        )
        model = SimpleNamespace(n_users=4, n_items=5, interaction_matrix=interactions)
        previous_R = None
        for item_adj in (visual, text):
            actual = MAGE.get_adj_mat(model, item_adj)
            expected = original_normalization(4, 5, interactions, item_adj)
            self.assertTrue(sp.isspmatrix_csr(actual))
            self.assertEqual(actual.dtype, np.float32)
            np.testing.assert_allclose(actual.toarray(), expected.toarray(), rtol=2e-6, atol=1e-7)
            np.testing.assert_allclose(model.R.toarray(), expected[:4, 4:].toarray(), rtol=2e-6, atol=1e-7)
            self.assertTrue(np.isfinite(actual.data).all())
            if previous_R is not None:
                self.assertFalse(np.array_equal(previous_R, model.R.toarray()))
            previous_R = model.R.toarray().copy()

    def test_max_p2b_k_truncates_and_zero_disables_injection(self):
        for k in (0, 1, 15, 20, 40):
            model = SimpleNamespace(n_items=32, max_p2b_k=k, p2b_weight=0.7)
            actual = MAGE.add_edge(model, {0: [31]}, {'0': list(range(1, 31))}).tocsr()
            self.assertEqual(actual[0, 31], 1.0)
            self.assertEqual(actual.getnnz(), min(k, 30) + 1)
            if k:
                self.assertAlmostEqual(float(actual[0, 1]), 0.7, places=6)

    def test_invalid_ids_and_duplicate_p2b_weights(self):
        model = SimpleNamespace(n_items=4, max_p2b_k=20, p2b_weight=0.4)
        actual = MAGE.add_edge(model, {0: [1]}, {'0': ['1', '-1', '0', '4', 'bad'], '-1': ['1'], '4': ['0']})
        self.assertEqual(actual.tocsr().getnnz(), 1)
        self.assertAlmostEqual(float(actual.tocsr()[0, 1]), 1.4, places=6)

    def test_safe_yaml_supports_scientific_notation(self):
        config = load_yaml_overrides(io.StringIO('reg_weight_1: 1e-05\nrate: -2E+1\nvalue: 0.001\n'))
        self.assertEqual(config, {'reg_weight_1': 0.00001, 'rate': -20.0, 'value': 0.001})
        self.assertIsInstance(config['reg_weight_1'], float)
        with self.assertRaises(yaml.constructor.ConstructorError):
            load_yaml_overrides(io.StringIO('!!python/object/apply:os.system [echo forbidden]'))

    def test_stage1_cli_applies_gap_ratio(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            np.save(root / 'features.npy', np.eye(4, dtype=np.float32))
            (root / 'related.json').write_text(json.dumps({'0': [1, 2]}), encoding='utf-8')
            command = [sys.executable, str(ROOT / 'stage1_process.py'),
                       '--emb-path', str(root / 'features.npy'), '--related-path', str(root / 'related.json'),
                       '--out-path', str(root / 'enhanced.npy'), '--gap-keep-ratio', '0.8']
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('GAP_KEEP_RATIO=0.8', result.stderr)
            self.assertEqual(np.load(root / 'enhanced.npy').shape, (4, 4))
            result = subprocess.run(command[:-1] + ['1.1'], capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('between 0 and 1', result.stderr)


if __name__ == '__main__':
    unittest.main()
