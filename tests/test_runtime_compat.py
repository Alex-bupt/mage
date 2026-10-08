"""Numerical checks for native graph normalization and modern NumPy metrics."""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'main' / 'src'))
from utils.utils import get_sparse_laplacian, get_dense_laplacian, build_sim
from utils.metrics import ndcg_, map_


class RuntimeCompatibility(unittest.TestCase):
    def test_sparse_matches_dense_with_duplicate_edges_and_isolated_node(self):
        edges = torch.tensor([[0,0,0,1,2],[1,1,2,0,2]])
        weights = torch.tensor([1.,2.,1.,3.,2.])
        dense = torch.sparse_coo_tensor(edges,weights,(4,4)).to_dense()
        for mode in ('none','rw','sym'):
            idx, val = get_sparse_laplacian(edges, weights, 4, mode)
            result = torch.sparse_coo_tensor(idx,val,(4,4)).to_dense()
            torch.testing.assert_close(result,get_dense_laplacian(dense,mode))

    def test_zero_features_stay_finite(self):
        result = build_sim(torch.tensor([[0.,0.],[1.,0.],[0.,1.]]))
        self.assertTrue(torch.isfinite(result).all().item())
        torch.testing.assert_close(result,torch.diag(torch.tensor([0.,1.,1.])))

    def test_metric_values(self):
        hits = np.array([[True,False,True],[False,True,False]])
        lengths = np.array([2,1])
        expected_ndcg = np.array([0.5, (1/(1+1/np.log2(3))+1/np.log2(3))/2,
                                 (1.5/(1+1/np.log2(3))+1/np.log2(3))/2])
        np.testing.assert_allclose(ndcg_(hits,lengths),expected_ndcg)
        np.testing.assert_allclose(map_(hits,lengths),[0.5,0.5,2/3])


if __name__ == '__main__':
    unittest.main()
