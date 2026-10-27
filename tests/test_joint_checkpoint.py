"""Exercise real CPU checkpoint snapshots without constructing datasets or GPU models."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'main/src'))
import torch
from common.trainer import Trainer


def metrics(r10, r20):
    return {'recall@10': r10, 'recall@20': r20}


class JointCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.t = Trainer.__new__(Trainer)
        self.t.config = {'target_recall10': .1, 'target_recall20': .2}
        self.t.run_dir = self.temp.name
        self.t.model = torch.nn.Linear(1, 1, bias=False)
        self.t.best_valid = self.t.best_test_exploratory = None
        self.t.best_joint_exploratory = self.t.target_match_exploratory = None
        self.t.first_target_match = None
        self.t._best_valid_state = None

    def payload(self, name='best_joint_exploratory.pth'):
        return torch.load(Path(self.temp.name) / name, map_location='cpu', weights_only=True)

    def test_r20_max_does_not_overwrite_better_pair(self):
        self.t.model.weight.data.fill_(1)
        self.t._record_evaluation(0, metrics(.08, .15), metrics(.099, .198), True, True)
        self.t.model.weight.data.fill_(2)
        self.t._record_evaluation(1, metrics(.08, .16), metrics(.09, .205), False, True)
        joint = self.payload()
        self.assertEqual(joint['epoch_number'], 1)
        self.assertEqual(joint['split'], 'test')
        self.assertEqual(joint['test'], metrics(.099, .198))
        self.assertEqual(joint['state_dict']['weight'].item(), 1)
        self.assertEqual(self.payload('best_test_exploratory.pth')['epoch_number'], 2)
        self.assertIsNone(self.t.target_match_exploratory)

    def test_no_cross_epoch_or_cross_split_target(self):
        self.t._record_evaluation(0, metrics(.11, .18), metrics(.09, .22), True, True)
        self.t._record_evaluation(1, metrics(.09, .22), metrics(.11, .18), False, True)
        self.assertIsNone(self.t.target_match_exploratory)
        self.assertFalse((Path(self.temp.name) / 'target_match_exploratory.pth').exists())
        self.t._record_evaluation(2, metrics(.101, .202), metrics(.095, .19), True, True)
        target = self.payload('target_match_exploratory.pth')
        self.assertEqual(target['epoch_number'], 3)
        self.assertEqual(target['split'], 'valid')
        self.assertEqual(self.payload()['valid'], target['valid'])
        self.assertGreaterEqual(target['minimum_target_ratio'], 1)
        self.assertEqual(self.t.first_target_match['epoch_number'], 3)

    def test_saved_false_has_no_checkpoint_side_effect(self):
        self.t._record_evaluation(0, metrics(.095, .19), metrics(.09, .18), True, False)
        self.assertEqual(self.t.best_joint_exploratory['split'], 'valid')
        self.assertFalse(list(Path(self.temp.name).iterdir()))


if __name__ == '__main__':
    unittest.main()
