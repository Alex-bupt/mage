"""Check dataset presets and precedence without loading data or training."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'main' / 'src'))
from utils.configurator import Config

entry_spec = importlib.util.spec_from_file_location('mage_entry', ROOT / 'main/src/main.py')
entry = importlib.util.module_from_spec(entry_spec)
entry_spec.loader.exec_module(entry)


class VerifiedConfigurationTests(unittest.TestCase):
    EXPECTED = {
        'baby': dict(p2b_weight=.9, max_p2b_k=10, gap_keep_ratio=.2, seed=[42],
                     bm_loss=.01, bm_temp=.2, um_loss=.003, um_temp=.1, vt_loss=.01,
                     learning_rate=.001, target_recall10=.0702, target_recall20=.1077),
        'sports': dict(p2b_weight=.4, max_p2b_k=20, gap_keep_ratio=.4, seed=[999],
                       bm_loss=.005, bm_temp=.2, um_loss=.01, um_temp=.15, vt_loss=.01,
                       learning_rate=.0007, target_recall10=.0821, target_recall20=.1221),
        'clothing': dict(p2b_weight=.2, max_p2b_k=15, gap_keep_ratio=.4, seed=[13],
                         bm_loss=.01, bm_temp=.2, um_loss=.01, um_temp=.2, vt_loss=.1,
                         learning_rate=.001, target_recall10=.0738, target_recall20=.1088),
    }

    def config(self, dataset, overrides=None):
        return Config('mage', dataset, dict(use_gpu=False, **(overrides or {})))

    def test_dataset_argument_resolves_selected_hyperparameters(self):
        for dataset, expected in self.EXPECTED.items():
            with self.subTest(dataset=dataset):
                cfg = self.config(dataset)
                for key, value in expected.items():
                    self.assertEqual(cfg[key], value, key)
                self.assertIs(type(cfg['max_p2b_k']), int)
                self.assertIs(type(cfg['reg_weight_1']), float)
                self.assertEqual(cfg['reg_weight_1'], 1e-5)
                self.assertNotIn('dataset_overrides', cfg)

    def test_recorded_training_and_selection_policy(self):
        common = dict(epochs=500, stopping_step=60, min_epochs=80, train_batch_size=8192,
                      eval_batch_size=4096, learner='adam', learning_rate_scheduler=[.96, 50],
                      weight_decay=0.0, eval_step=1, training_neg_sample_num=1,
                      use_neg_sampling=True, use_full_sampling=False, grad_sample_batches=1,
                      metrics=['Recall', 'NDCG', 'Precision', 'MAP'], topk=[5, 10, 20, 50],
                      valid_metric='Recall@20', save_recommended_topk=False,
                      visual_user_graph='shared_text', explicit_visual_graph='raw_knn',
                      run_dir=None, checkpoint_dir='saved/verified')
        for dataset in self.EXPECTED:
            with self.subTest(dataset=dataset):
                cfg = self.config(dataset)
                for key, value in common.items():
                    self.assertEqual(cfg[key], value, key)
                self.assertTrue(cfg['valid_metric_bigger'])

    def test_flat_presets_agree_with_automatic_defaults(self):
        for dataset in self.EXPECTED:
            with self.subTest(dataset=dataset):
                path = ROOT / 'main/src/configs/experiments' / (dataset + '_verified.yaml')
                with path.open(encoding='utf-8') as handle:
                    values = entry.load_yaml_overrides(handle)
                cfg = self.config(dataset)
                for key, value in values.items():
                    self.assertEqual(cfg[key], value, key)
                self.assertFalse(Path(values['checkpoint_dir']).is_absolute())
                self.assertIsNone(values['run_dir'])

    def test_explicit_overrides_beat_dataset_defaults(self):
        cfg = self.config('sports', dict(learning_rate=.004, seed=[7], epochs=1,
                                        max_p2b_k=4, target_recall10=None,
                                        target_recall20=None, run_dir='custom/run'))
        self.assertEqual(cfg['learning_rate'], .004)
        self.assertEqual(cfg['seed'], [7])
        self.assertEqual(cfg['epochs'], 1)
        self.assertEqual(cfg['max_p2b_k'], 4)
        self.assertIsNone(cfg['target_recall10'])
        self.assertIsNone(cfg['target_recall20'])
        self.assertEqual(cfg['run_dir'], 'custom/run')
        self.assertEqual(cfg['bm_loss'], .005)

    def test_yaml_overrides_keep_exponent_only_float_numeric(self):
        cfg = self.config('baby', entry.load_yaml_overrides('reg_weight_1: 1e-05\nlearning_rate: 7e-4\n'))
        self.assertIs(type(cfg['reg_weight_1']), float)
        self.assertEqual(cfg['reg_weight_1'], 1e-5)
        self.assertEqual(cfg['learning_rate'], .0007)

    def test_electronics_retains_legacy_defaults(self):
        cfg = self.config('elec')
        expected = dict(p2b_weight=.2, seed=[13], learning_rate=.001,
                        bm_loss=.01, bm_temp=.2, um_loss=.01, um_temp=.2, vt_loss=.1,
                        stopping_step=15, save_recommended_topk=True, checkpoint_dir='saved')
        for key, value in expected.items():
            self.assertEqual(cfg[key], value, key)
        for key in ('target_recall10', 'target_recall20', 'gap_keep_ratio',
                    'min_epochs', 'max_p2b_k', 'run_dir'):
            self.assertIsNone(cfg[key], key)

    def test_other_models_do_not_receive_mage_dataset_profiles(self):
        cfg = Config('unconfigured_model', 'sports', {'use_gpu': False})
        self.assertEqual(cfg['seed'], [999])
        self.assertEqual(cfg['stopping_step'], 15)
        self.assertEqual(cfg['learning_rate'], .001)
        for key in ('p2b_weight', 'max_p2b_k', 'target_recall10', 'min_epochs'):
            self.assertIsNone(cfg[key], key)


if __name__ == '__main__':
    unittest.main()
