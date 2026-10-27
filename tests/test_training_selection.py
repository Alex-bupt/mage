"""CPU regression tests for checkpoint identity and non-leaking run selection."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'main' / 'src'))
from common.trainer import Trainer
import utils.quick_start as quick_start_module


def config(directory, **overrides):
    result = dict(learner='sgd', learning_rate=0.01, epochs=10, eval_step=1,
                  stopping_step=1, clip_grad_norm=None, valid_metric='Recall@20',
                  valid_metric_bigger=True, eval_batch_size=2, device='cpu',
                  weight_decay=0, req_training=True, metrics=['Recall'], topk=[10, 20],
                  learning_rate_scheduler=[1, 50], eval_type='ranking',
                  save_recommended_topk=False, model='MOCK', run_dir=directory,
                  grad_sample_batches=1, seed=[3], hyper_parameters=['seed'],
                  train_batch_size=2, dataset='mock')
    result.update(overrides)
    return result


class TinyModel(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.user_embedding = torch.nn.Parameter(torch.tensor(0.0))
        self.last_loss_components = {}

    def pre_epoch_processing(self):
        pass

    def post_epoch_processing(self):
        pass

    def calculate_loss(self, interaction):
        loss = (self.user_embedding - interaction).square()
        self.last_loss_components = {'fit': loss.detach()}
        return loss


class ScriptedTrainer(Trainer):
    """Weights encode epochs, while independent scripted metrics control selection."""
    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        with torch.no_grad():
            self.model.user_embedding.fill_(epoch_idx + 1)
        # Step the optimizer so the real LR scheduler has the normal lifecycle.
        self.optimizer.zero_grad()
        self.model.user_embedding.grad = torch.zeros_like(self.model.user_embedding)
        self.optimizer.step()
        self.epoch_diagnostics = {'gradient_samples': [], 'batches': 1}
        return 1.0, []

    def evaluate(self, eval_data, **kwargs):
        index = int(self.model.user_embedding.item()) - 1
        return dict(self.config[eval_data + '_sequence'][index])


def recall(at10, at20):
    return {'recall@10': at10, 'recall@20': at20}


class TrainingSelection(unittest.TestCase):
    def test_best_checkpoint_is_not_last_epoch_and_target_is_one_split(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(directory, target_recall10=0.5, target_recall20=0.7,
                         valid_sequence=[recall(.4, .6), recall(.3, .5), recall(.2, .4)],
                         test_sequence=[recall(.3, .5), recall(.1, .9), recall(.6, .8)])
            model = TinyModel()
            trainer = ScriptedTrainer(cfg, model)
            trainer.fit([], 'valid', 'test', saved=True, verbose=False)
            self.assertEqual(trainer.epochs_completed, 3)
            self.assertEqual(model.user_embedding.item(), 1.0)
            for filename, expected_weight, expected_epoch in [
                    ('best_valid.pth', 1., 0), ('best_test_exploratory.pth', 2., 1),
                    ('target_match_exploratory.pth', 3., 2)]:
                checkpoint = torch.load(Path(directory) / filename, weights_only=True)
                self.assertEqual(checkpoint['state_dict']['user_embedding'].item(), expected_weight)
                self.assertEqual(checkpoint['epoch'], expected_epoch)
            result = json.loads((Path(directory) / 'run_result.json').read_text())
            selected = result['target_match_exploratory']
            self.assertEqual(selected['split'], 'test')
            self.assertEqual(selected['test'], recall(.6, .8))
            self.assertEqual(selected['valid'], recall(.2, .4))
            self.assertGreaterEqual(selected['minimum_target_ratio'], 1)
            self.assertEqual(result['best_valid']['test'], recall(.3, .5))
            lines = (Path(directory) / 'metrics.jsonl').read_text().splitlines()
            self.assertEqual(len(lines), 3)

    def test_no_target_created_by_mixing_splits_or_rounding(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(directory, epochs=2, target_recall10=.5, target_recall20=.7,
                         valid_sequence=[recall(.6, .69999), recall(.49999, .8)],
                         test_sequence=[recall(.49999, .8), recall(.6, .69999)])
            trainer = ScriptedTrainer(cfg, TinyModel())
            trainer.fit([], 'valid', 'test', saved=True, verbose=False)
            self.assertIsNone(trainer.target_match_exploratory)
            self.assertFalse((Path(directory) / 'target_match_exploratory.pth').exists())

    def test_evaluator_keeps_precision_below_target(self):
        class EvaluationData:
            def get_eval_items(self):
                return [[0]]
            def get_eval_len_list(self):
                return [1]
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(config(directory), TinyModel())
            values = [0.] * 20
            values[9], values[19] = .49999, .69999
            with patch.object(trainer.evaluator, '_calculate_metrics', return_value=[values]):
                metrics = trainer.evaluator.evaluate([torch.arange(20).unsqueeze(0)], EvaluationData())
            self.assertEqual(metrics, recall(.49999, .69999))

    def test_min_epochs_and_restore_without_disk_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(directory, min_epochs=4, stopping_step=0,
                         valid_sequence=[recall(.4, .6)] + [recall(.2, .3)] * 3,
                         test_sequence=[recall(.4, .5)] * 4)
            model = TinyModel()
            trainer = ScriptedTrainer(cfg, model)
            trainer.fit([], 'valid', 'test', saved=False, verbose=False)
            self.assertEqual(trainer.epochs_completed, 4)
            self.assertEqual(model.user_embedding.item(), 1.)
            self.assertFalse((Path(directory) / 'best_valid.pth').exists())

    def test_real_backward_gradient_and_loss_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(config(directory, epochs=1), TinyModel())
            trainer.fit([torch.tensor(2.)], verbose=False)
            record = json.loads((Path(directory) / 'metrics.jsonl').read_text())
            sample = record['gradient_samples'][0]
            self.assertEqual(sample['total_norm'], 4.)
            self.assertEqual(sample['parameter_norms']['user_embedding'], 4.)
            self.assertEqual(sample['loss_components']['fit'], 4.)
            self.assertGreaterEqual(record['train_seconds'], 0)

    def test_cross_combination_selection_uses_valid_and_saves_selected_weights(self):
        class FakeConfig(dict):
            def __init__(self, model, dataset, values):
                super().__init__(values)
                self.final_config_dict = self
            def __getitem__(self, key):
                return self.get(key)
        class FakeDataset:
            def __init__(self, cfg):
                pass
            def split(self):
                return ('train', 'valid', 'test')
        class FakeTrainData:
            def __init__(self, *args, **kwargs):
                pass
            def pretrain_setup(self):
                pass
        class CombinationTrainer(ScriptedTrainer):
            def _train_epoch(self, train_data, epoch_idx, loss_func=None):
                output = super()._train_epoch(train_data, epoch_idx, loss_func)
                with torch.no_grad():
                    self.model.user_embedding.add_(10 * self.config['choice'])
                return output

            def evaluate(self, eval_data, **kwargs):
                first = self.config['choice'] == 0
                return recall(.5, .8 if first else .3) if eval_data == 'valid' else recall(.5, .1 if first else .9)
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(directory, epochs=1, choice=[0, 1], hyper_parameters=['seed', 'choice'])
            with patch.multiple(quick_start_module, Config=FakeConfig, init_logger=lambda cfg: 'test',
                                RecDataset=FakeDataset, TrainDataLoader=FakeTrainData,
                                EvalDataLoader=lambda cfg, dataset, **kw: dataset,
                                get_model=lambda name: TinyModel,
                                get_trainer=lambda: CombinationTrainer):
                summary = quick_start_module.quick_start('MOCK', 'mock', cfg)
            self.assertEqual(summary['selected_combination'], 0)
            self.assertEqual(summary['selected']['best_valid']['test']['recall@20'], .1)
            state = torch.load(Path(directory) / 'MOCK_global_best.pth', weights_only=True)
            self.assertEqual(state['user_embedding'].item(), 1.)
            self.assertTrue((Path(directory) / 'combo_000' / 'metrics.jsonl').exists())
            self.assertTrue((Path(directory) / 'combo_001' / 'metrics.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
