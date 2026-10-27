
r"""
################################
"""

import os
import itertools
import json
import math
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
import matplotlib.pyplot as plt

from time import time
from logging import getLogger

from utils.utils import get_local_time, early_stopping, dict2str
from utils.topk_evaluator import TopKEvaluator


class AbstractTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def fit(self, train_data):
        r"""Train the model based on the train data.

        """
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data.

        """

        raise NotImplementedError('Method [next] should be implemented.')


class Trainer(AbstractTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
   and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    More information can be found in [placeholder]. `model` is the instantiated object of a Model Class.

    """

    def __init__(self, config, model):
        super(Trainer, self).__init__(config, model)

        self.logger = getLogger()
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.device = config['device']
        self.weight_decay = 0.0
        if config['weight_decay'] is not None:
            wd = config['weight_decay']
            self.weight_decay = eval(wd) if isinstance(wd, str) else wd

        self.req_training = config['req_training']

        self.start_epoch = 0
        self.cur_step = 0

        tmp_dd = {}
        for j, k in list(itertools.product(config['metrics'], config['topk'])):
            tmp_dd[f'{j.lower()}@{k}'] = 0.0
        self.best_valid_score = -float('inf') if self.valid_metric_bigger else float('inf')
        self.best_valid_result = tmp_dd
        self.best_test_upon_valid = tmp_dd
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()

        #fac = lambda epoch: 0.96 ** (epoch / 50)
        lr_scheduler = config['learning_rate_scheduler']        # check zero?
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        self.lr_scheduler = scheduler

        self.eval_type = config['eval_type']
        self.evaluator = TopKEvaluator(config)

        self.item_tensor = None
        self.tot_item_num = None
        self.run_dir = self._option('run_dir', self._option('checkpoint_dir', 'saved'))
        os.makedirs(self.run_dir, exist_ok=True)
        metrics_file = self._option('metric_jsonl', 'metrics.jsonl')
        self.metric_jsonl = metrics_file if os.path.isabs(metrics_file) else os.path.join(self.run_dir, metrics_file)
        os.makedirs(os.path.dirname(os.path.abspath(self.metric_jsonl)), exist_ok=True)
        self.min_epochs = int(self._option('min_epochs', 0))
        self.grad_sample_batches = max(0, int(self._option('grad_sample_batches', 2)))
        self.best_valid = None
        self.best_test_exploratory = None
        self.best_joint_exploratory = None
        self.target_match_exploratory = None
        self.first_target_match = None
        self.epochs_completed = 0
        self.epoch_diagnostics = {}
        self.gradient_history = []
        self._best_valid_state = None

    def _option(self, name, default):
        value = self.config[name] if name in self.config else None
        return default if value is None else value

    def _synchronize(self):
        if torch.device(self.device).type == 'cuda':
            torch.cuda.synchronize(self.device)

    @staticmethod
    def _write_json(path, value):
        temporary = str(path) + '.tmp'
        with open(temporary, 'w', encoding='utf-8') as handle:
            json.dump(value, handle, indent=2, allow_nan=False, default=str)
        os.replace(temporary, path)

    def _gradient_sample(self, batch_idx):
        # Sample before clipping. All parameter norms also expose disconnected branches.
        norms = {}
        squared = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.grad is None:
                norms[name] = None
                continue
            gradient = parameter.grad.detach()
            if gradient.is_sparse:
                gradient = gradient.coalesce().values()
            norm = gradient.float().norm(2)
            squared.append(norm.square())
            norms[name] = float(norm.item())
        total = float(torch.stack(squared).sum().sqrt().item()) if squared else 0.0
        components = {}
        for name, value in getattr(self.model, 'last_loss_components', {}).items():
            components[name] = float(value.detach().item() if torch.is_tensor(value) else value)
        return {'batch': batch_idx, 'total_norm': total, 'parameter_norms': norms,
                'loss_components': components}

    def _checkpoint(self, filename, metadata, state):
        path = os.path.join(self.run_dir, filename)
        temporary = path + '.tmp'
        torch.save({'state_dict': state, **metadata}, temporary)
        os.replace(temporary, path)

    def _record_evaluation(self, epoch_idx, valid_result, test_result, update_flag, saved):
        paired = {'epoch': epoch_idx, 'epoch_number': epoch_idx + 1,
                  'valid': dict(valid_result), 'test': dict(test_result)}
        pending = []
        if update_flag:
            self.best_valid = dict(paired, selection='validation', checkpoint='best_valid.pth')
            self.best_valid_result = dict(valid_result)
            self.best_test_upon_valid = dict(test_result)
            pending.append(('best_valid.pth', self.best_valid))
        if test_result and (self.best_test_exploratory is None or
                test_result['recall@20'] > self.best_test_exploratory['test']['recall@20']):
            self.best_test_exploratory = dict(paired, selection='test_recall20_exploratory',
                                             checkpoint='best_test_exploratory.pth')
            pending.append(('best_test_exploratory.pth', self.best_test_exploratory))
        target10 = self._option('target_recall10', None)
        target20 = self._option('target_recall20', None)
        if target10 is not None and target20 is not None:
            target10, target20 = float(target10), float(target20)
            if target10 <= 0 or target20 <= 0:
                raise ValueError('Recall targets must be positive.')
            candidates = []
            for split, metrics in [('valid', valid_result), ('test', test_result)]:
                r10, r20 = metrics.get('recall@10', -1), metrics.get('recall@20', -1)
                if math.isfinite(r10) and math.isfinite(r20) and r10 >= 0 and r20 >= 0:
                    candidates.append((min(r10 / target10, r20 / target20), split))
            # Retain the strongest paired Recall result even before the target
            # is met. Maximizing R20 alone can discard the strongest R10/R20 pair.
            if candidates:
                joint_score, joint_split = max(candidates, key=lambda candidate: candidate[0])
                if self.best_joint_exploratory is None or joint_score > self.best_joint_exploratory['minimum_target_ratio']:
                    self.best_joint_exploratory = dict(
                        paired, split=joint_split, selection='joint_recall_exploratory',
                        target_recall10=target10, target_recall20=target20,
                        minimum_target_ratio=joint_score, checkpoint='best_joint_exploratory.pth')
                    pending.append(('best_joint_exploratory.pth', self.best_joint_exploratory))
            candidates = [(score, split) for score, split in candidates if score >= 1.0]
            if candidates:
                score, split = max(candidates, key=lambda candidate: candidate[0])
                candidate = dict(paired, split=split, selection='target_match_exploratory',
                                 target_recall10=target10, target_recall20=target20,
                                 minimum_target_ratio=score, checkpoint='target_match_exploratory.pth')
                if self.first_target_match is None:
                    # A historical event; the retained target checkpoint can later
                    # move to a better epoch and must not be attributed to this one.
                    self.first_target_match = {key: value for key, value in candidate.items()
                                               if key != 'checkpoint'}
                    self.first_target_match['checkpoint_retained'] = False
                if self.target_match_exploratory is None or score > self.target_match_exploratory['minimum_target_ratio']:
                    self.target_match_exploratory = candidate
                    pending.append(('target_match_exploratory.pth', candidate))
        if pending:
            # Clone once; a later optimizer step must never mutate the selected epoch.
            state = {name: tensor.detach().cpu().clone() for name, tensor in self.model.state_dict().items()}
            if update_flag:
                self._best_valid_state = state
            if saved:
                for filename, metadata in pending:
                    self._checkpoint(filename, metadata, state)

    def run_result(self):
        return {'selection_policy': 'validation; test and target maxima are exploratory',
                'epoch_index_base': 0, 'best_valid': self.best_valid,
                'best_test_exploratory': self.best_test_exploratory,
                'best_joint_exploratory': self.best_joint_exploratory,
                'target_match_exploratory': self.target_match_exploratory,
                'first_target_match': self.first_target_match,
                'epochs_completed': self.epochs_completed,
                'gradient_summary': self.gradient_history,
                'metric_jsonl': os.path.abspath(self.metric_jsonl),
                'run_dir': os.path.abspath(self.run_dir)}

    def _build_optimizer(self):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, It will return a
            tuple which includes the sum of loss in each part.
        """
        self.epoch_diagnostics = {'gradient_samples': [], 'batches': 0}
        if not self.req_training:
            return 0.0, []
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        loss_batches = []
        for batch_idx, interaction in enumerate(train_data):
            self.optimizer.zero_grad()
            losses = loss_func(interaction)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            if self._check_nan(loss):
                raise FloatingPointError('Nonfinite loss at epoch {}, batch {}'.format(epoch_idx, batch_idx))
            loss.backward()
            if batch_idx < self.grad_sample_batches:
                self.epoch_diagnostics['gradient_samples'].append(self._gradient_sample(batch_idx))
            self.epoch_diagnostics['batches'] += 1
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            loss_batches.append(loss.detach())
            # for test
            #if batch_idx == 0:
            #    break
        return total_loss, loss_batches

    def _valid_epoch(self, valid_data):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate(valid_data)
        valid_score = valid_result[self.valid_metric] if self.valid_metric else valid_result['NDCG@20']
        return valid_score, valid_result

    def _check_nan(self, loss):
        if not torch.isfinite(loss).all():
            #raise ValueError('Training loss is nan')
            return True

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        train_loss_output = 'epoch %d training [time: %.2fs, ' % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            train_loss_output = ', '.join('train_loss%d: %.4f' % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            train_loss_output += 'train loss: %.4f' % losses
        return train_loss_output + ']'

    def fit(self, train_data, valid_data=None, test_data=None, saved=False, verbose=True):
        """Train with validation early stopping and retain paired, auditable results."""
        # Each trainer represents one run; quick_start allocates independent directories.
        with open(self.metric_jsonl, 'w', encoding='utf-8'):
            pass
        for epoch_idx in range(self.start_epoch, self.epochs):
            self._synchronize()
            training_start_time = time()
            self.model.pre_epoch_processing()
            train_loss, _ = self._train_epoch(train_data, epoch_idx)
            self.lr_scheduler.step()
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            post_info = self.model.post_epoch_processing()
            self._synchronize()
            training_end_time = time()
            self.epochs_completed = epoch_idx + 1
            samples = self.epoch_diagnostics.get('gradient_samples', [])
            summary = {'epoch': epoch_idx, 'sample_count': len(samples),
                       'total_norm_mean': sum(s['total_norm'] for s in samples) / len(samples) if samples else None,
                       'total_norm_max': max((s['total_norm'] for s in samples), default=None)}
            self.gradient_history.append(summary)
            record = {'epoch': epoch_idx, 'epoch_number': epoch_idx + 1,
                      'loss': train_loss, 'train_seconds': training_end_time - training_start_time,
                      'learning_rates': [group['lr'] for group in self.optimizer.param_groups],
                      **self.epoch_diagnostics}
            if verbose:
                self.logger.info(self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss))
                if post_info is not None:
                    self.logger.info(post_info)
            stop_flag = False
            if valid_data is not None and (epoch_idx + 1) % self.eval_step == 0:
                self._synchronize()
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data)
                test_result = self.evaluate(test_data) if test_data is not None else {}
                self._synchronize()
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, self.best_valid_score, self.cur_step,
                    max_step=self.stopping_step, bigger=self.valid_metric_bigger)
                record.update(valid=dict(valid_result), test=dict(test_result),
                              evaluation_seconds=time() - valid_start_time, best_valid_updated=update_flag)
                self._record_evaluation(epoch_idx, valid_result, test_result, update_flag, saved)
                if verbose:
                    self.logger.info('valid result: ' + dict2str(valid_result))
                    self.logger.info('test result: ' + dict2str(test_result))
                stop_flag = stop_flag and self.epochs_completed >= self.min_epochs
            with open(self.metric_jsonl, 'a', encoding='utf-8') as handle:
                handle.write(json.dumps(record, allow_nan=False) + '\n')
            self._write_json(os.path.join(self.run_dir, 'run_result.json'), self.run_result())
            if stop_flag:
                if verbose:
                    self.logger.info('Finished training; best validation epoch: %s', self.best_valid['epoch'])
                break
        # Leave the live model at the selected epoch, including when saved=False.
        if self._best_valid_state is not None:
            self.model.load_state_dict(self._best_valid_state)
        return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid


    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0):
        r"""Evaluate the model based on the eval data.
        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value
        """
        self.model.eval()

        # batch full users
        batch_matrix_list = []
        for batch_idx, batched_data in enumerate(eval_data):
            # predict: interaction without item ids
            scores = self.model.full_sort_predict(batched_data)
            masked_items = batched_data[1]
            # mask out pos items
            scores[masked_items[0], masked_items[1]] = -1e10
            # rank and get top-k
            _, topk_index = torch.topk(scores, max(self.config['topk']), dim=-1)  # nusers x topk
            batch_matrix_list.append(topk_index)
        return self.evaluator.evaluate(batch_matrix_list, eval_data, is_test=is_test, idx=idx)

    def plot_train_loss(self, show=True, save_path=None):
        r"""Plot the train loss in each epoch

        Args:
            show (bool, optional): whether to show this figure, default: True
            save_path (str, optional): the data path to save the figure, default: None.
                                       If it's None, it will not be saved.
        """
        epochs = list(self.train_loss_dict.keys())
        epochs.sort()
        values = [float(self.train_loss_dict[epoch]) for epoch in epochs]
        plt.plot(epochs, values)
        plt.xticks(epochs)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)

