from logging import getLogger
from itertools import product
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str
import platform
import os
import torch
import json
import uuid


def quick_start(model, dataset, config_dict, save_model=True):
    # merge config dict
    config = Config(model, dataset, config_dict)
    local_time = init_logger(config)
    logger = getLogger()
    # print config infor
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    # load data
    dataset = RecDataset(config)
    # print dataset statistics
    logger.info(str(dataset))

    train_dataset, valid_dataset, test_dataset = dataset.split()
    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))

    # wrap into dataloader
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    (valid_data, test_data) = (
        EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
        EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))

    ############ Dataset loadded, run model
    hyper_ret = []
    best_valid_value = None
    idx = best_valid_idx = 0

    # A supplied run_dir is a complete, independently auditable experiment.
    # Otherwise allocate a fresh directory to avoid overwriting another invocation.
    root_dir = config['run_dir']
    if root_dir is None:
        save_dir = config['checkpoint_dir'] or 'saved'
        root_dir = os.path.join(save_dir, '{}_{}_{}'.format(config['dataset'], local_time, uuid.uuid4().hex[:8]))
    os.makedirs(root_dir, exist_ok=True)
    global_best_model_path = os.path.join(root_dir, f"{config['model']}_global_best.pth")
    run_results = []

    logger.info('\n\n=================================\n\n')

    # hyper-parameters
    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
    # combinations
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    for hyper_tuple in combinators:
        # random seed reset
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'])
        config['run_dir'] = root_dir if total_loops == 1 else os.path.join(root_dir, 'combo_{:03d}'.format(idx))
        os.makedirs(config['run_dir'], exist_ok=True)
        with open(os.path.join(config['run_dir'], 'config.json'), 'w', encoding='utf-8') as handle:
            json.dump(config.final_config_dict, handle, indent=2, default=str)

        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx + 1, total_loops, config['hyper_parameters'], hyper_tuple))

        # set random state of dataloader
        train_data.pretrain_setup()
        # model loading and initialization
        model = get_model(config['model'])(config, train_data, local_time).to(config['device'])
        logger.info(model)

        # trainer loading and initialization
        trainer = get_trainer()(config, model)
        # debug
        # model training
        best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(train_data, valid_data=valid_data,
                                                                                test_data=test_data, saved=save_model)
        #########
        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))

        result = trainer.run_result()
        result['hyper_parameters'] = dict(zip(config['hyper_parameters'], hyper_tuple))
        run_results.append(result)
        trainer._write_json(os.path.join(config['run_dir'], 'run_result.json'), result)

        # The trainer restored the actual best validation epoch before returning.
        if best_valid_value is None or _is_better_valid(best_valid_score, best_valid_value, config['valid_metric_bigger']):
            best_valid_value = best_valid_score
            best_valid_idx = idx
            if save_model:
                logger.info('Saving validation-selected model to %s', global_best_model_path)
                torch.save(model.state_dict(), global_best_model_path)

        idx += 1

        logger.info('best valid result: {}'.format(dict2str(best_valid_result)))
        logger.info('test result: {}'.format(dict2str(best_test_upon_valid)))
        logger.info('████Current BEST████:\nParameters: {}={},\n'
                    'Valid: {},\nTest: {}\n\n\n'.format(config['hyper_parameters'],
                                                        hyper_ret[best_valid_idx][0],
                                                        dict2str(hyper_ret[best_valid_idx][1]),
                                                        dict2str(hyper_ret[best_valid_idx][2])))

    # log info
    logger.info('\n============All Over=====================')
    for (p, k, v) in hyper_ret:
        logger.info('Parameters: {}={},\n best valid: {},\n test paired with best valid: {}'.format(config['hyper_parameters'],
                                                                                  p, dict2str(k), dict2str(v)))

    logger.info('\n\n█████████████ BEST ████████████████')
    logger.info('\tParameters: {}={},\nValid: {},\nTest: {}\n\n'.format(config['hyper_parameters'],
                                                                        hyper_ret[best_valid_idx][0],
                                                                        dict2str(hyper_ret[best_valid_idx][1]),
                                                                        dict2str(hyper_ret[best_valid_idx][2])))
    summary = {'selection_policy': 'validation', 'selected_combination': best_valid_idx,
               'selected': run_results[best_valid_idx], 'runs': run_results}
    trainer._write_json(os.path.join(root_dir, 'search_result.json'), summary)
    return summary


def _is_better_valid(candidate, incumbent, bigger=True):
    """A single comparison policy for cross-combination selection."""
    return candidate > incumbent if bigger else candidate < incumbent
