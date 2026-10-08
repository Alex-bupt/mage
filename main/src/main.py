
import os
import argparse
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='mage', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='clothing', choices=['baby', 'sports', 'clothing', 'elec'])
    # parser.add_argument('--dataset', '-d', type=str, default='clothing', help='name of datasets')
    # parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    
    parser.add_argument('--config', help='YAML overrides applied after bundled defaults')
    parser.add_argument('--data-path', help='Data root: absolute, or relative to main/')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int)
    config_dict = {
        'gpu_id': 0,
    }

    args = parser.parse_args()
    import yaml
    from utils.quick_start import quick_start
    if args.config:
        with open(args.config, encoding='utf-8') as handle:
            overrides = yaml.safe_load(handle) or {}
        if not isinstance(overrides, dict):
            raise ValueError('--config must contain a YAML mapping')
        config_dict.update(overrides)
    if args.data_path is not None:
        config_dict['data_path'] = args.data_path
    if args.cpu:
        config_dict['use_gpu'] = False
    if args.epochs is not None:
        config_dict['epochs'] = args.epochs

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)


