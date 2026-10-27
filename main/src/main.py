
import os
import argparse
os.environ['NUMEXPR_MAX_THREADS'] = '48'


def load_yaml_overrides(handle):
    """Load safe YAML while recognizing exponent-only floats such as 1e-05."""
    import re
    import yaml

    class FloatSafeLoader(yaml.SafeLoader):
        pass

    FloatSafeLoader.add_implicit_resolver(
        'tag:yaml.org,2002:float',
        re.compile(r'^[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)$'),
        list('-+0123456789'),
    )
    return yaml.load(handle, Loader=FloatSafeLoader)


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
    from utils.quick_start import quick_start
    if args.config:
        with open(args.config, encoding='utf-8') as handle:
            overrides = load_yaml_overrides(handle) or {}
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


