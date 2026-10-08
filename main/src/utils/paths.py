"""Shared data paths, independent of the launch directory."""
from pathlib import Path

MAIN_ROOT = Path(__file__).resolve().parents[2]


def dataset_path(config):
    root = Path(config['data_path']).expanduser()
    if not root.is_absolute():
        root = MAIN_ROOT / root
    return str((root / config['dataset']).resolve())
