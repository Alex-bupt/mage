"""Lightweight checks requiring NumPy, without data, PyTorch or network access."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'main' / 'src'))
from stage1_process import AdvancedPaCE, Config
from utils.paths import dataset_path


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    files = list(ROOT.rglob('*.py'))
    for path in files:
        compile(path.read_text(encoding='utf-8'), str(path), 'exec')
    expected = str((ROOT / 'main' / 'data' / 'clothing').resolve())
    original = Path.cwd()
    try:
        for cwd in (ROOT, ROOT / 'main' / 'src'):
            os.chdir(cwd)
            check(dataset_path({'data_path': 'data', 'dataset': 'clothing'}) == expected, 'Relative path stability')
            check(dataset_path({'data_path': str(ROOT / 'main' / 'data'), 'dataset': 'clothing'}) == expected, 'Absolute data root')
    finally:
        os.chdir(original)

    kernel = AdvancedPaCE(Config())
    seed = np.array([1., 0., 0.])
    residual = kernel.process_vector(seed, np.array([[1., 0., 0.]]), np.array([[0., 1., 0.]]))
    np.testing.assert_allclose(residual, [0., 1., 0.], atol=1e-10)
    rng = np.random.default_rng(13)
    seed = rng.normal(size=8)
    seed /= np.linalg.norm(seed)
    residual = kernel.process_vector(seed, rng.normal(size=(4, 8)), rng.normal(size=(5, 8)))
    check(np.isfinite(residual).all(), 'Finite augmentation residual')
    check(abs(np.dot(seed, residual)) < 1e-10, 'Residual orthogonal to seed')
    for script in ('main/src/main.py', 'stage1_process.py'):
        result = subprocess.run([sys.executable, str(ROOT / script), '--help'], capture_output=True, text=True, timeout=15)
        check(result.returncode == 0, result.stderr)
    print(f'PASS: {len(files)} Python sources compile; stable data paths, synthetic residuals, two CLI entry points')
    print('Full training, GPU execution and paper metrics were not tested.')


if __name__ == '__main__':
    main()
