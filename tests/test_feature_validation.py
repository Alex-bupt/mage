from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'main' / 'src'))
from utils.feature_validation import validate_feature_file


class FeatureValidation(unittest.TestCase):
    def test_valid_and_expanded_catalog(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'text_feat.npy'
            np.save(path, np.ones((8,4),dtype=np.float32))
            self.assertEqual(validate_feature_file(path,8),(8,4))
            with self.assertRaisesRegex(ValueError,'8 feature rows but 3 interaction items'):
                validate_feature_file(path,3)

    def test_malformed_shape_and_missing_file(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'image_feat.npy'
            with self.assertRaises(FileNotFoundError):
                validate_feature_file(path,8)
            np.save(path,np.ones(8))
            with self.assertRaisesRegex(ValueError,'2D feature matrix'):
                validate_feature_file(path,8)


if __name__ == '__main__':
    unittest.main()
