"""
Tests for circuit detector functionality.
"""

import unittest
import numpy as np
from pathlib import Path
from circuit_detector.features import extract_features_from_uzf, CircuitFeatures


class TestFeatureExtraction(unittest.TestCase):
    """Test feature extraction from UZF files."""

    def setUp(self):
        """Set up test data paths."""
        self.test_data_dir = Path(__file__).parent / "data"
        self.test_uzf_path = self.test_data_dir / "test.uzf"

    def test_extract_features_from_uzf_success(self):
        """Test successful feature extraction from UZF file."""
        # Extract features from test UZF file
        features = extract_features_from_uzf(self.test_uzf_path)

        # Validate that we got a CircuitFeatures object
        self.assertIsInstance(features, CircuitFeatures)

        # Validate comment extraction
        self.assertIsInstance(features.comment, str)
        self.assertTrue(len(features.comment) > 0)

        # Validate measurement settings extraction
        self.assertIsNotNone(features.measurement_settings)
        # Check for expected measurement parameters
        self.assertTrue(hasattr(features.measurement_settings, "sampling_rate"))
        self.assertTrue(hasattr(features.measurement_settings, "internal_resistance"))
        self.assertTrue(hasattr(features.measurement_settings, "max_voltage"))

        # Validate voltages array
        self.assertIsInstance(features.voltages, np.ndarray)
        self.assertTrue(len(features.voltages) > 0)

        # Validate currents array
        self.assertIsInstance(features.currents, np.ndarray)
        self.assertTrue(len(features.currents) > 0)

        # Validate array lengths match
        self.assertEqual(len(features.voltages), len(features.currents))
        print(f"Successfully extracted {len(features.voltages)} I-V points")

    def test_extract_features_from_uzf_file_not_found(self):
        """Test error handling when UZF file doesn't exist."""
        non_existent_path = self.test_data_dir / "non_existent.uzf"

        with self.assertRaises(FileNotFoundError):
            extract_features_from_uzf(non_existent_path)


if __name__ == "__main__":
    unittest.main()
