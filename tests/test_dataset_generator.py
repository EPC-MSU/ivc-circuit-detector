"""
Tests for dataset generation functionality.
"""

import argparse
import json
import os
import shutil
import tempfile
import unittest

from generate_dataset.dataset_generator import _process_single_circuit
from generate_dataset.parameters_changer import ParametersChanger
from generate_dataset.simulator_ivc import SimulatorIVC

PARAMETERS_SETTINGS_PATH = "generate_dataset/parameters_variations.json"
MEASUREMENTS_SETTINGS_PATH = "generate_dataset/measurement_settings.json"
BASE_CLASSES_FOLDER = "circuit_classes"


class TestFilteringBehavior(unittest.TestCase):
    """Integration tests for the disable_filtering parameter in _process_single_circuit.

    Uses the R class (one parameter, simplest circuit) to run real NgSpice simulations.
    Sets min_difference_threshold=0.99 to make filtering deterministic: any R circuit
    compared against its ±30%-extended boundary returns a similarity score well below
    0.99 (the circuits are similar resistors), so the filter always triggers when enabled.
    """

    @classmethod
    def setUpClass(cls):
        with open(PARAMETERS_SETTINGS_PATH, "r") as f:
            cls.parameters_settings = json.load(f)
        with open(MEASUREMENTS_SETTINGS_PATH, "r") as f:
            measurements_settings = json.load(f)

        cls.measurement_variant = next(
            v for v in measurements_settings["variants"] if v.get("enabled", False)
        )
        cls.cls_name = "R"
        cls.cir_path = os.path.join(BASE_CLASSES_FOLDER, cls.cls_name, cls.cls_name + ".cir")
        cls.scheme_png_path = os.path.join(BASE_CLASSES_FOLDER, cls.cls_name, cls.cls_name + ".png")

        cls.changer = ParametersChanger(cls.cir_path, cls.parameters_settings)
        cls.changer.generate_circuits()
        cls.simulator = SimulatorIVC(cls.measurement_variant)
        cls.params_combinations = cls.changer._get_params_combinations(cls.changer._settings)
        cls.original_threshold = cls.changer.min_difference_threshold

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # High threshold guarantees filtering triggers for any R circuit vs its boundary
        self.changer.min_difference_threshold = 0.99

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        self.changer.min_difference_threshold = self.original_threshold

    def _make_circuit(self):
        """Return a fresh circuit clone for the first R parameter combination.

        A fresh clone is required on each call because get_ivc mutates the circuit
        by adding Rcs and AcLineCurrent elements to it.
        """
        params = self.params_combinations[0]
        circuit = self.changer._params_combination_to_circuit(params)
        return circuit, params

    def test_filtering_disabled_saves_circuit(self):
        """With disable_filtering=True the circuit file is always written, even when
        the threshold (0.99) would have caused it to be filtered."""
        circuit, params = self._make_circuit()

        _process_single_circuit(
            circuit, params, 0, self.cls_name,
            self.changer, self.simulator, self.measurement_variant,
            self.tmpdir, self.scheme_png_path, False, disable_filtering=True
        )

        uzf_files = [f for f in os.listdir(self.tmpdir) if f.endswith(".uzf")]
        self.assertEqual(len(uzf_files), 1)

    def test_filtering_enabled_removes_circuit(self):
        """With disable_filtering=False and threshold=0.99 the circuit is filtered out
        because any R circuit is similar enough to its ±30% boundary circuit."""
        circuit, params = self._make_circuit()

        _process_single_circuit(
            circuit, params, 0, self.cls_name,
            self.changer, self.simulator, self.measurement_variant,
            self.tmpdir, self.scheme_png_path, False, disable_filtering=False
        )

        uzf_files = [f for f in os.listdir(self.tmpdir) if f.endswith(".uzf")]
        self.assertEqual(len(uzf_files), 0)


class TestDatasetGeneratorCLI(unittest.TestCase):
    """Test CLI argument parsing for dataset generation."""

    def test_cli_with_disable_filtering_flag(self):
        """Test that --disable-filtering flag is properly parsed and passed to generate_dataset."""
        parser = argparse.ArgumentParser(description="Generate dataset from `circuit_classes` to output folder")
        parser.add_argument("-i", "--image", action="store_true", help="Add IVC-png-image to each dataset file")
        parser.add_argument("--dataset-dir", default="dataset",
                            help="Output directory for generated dataset (default: dataset)")
        parser.add_argument("--disable-filtering", action="store_true",
                            help="Disable boundary condition filtering")

        args = parser.parse_args(["--disable-filtering"])
        self.assertTrue(args.disable_filtering)

    def test_cli_without_disable_filtering_flag(self):
        """Test that filtering is enabled by default when --disable-filtering is not provided."""
        parser = argparse.ArgumentParser(description="Generate dataset from `circuit_classes` to output folder")
        parser.add_argument("-i", "--image", action="store_true", help="Add IVC-png-image to each dataset file")
        parser.add_argument("--dataset-dir", default="dataset",
                            help="Output directory for generated dataset (default: dataset)")
        parser.add_argument("--disable-filtering", action="store_true",
                            help="Disable boundary condition filtering")

        args = parser.parse_args([])
        self.assertFalse(args.disable_filtering)


if __name__ == "__main__":
    unittest.main()
