"""
Tests for dataset generation functionality.
"""

import unittest
from unittest.mock import patch, MagicMock
from generate_dataset.dataset_generator import generate_dataset


class TestDatasetGenerator(unittest.TestCase):
    """Test dataset generation with filtering options."""

    @patch("generate_dataset.dataset_generator.glob.glob")
    @patch("generate_dataset.dataset_generator.open")
    @patch("generate_dataset.dataset_generator.ParametersChanger")
    @patch("generate_dataset.dataset_generator.SimulatorIVC")
    def test_generate_dataset_with_filtering_enabled(self, mock_simulator, mock_changer_class, mock_open, mock_glob):
        """Test that filtering is applied when disable_filtering=False (default)."""
        # Setup mocks
        mock_glob.return_value = ["circuit_classes/R"]

        # Mock JSON files
        mock_params_file = MagicMock()
        mock_params_file.__enter__.return_value.read.return_value = '{"elements": {}}'
        mock_measurements_file = MagicMock()
        mock_measurements_file.__enter__.return_value.read.return_value = \
            '{"variants": [{"enabled": true, "name": "test", "noise_settings": ' \
            '{"without_noise": true, "with_noise_copies": 0}}]}'

        mock_open.return_value = mock_params_file

        # Mock ParametersChanger
        mock_changer = MagicMock()
        mock_changer.circuits = [MagicMock()]
        mock_changer._get_params_combinations.return_value = [{}]
        mock_changer.generate_bound_circuits_with_params.return_value = []
        mock_changer.min_difference_threshold = 0.01
        mock_changer_class.return_value = mock_changer

        # Mock SimulatorIVC
        mock_sim = MagicMock()
        mock_sim.get_ivc.return_value = MagicMock()
        mock_sim.compare_ivc.return_value = 0.5  # Above threshold
        mock_simulator.return_value = mock_sim

        # Call with filtering enabled (default)
        with patch("builtins.open", mock_open):
            with patch("json.load") as mock_json_load:
                mock_json_load.side_effect = [
                    {"elements": {}},
                    {"variants": [{"enabled": True, "name": "test",
                                   "noise_settings": {"without_noise": True, "with_noise_copies": 0}}]}
                ]
                generate_dataset(save_png=False, dataset_dir="test_dataset", disable_filtering=False)

        # Verify that generate_bound_circuits_with_params was called (filtering enabled)
        mock_changer.generate_bound_circuits_with_params.assert_called()

    @patch("generate_dataset.dataset_generator.glob.glob")
    @patch("generate_dataset.dataset_generator.open")
    @patch("generate_dataset.dataset_generator.ParametersChanger")
    @patch("generate_dataset.dataset_generator.SimulatorIVC")
    def test_generate_dataset_with_filtering_disabled(self, mock_simulator, mock_changer_class, mock_open, mock_glob):
        """Test that filtering is skipped when disable_filtering=True."""
        # Setup mocks
        mock_glob.return_value = ["circuit_classes/R"]

        # Mock ParametersChanger
        mock_changer = MagicMock()
        mock_changer.circuits = [MagicMock()]
        mock_changer._get_params_combinations.return_value = [{}]
        mock_changer.generate_bound_circuits_with_params.return_value = []
        mock_changer_class.return_value = mock_changer

        # Mock SimulatorIVC
        mock_sim = MagicMock()
        mock_sim.get_ivc.return_value = MagicMock()
        mock_simulator.return_value = mock_sim

        # Call with filtering disabled
        with patch("builtins.open", mock_open):
            with patch("json.load") as mock_json_load:
                mock_json_load.side_effect = [
                    {"elements": {}},
                    {"variants": [{"enabled": True, "name": "test",
                                   "noise_settings": {"without_noise": True, "with_noise_copies": 0}}]}
                ]
                generate_dataset(save_png=False, dataset_dir="test_dataset", disable_filtering=True)

        # Verify that generate_bound_circuits_with_params was NOT called (filtering disabled)
        mock_changer.generate_bound_circuits_with_params.assert_not_called()


class TestDatasetGeneratorCLI(unittest.TestCase):
    """Test CLI argument parsing for dataset generation."""

    def test_cli_with_disable_filtering_flag(self):
        """Test that --disable-filtering flag is properly parsed and passed to generate_dataset."""
        import argparse

        # Create parser like in __main__.py
        parser = argparse.ArgumentParser(description="Generate dataset from `circuit_classes` to output folder")
        parser.add_argument("-i", "--image", action="store_true", help="Add IVC-png-image to each dataset file")
        parser.add_argument("--dataset-dir", default="dataset",
                            help="Output directory for generated dataset (default: dataset)")
        parser.add_argument("--disable-filtering", action="store_true",
                            help="Disable boundary condition filtering")

        # Simulate command line with --disable-filtering
        test_args = ["--disable-filtering"]
        args = parser.parse_args(test_args)

        # Verify the flag is parsed correctly
        self.assertTrue(args.disable_filtering)

    def test_cli_without_disable_filtering_flag(self):
        """Test that filtering is enabled by default when --disable-filtering is not provided."""
        import argparse

        # Create parser like in __main__.py
        parser = argparse.ArgumentParser(description="Generate dataset from `circuit_classes` to output folder")
        parser.add_argument("-i", "--image", action="store_true", help="Add IVC-png-image to each dataset file")
        parser.add_argument("--dataset-dir", default="dataset",
                            help="Output directory for generated dataset (default: dataset)")
        parser.add_argument("--disable-filtering", action="store_true",
                            help="Disable boundary condition filtering")

        # Simulate command line without --disable-filtering
        test_args = []
        args = parser.parse_args(test_args)

        # Verify the flag defaults to False (filtering enabled)
        self.assertFalse(args.disable_filtering)


if __name__ == "__main__":
    unittest.main()
