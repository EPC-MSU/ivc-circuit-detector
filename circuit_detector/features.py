"""
Circuit Feature Extraction Module

This module provides functionality for extracting features from I-V curve data
and managing CircuitFeatures objects for machine learning classification.
"""

from typing import List, Any, Union
from collections import OrderedDict
import numpy as np
import re
from pathlib import Path
from epcore.filemanager.ufiv import load_board_from_ufiv


class CircuitFeatures:
    """
    Container class for extracted features from I-V curve data.

    This class holds the feature vector and metadata extracted from UZF files
    that will be used for machine learning classification.
    """

    def __init__(self, comment: str, measurement_settings: Any, voltages: np.ndarray, currents: np.ndarray):
        """
        Initialize CircuitFeatures.

        Args:
            comment: Circuit comment containing component information
            measurement_settings: Measurement configuration parameters
            voltages: Voltage values from I-V curve
            currents: Current values from I-V curve
        """
        self.comment = comment
        self.measurement_settings = measurement_settings

        self.voltages = voltages
        self.currents = currents

        # Extract circuit class name from comment (if class is unknown then class_name is "")
        self.class_name = self._extract_class_name(comment)

        # Extract features as an ordered dictionary (name -> value)
        self._features = self._extract_features()

    @property
    def feature_vector(self) -> np.ndarray:
        """Get the feature vector as numpy array."""
        return np.array(list(self._features.values()))

    @property
    def feature_names(self) -> List[str]:
        """Get names of the extracted features."""
        return list(self._features.keys())

    def _extract_features(self) -> OrderedDict:
        """
        Extract numerical features from I-V curve data.

        This implementation extracts statistical features from the voltage and current
        arrays and stores them in an ordered dictionary to ensure feature names
        and values are always paired correctly.

        Returns:
            OrderedDict mapping feature names to their values
        """
        features = OrderedDict()

        # Add measurement parameters as features
        features["probe_signal_frequency [Hz]"] = self.measurement_settings.probe_signal_frequency
        features["max_voltage [V]"] = self.measurement_settings.max_voltage
        features["internal_resistance [Ohm]"] = self.measurement_settings.internal_resistance

        # Voltages and currents must be normalized because only their form matters for classification
        norm_voltage = self.voltages / self.measurement_settings.max_voltage
        max_current = self.measurement_settings.max_voltage / self.measurement_settings.internal_resistance
        norm_current = self.currents / max_current

        # Let's measure the loop square in I(t) V(t) curve. It will correspond to the circuit capacitance
        features["iv_curve_area [V*A]"] = np.trapz(norm_current, norm_voltage)

        # FFT features for first three frequencies (excluding DC component)
        # When there is no second and third harmonic the circuit contains only L, C and R components
        self.norm_voltage_fft = np.fft.fft(norm_voltage)
        self.norm_current_fft = np.fft.fft(norm_current)

        # Extract amplitude and phase for first three non-DC frequencies
        for i in range(1, 4):  # Skip DC component (index 0), take frequencies 1, 2, 3
            # Voltage FFT amplitude and phase
            voltage_amplitude = np.abs(self.norm_voltage_fft[i])
            voltage_phase = np.angle(self.norm_voltage_fft[i])
            features[f"voltage_fft_freq{i}_amplitude"] = voltage_amplitude
            features[f"voltage_fft_freq{i}_phase [rad]"] = voltage_phase

            # Current FFT amplitude and phase
            current_amplitude = np.abs(self.norm_current_fft[i])
            current_phase = np.angle(self.norm_current_fft[i])
            features[f"current_fft_freq{i}_amplitude"] = current_amplitude
            features[f"current_fft_freq{i}_phase [rad]"] = current_phase

        # Statistical features from voltages
        features["voltage_mean [V]"] = np.mean(self.voltages)
        features["voltage_min [V]"] = np.min(self.voltages)
        features["voltage_max [V]"] = np.max(self.voltages)
        features["voltage_median [v]"] = np.median(self.voltages)

        # Statistical features from currents
        features["current_mean [A]"] = np.mean(self.currents)
        features["current_min [A]"] = np.min(self.currents)
        features["current_max [A]"] = np.max(self.currents)
        features["current_median [A]"] = np.median(self.currents)

        # Resistance curve features (voltage / current)
        # Avoid division by zero by using a small epsilon
        epsilon = 1e-10
        resistances = self.voltages / (self.currents + epsilon)

        # Normalize resistance curve
        norm_resistances = resistances / self.measurement_settings.internal_resistance

        # FFT features for resistance curve
        norm_resistance_fft = np.fft.fft(norm_resistances)
        for i in range(1, 4):  # Skip DC component (index 0), take frequencies 1, 2, 3
            resistance_amplitude = np.abs(norm_resistance_fft[i])
            resistance_phase = np.angle(norm_resistance_fft[i])
            features[f"resistance_fft_freq{i}_amplitude"] = resistance_amplitude
            features[f"resistance_fft_freq{i}_phase [rad]"] = resistance_phase

        # Statistical features for resistance
        features["resistance_mean [Ohm]"] = np.mean(resistances)
        features["resistance_min [Ohm]"] = np.min(resistances)
        features["resistance_max [Ohm]"] = np.max(resistances)
        features["resistance_median [Ohm]"] = np.median(resistances)

        return features

    @staticmethod
    def _extract_class_name(comment: str) -> str:
        """
        Extract circuit class name from comment string.

        The class name is expected to be in the format "Class: [ClassName]"
        where ClassName is extracted without the brackets.

        Args:
            comment: Circuit comment string

        Returns:
            Circuit class name without brackets, or empty string if not found
        """
        pattern = r"Class:\s*\[([^\]]+)\]"
        match = re.search(pattern, comment)
        if match:
            return match.group(1)
        else:
            return ""

    def print_features(self, verbose: bool = False):
        """
        Print feature information in a human-readable format.

        Args:
            verbose: If True, show detailed feature values and names
        """
        print("Circuit Features Summary:")
        print(f"  Circuit Class: {self.class_name}")
        print(f"  Comment: {self.comment}")
        print(f"  Data Points: {len(self.voltages)} I-V measurements")

        print("  Measurement Settings:")
        print(f"    Sampling Rate: {self.measurement_settings.sampling_rate} Hz")
        print(f"    Internal Resistance: {self.measurement_settings.internal_resistance} Ohm")
        print(f"    Max Voltage: {self.measurement_settings.max_voltage} V")
        print(f"    Probe Frequency: {self.measurement_settings.probe_signal_frequency} Hz")
        print(f"    Precharge Delay: {self.measurement_settings.precharge_delay} s")

        print(f"  Feature Vector: {len(self.feature_vector)} features extracted")

        if verbose:
            print("  Detailed Features:")
            for name, value in self._features.items():
                print(f"    {name}: {value:.6f}")


def extract_features_from_uzf(uzf_path: Union[str, Path]) -> CircuitFeatures:
    """
    Extract machine learning features from UZF file.

    This function reads a UZF file using EPCore and extracts relevant features
    from the I-V curve data that can be used for circuit classification.

    Args:
        uzf_path: Path to the UZF file containing I-V curve data

    Returns:
        CircuitFeatures object containing extracted features

    Raises:
        FileNotFoundError: If UZF file doesn't exist
        ValueError: If UZF file is corrupted or invalid
    """
    uzf_path = Path(uzf_path)
    if not uzf_path.exists():
        raise FileNotFoundError(f"UZF file not found: {uzf_path}")

    try:
        # Load UZF file using EPCore
        measurement = load_board_from_ufiv(str(uzf_path))

        # Extract comment from the first element and the first pin
        # Since UZF file is a container for multiple measurements, we have to assume that
        # the first pin and element is the one we need
        comment = measurement.elements[0].pins[0].comment

        # Extract measurement settings and I-V curve data
        measurement_obj = measurement.elements[0].pins[0].measurements[0]
        measurement_settings = measurement_obj.settings
        iv_curve = measurement_obj.ivc
        voltages = np.array(iv_curve.voltages)
        currents = np.array(iv_curve.currents)

        # Validate extracted data
        if len(voltages) == 0 or len(currents) == 0:
            raise ValueError("UZF file contains empty voltage or current arrays")

        if len(voltages) != len(currents):
            raise ValueError(f"Voltage and current arrays have different lengths: {len(voltages)} vs {len(currents)}")

        # Create CircuitFeatures with extracted data
        return CircuitFeatures(
            comment=comment,
            measurement_settings=measurement_settings,
            voltages=voltages,
            currents=currents
        )

    except Exception as e:
        raise ValueError(f"Error reading UZF file {uzf_path}: {str(e)}")
