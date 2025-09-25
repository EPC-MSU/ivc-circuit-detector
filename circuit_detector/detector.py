"""
Circuit Detector API Module

This module provides the main API for circuit classification from I-V curves.
It includes functions for feature extraction, model training, inference, and model persistence.
"""

from typing import Dict, List, Union, Optional, Any
from collections import OrderedDict
import numpy as np
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

        # Voltages and currents must be normalized because only their form matters for classification
        norm_voltage = self.voltages / self.measurement_settings.max_voltage
        max_current = self.measurement_settings.max_voltage / self.measurement_settings.internal_resistance
        norm_current = self.currents / max_current

        # Let's measure the loop square in I(t) V(t) curve. It will correspond to the circuit capacitance
        features['iv_curve_area [V*A]'] = np.trapz(norm_current, norm_voltage)

        # FFT features for first three frequencies (excluding DC component)
        # When there is no second and third harmonic the circuit contains only L, C and R components
        self.norm_voltage_fft = np.fft.fft(norm_voltage)
        self.norm_current_fft = np.fft.fft(norm_current)

        # Extract amplitude and phase for first three non-DC frequencies
        for i in range(1, 4):  # Skip DC component (index 0), take frequencies 1, 2, 3
            # Voltage FFT amplitude and phase
            voltage_amplitude = np.abs(self.norm_voltage_fft[i])
            voltage_phase = np.angle(self.norm_voltage_fft[i])
            features[f'voltage_fft_freq{i}_amplitude'] = voltage_amplitude
            features[f'voltage_fft_freq{i}_phase [rad]'] = voltage_phase

            # Current FFT amplitude and phase
            current_amplitude = np.abs(self.norm_current_fft[i])
            current_phase = np.angle(self.norm_current_fft[i])
            features[f'current_fft_freq{i}_amplitude'] = current_amplitude
            features[f'current_fft_freq{i}_phase [rad]'] = current_phase

        # Statistical features from voltages
        features['voltage_mean [V]'] = np.mean(self.voltages)
        features['voltage_std [V]'] = np.std(self.voltages)
        features['voltage_min [V]'] = np.min(self.voltages)
        features['voltage_max [V]'] = np.max(self.voltages)
        features['voltage_median [v]'] = np.median(self.voltages)
        features['voltage_range [V]'] = np.max(self.voltages) - np.min(self.voltages)

        # Statistical features from currents
        features['current_mean [A]'] = np.mean(self.currents)
        features['current_std [A]'] = np.std(self.currents)
        features['current_min [A]'] = np.min(self.currents)
        features['current_max [A]'] = np.max(self.currents)
        features['current_median [A]'] = np.median(self.currents)
        features['current_range [A]'] = np.max(self.currents) - np.min(self.currents)

        return features

    def print(self, verbose: bool = False):
        """
        Print feature information in a human-readable format.

        Args:
            verbose: If True, show detailed feature values and names
        """
        print(f"Circuit Features Summary:")
        print(f"  Comment: {self.comment}")
        print(f"  Data Points: {len(self.voltages)} I-V measurements")

        print(f"  Measurement Settings:")
        print(f"    Sampling Rate: {self.measurement_settings.sampling_rate} Hz")
        print(f"    Internal Resistance: {self.measurement_settings.internal_resistance} Ohm")
        print(f"    Max Voltage: {self.measurement_settings.max_voltage} V")
        print(f"    Probe Frequency: {self.measurement_settings.probe_signal_frequency} Hz")
        print(f"    Precharge Delay: {self.measurement_settings.precharge_delay} s")

        print(f"  Feature Vector: {len(self.feature_vector)} features extracted")

        if verbose:
            print(f"  Detailed Features:")
            for name, value in self._features.items():
                print(f"    {name}: {value:.6f}")


class CircuitClassifier:
    """
    Machine learning classifier for circuit type recognition.

    This class encapsulates the trained model and provides methods for
    inference and model management.
    """

    def __init__(self):
        """Initialize empty classifier."""
        pass

    def predict(self, features: CircuitFeatures) -> int:
        """
        Predict circuit class from features.

        Args:
            features: Extracted features from I-V curve

        Returns:
            Class number representing the predicted circuit type
        """
        pass

    def predict_proba(self, features: CircuitFeatures) -> np.ndarray:
        """
        Get prediction probabilities for all classes.

        Args:
            features: Extracted features from I-V curve

        Returns:
            Array of probabilities for each class
        """
        pass

    @property
    def classes_(self) -> List[str]:
        """Get list of circuit class names."""
        pass

    @property
    def n_classes(self) -> int:
        """Get number of circuit classes."""
        pass


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


def train_classifier(dataset_dir: Union[str, Path],
                    model_params: Optional[Dict[str, Any]] = None) -> CircuitClassifier:
    """
    Train a machine learning model on the dataset.

    This function reads all UZF files from the dataset directory structure,
    extracts features, and trains a classifier to recognize circuit types.

    Args:
        dataset_dir: Path to dataset directory with subdirectories for each circuit class
        model_params: Optional parameters for the machine learning model

    Returns:
        Trained CircuitClassifier instance

    Raises:
        FileNotFoundError: If dataset directory doesn't exist
        ValueError: If dataset structure is invalid or insufficient data
    """
    pass


def save_model(classifier: CircuitClassifier, model_path: Union[str, Path]) -> None:
    """
    Save trained classifier model to file.

    Args:
        classifier: Trained CircuitClassifier instance
        model_path: Path where to save the model file

    Raises:
        ValueError: If classifier is not trained
        IOError: If unable to write to model_path
    """
    pass


def load_model(model_path: Union[str, Path]) -> CircuitClassifier:
    """
    Load trained classifier model from file.

    Args:
        model_path: Path to saved model file

    Returns:
        Loaded CircuitClassifier instance ready for inference

    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If model file is corrupted or incompatible
    """
    pass


def predict_circuit_class(uzf_path: Union[str, Path],
                         classifier: CircuitClassifier) -> Dict[str, Any]:
    """
    Complete pipeline function: extract features and predict circuit class.

    This is a convenience function that combines feature extraction and inference
    in a single call.

    Args:
        uzf_path: Path to UZF file containing I-V curve data
        classifier: Trained CircuitClassifier instance

    Returns:
        Dictionary containing:
            - 'class_id': Predicted class number
            - 'class_name': Predicted class name
            - 'confidence': Prediction confidence score
            - 'probabilities': Full probability distribution

    Raises:
        FileNotFoundError: If UZF file doesn't exist
        ValueError: If UZF file is invalid or classifier not trained
    """
    pass
