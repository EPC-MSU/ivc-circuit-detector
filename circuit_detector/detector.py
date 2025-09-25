"""
Circuit Detector API Module

This module provides the main API for circuit classification from I-V curves.
It includes functions for feature extraction, model training, inference, and model persistence.
"""

from typing import Dict, List, Union, Optional, Any
from collections import OrderedDict
import numpy as np
import re
from pathlib import Path
from epcore.filemanager.ufiv import load_board_from_ufiv
from sklearn.ensemble import RandomForestClassifier
import pickle


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

        # Extract circuit class name from comment
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
        pattern = r'Class:\s*\[([^\]]+)\]'
        match = re.search(pattern, comment)
        if match:
            return match.group(1)
        else:
            raise ValueError("UZF file comment does not contain class information.")

    def print(self, verbose: bool = False):
        """
        Print feature information in a human-readable format.

        Args:
            verbose: If True, show detailed feature values and names
        """
        print(f"Circuit Features Summary:")
        print(f"  Circuit Class: {self.class_name}")
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

    def __init__(self, model=None, classes=None, class_to_index=None, trained=False):
        """
        Initialize classifier.

        Args:
            model: Trained RandomForest model (optional)
            classes: List of class names (optional)
            class_to_index: Dictionary mapping class names to indices (optional)
            trained: Whether the classifier is trained (optional)
        """
        self.model = model
        self._classes = classes if classes is not None else []
        self._class_to_index = class_to_index if class_to_index is not None else {}
        self._trained = trained

    def predict(self, features: CircuitFeatures) -> int:
        """
        Predict circuit class from features.

        Args:
            features: Extracted features from I-V curve

        Returns:
            Class number representing the predicted circuit type
        """
        if not self._trained:
            raise ValueError("Classifier must be trained before making predictions")

        feature_vector = features.feature_vector.reshape(1, -1)
        return self.model.predict(feature_vector)[0]

    def predict_proba(self, features: CircuitFeatures) -> np.ndarray:
        """
        Get prediction probabilities for all classes.

        Args:
            features: Extracted features from I-V curve

        Returns:
            Array of probabilities for each class
        """
        if not self._trained:
            raise ValueError("Classifier must be trained before making predictions")

        feature_vector = features.feature_vector.reshape(1, -1)
        return self.model.predict_proba(feature_vector)[0]

    @property
    def classes_(self) -> List[str]:
        """Get list of circuit class names."""
        return self._classes.copy()

    @property
    def n_classes(self) -> int:
        """Get number of circuit classes."""
        return len(self._classes)

    def save(self, model_path: Union[str, Path]) -> None:
        """
        Save trained classifier model to file.

        Args:
            model_path: Path where to save the model file

        Raises:
            ValueError: If classifier is not trained
            IOError: If unable to write to model_path
        """
        if not self._trained:
            raise ValueError("Cannot save untrained classifier")

        model_path = Path(model_path)

        # Create directory if it doesn't exist
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data to save
        model_data = {
            'model': self.model,
            'classes': self._classes,
            'class_to_index': self._class_to_index,
            'trained': self._trained
        }

        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Model saved successfully to {model_path}")
        except Exception as e:
            raise IOError(f"Failed to save model to {model_path}: {str(e)}")

    @classmethod
    def load(cls, model_path: Union[str, Path]) -> 'CircuitClassifier':
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
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            # Validate model data structure
            required_keys = ['model', 'classes', 'class_to_index', 'trained']
            for key in required_keys:
                if key not in model_data:
                    raise ValueError(f"Invalid model file: missing '{key}' data")

            # Create and configure classifier
            classifier = cls(
                model=model_data['model'],
                classes=model_data['classes'],
                class_to_index=model_data['class_to_index'],
                trained=model_data['trained']
            )

            print(f"Model loaded successfully from {model_path}")
            print(f"Model contains {len(classifier._classes)} classes: {classifier._classes}")

            return classifier

        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path}: {str(e)}")


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
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    print(f"Scanning dataset directory: {dataset_path}")

    # Step 1: Find all UZF files and extract features
    uzf_files = list(dataset_path.rglob("*.uzf"))
    if not uzf_files:
        raise ValueError(f"No UZF files found in dataset directory: {dataset_dir}")

    print(f"Found {len(uzf_files)} UZF files")

    # Extract features from all UZF files
    all_features = []
    failed_files = []

    for i, uzf_file in enumerate(uzf_files):
        try:
            features = extract_features_from_uzf(uzf_file)
            all_features.append(features)

            # Progress reporting
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(uzf_files)} files...")

        except Exception as e:
            print(f"Failed to extract features from {uzf_file}: {e}")
            failed_files.append(uzf_file)
            continue

    if not all_features:
        raise ValueError("Failed to extract features from any UZF files")

    print(f"Successfully extracted features from {len(all_features)} files")
    if failed_files:
        print(f"Failed to process {len(failed_files)} files")

    # Step 2: Create set of unique class names
    class_names_set = set()
    for features in all_features:
        if features.class_name:  # Only add non-empty class names
            class_names_set.add(features.class_name)
        else:
            raise ValueError(f"An object without class name have been found with comment = {features.comment}")

    # Convert to sorted list for consistent ordering
    unique_classes = sorted(list(class_names_set))
    print(f"Found {len(unique_classes)} unique circuit classes: {unique_classes}")

    # Create class name to index mapping
    class_to_index = {class_name: idx for idx, class_name in enumerate(unique_classes)}

    # Step 3: Prepare training data
    x = []  # Feature vectors
    y = []  # Class indices

    for features in all_features:
        if features.class_name in class_to_index:  # Only include valid classes
            x.append(features.feature_vector)
            y.append(class_to_index[features.class_name])

    x = np.array(x)
    y = np.array(y)

    print(f"Training data shape: X={x.shape}, y={y.shape}")

    # Step 4: Train Random Forest model
    if model_params is None:
        model_params = {
            'n_estimators': 100,
            'random_state': 42,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2
        }

    print(f"Training Random Forest with parameters: {model_params}")

    rf_model = RandomForestClassifier(**model_params)
    rf_model.fit(x, y)

    # Step 5: Create and configure classifier
    classifier = CircuitClassifier(
        model=rf_model,
        classes=unique_classes,
        class_to_index=class_to_index,
        trained=True
    )

    print(f"Training completed successfully!")
    print(f"Model trained on {len(x)} samples with {len(unique_classes)} classes")

    return classifier


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
    # Step 1: Extract features from UZF file
    features = extract_features_from_uzf(uzf_path)

    # Step 2: Get prediction and probabilities
    class_id = classifier.predict(features)
    probabilities = classifier.predict_proba(features)

    # Step 3: Get class name from class_id
    class_name = classifier.classes_[class_id]

    # Step 4: Calculate confidence as the highest probability
    confidence = float(np.max(probabilities))

    # Step 5: Create comprehensive result dictionary
    result = {
        'class_id': int(class_id),
        'class_name': class_name,
        'confidence': confidence,
        'probabilities': probabilities.tolist(),
        'feature_count': len(features.feature_vector),
        'uzf_path': str(uzf_path)
    }

    return result
