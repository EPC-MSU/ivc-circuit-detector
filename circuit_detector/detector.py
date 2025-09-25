"""
Circuit Detector API Module

This module provides the main API for circuit classification from I-V curves.
It includes functions for feature extraction, model training, inference, and model persistence.
"""

from typing import Dict, List, Union, Optional, Any
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

    @property
    def feature_vector(self) -> np.ndarray:
        """Get the feature vector as numpy array."""
        pass

    @property
    def feature_names(self) -> List[str]:
        """Get names of the extracted features."""
        pass


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
