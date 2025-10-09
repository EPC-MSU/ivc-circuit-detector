"""
Circuit Detector Module

A machine learning module for equivalent circuit recognition from I-V curves.
Provides feature extraction from UZF files and circuit classification capabilities.
"""

from .features import (
    CircuitFeatures,
    extract_features_from_uzf
)
from .classifier import (
    CircuitClassifier,
    train_classifier,
    predict_circuit_class
)
from .regression import (
    detect_parameters
)

__version__ = "0.3.0"

__all__ = [
    "CircuitFeatures",
    "CircuitClassifier",
    "extract_features_from_uzf",
    "train_classifier",
    "predict_circuit_class",
    "detect_parameters"
]
