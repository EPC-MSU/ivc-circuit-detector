"""
Circuit Detector Module

A machine learning module for equivalent circuit recognition from I-V curves.
Provides feature extraction from UZF files and circuit classification capabilities.
"""

from .detector import (
    CircuitFeatures,
    CircuitClassifier,
    extract_features_from_uzf,
    train_classifier,
    predict_circuit_class
)

__version__ = "0.1.0"

__all__ = [
    "CircuitFeatures",
    "CircuitClassifier",
    "extract_features_from_uzf",
    "train_classifier",
    "predict_circuit_class"
]