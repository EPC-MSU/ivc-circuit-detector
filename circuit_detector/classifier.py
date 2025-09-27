"""
Circuit Classification Module

This module provides the machine learning classifier for circuit type recognition
and related functions for training, inference, and model persistence.
"""

from typing import Dict, List, Union, Optional, Any, Tuple
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pickle

from .features import CircuitFeatures, extract_features_from_uzf


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
            "model": self.model,
            "classes": self._classes,
            "class_to_index": self._class_to_index,
            "trained": self._trained
        }

        try:
            with open(model_path, "wb") as f:
                pickle.dump(model_data, f)
            print(f"Model saved successfully to {model_path}")
        except Exception as e:
            raise IOError(f"Failed to save model to {model_path}: {str(e)}")

    @classmethod
    def load(cls, model_path: Union[str, Path]) -> "CircuitClassifier":
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
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)

            # Validate model data structure
            required_keys = ["model", "classes", "class_to_index", "trained"]
            for key in required_keys:
                if key not in model_data:
                    raise ValueError(f"Invalid model file: missing '{key}' data")

            # Create and configure classifier
            classifier = cls(
                model=model_data["model"],
                classes=model_data["classes"],
                class_to_index=model_data["class_to_index"],
                trained=model_data["trained"]
            )

            print(f"Model loaded successfully from {model_path}")
            print(f"Model contains {len(classifier._classes)} classes: {classifier._classes}")

            return classifier

        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path}: {str(e)}")

    def evaluate(self, test_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        Evaluate model performance on test dataset.

        Args:
            test_dir: Path to test dataset directory with UZF files

        Returns:
            Dictionary containing evaluation metrics:
                - "accuracy": Overall accuracy
                - "precision_macro": Macro-averaged precision
                - "recall_macro": Macro-averaged recall
                - "f1_macro": Macro-averaged F1-score
                - "precision_weighted": Weighted-averaged precision
                - "recall_weighted": Weighted-averaged recall
                - "f1_weighted": Weighted-averaged F1-score
                - "per_class_metrics": Per-class precision, recall, F1, support
                - "confusion_matrix": Confusion matrix
                - "processed_files": Number of successfully processed files
                - "failed_files": Number of failed files
                - "class_names": List of class names

        Raises:
            ValueError: If classifier is not trained or no valid predictions made
            FileNotFoundError: If test directory doesn't exist
        """
        if not self._trained:
            raise ValueError("Classifier must be trained before evaluation")

        # Step 1: Find all UZF files in test dataset
        uzf_files = self.validate_directory_and_find_uzf_files(test_dir)

        # Step 2: Extract features from all files
        all_features, failed_files = self.process_uzf_files(uzf_files, "test files")

        # Step 3: Make predictions and collect results
        y_true = []  # True class names
        y_pred = []  # Predicted class names
        y_true_indices = []  # True class indices
        y_pred_indices = []  # Predicted class indices

        for features in all_features:
            true_class = features.class_name

            if not true_class:
                print("Warning: Skipping file with no class label")
                continue

            # Make prediction using extracted features
            class_id = self.predict(features)
            pred_class = self.classes_[class_id]

            # Store results
            y_true.append(true_class)
            y_pred.append(pred_class)

            # Convert to indices for sklearn metrics
            if true_class in self._class_to_index:
                y_true_indices.append(self._class_to_index[true_class])
            else:
                print(f"Warning: True class '{true_class}' not in training classes, skipping")
                y_true.pop()  # Remove the last added true class
                y_pred.pop()  # Remove the last added pred class
                continue

            y_pred_indices.append(self._class_to_index[pred_class])

        if not y_true:
            raise ValueError("No valid predictions were made")

        # Step 3: Calculate metrics
        y_true_indices = np.array(y_true_indices)
        y_pred_indices = np.array(y_pred_indices)

        # Overall accuracy
        accuracy = accuracy_score(y_true_indices, y_pred_indices)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_indices, y_pred_indices, average=None, labels=range(len(self.classes_))
        )

        # Macro and weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true_indices, y_pred_indices, average="macro"
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true_indices, y_pred_indices, average="weighted"
        )

        # Confusion matrix
        cm = confusion_matrix(y_true_indices, y_pred_indices, labels=range(len(self.classes_)))

        # Step 4: Create results dictionary
        results = {
            "accuracy": float(accuracy),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(precision_weighted),
            "recall_weighted": float(recall_weighted),
            "f1_weighted": float(f1_weighted),
            "per_class_metrics": {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "f1": f1.tolist(),
                "support": support.tolist()
            },
            "confusion_matrix": cm.tolist(),
            "processed_files": len(y_true),
            "failed_files": len(failed_files),
            "class_names": self.classes_.copy()
        }

        return results

    @classmethod
    def validate_directory_and_find_uzf_files(cls, directory: Union[str, Path]) -> List[Path]:
        """
        Validate directory exists and find all UZF files within it.

        Args:
            directory: Path to directory to validate and search

        Returns:
            List of Path objects for all UZF files found

        Raises:
            FileNotFoundError: If directory doesn't exist
            ValueError: If no UZF files found in directory
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        print(f"Scanning directory: {dir_path}")

        uzf_files = list(dir_path.rglob("*.uzf"))
        if not uzf_files:
            raise ValueError(f"No UZF files found in directory: {directory}")

        print(f"Found {len(uzf_files)} UZF files")
        return uzf_files

    @classmethod
    def process_uzf_files(cls, uzf_files: List[Path], progress_message: str = "files") \
            -> Tuple[List[CircuitFeatures], List[Path]]:
        """
        Process a list of UZF files and extract features from each.

        Args:
            uzf_files: List of UZF file paths to process
            progress_message: Message to show in progress reports

        Returns:
            Tuple containing:
                - List of successfully extracted CircuitFeatures objects
                - List of failed file paths

        Raises:
            ValueError: If no features could be extracted from any files
        """
        all_features = []
        failed_files = []

        for i, uzf_file in enumerate(uzf_files):
            try:
                features = extract_features_from_uzf(uzf_file)
                all_features.append(features)

                # Progress reporting every 50 files
                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1}/{len(uzf_files)} {progress_message}...")

            except Exception as e:
                print(f"Warning: Failed to process {uzf_file}: {e}")
                failed_files.append(uzf_file)
                continue

        if not all_features:
            raise ValueError("Failed to extract features from any UZF files")

        print(f"Successfully processed {len(all_features)} {progress_message}")
        if failed_files:
            print(f"Failed to process {len(failed_files)} {progress_message}")

        return all_features, failed_files


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

    # Step 1: Find all UZF files and extract features
    uzf_files = CircuitClassifier.validate_directory_and_find_uzf_files(dataset_dir)
    all_features, failed_files = CircuitClassifier.process_uzf_files(uzf_files, "training files")

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
            "n_estimators": 100,
            "random_state": 42,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2
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

    print("Training completed successfully!")
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
            - "class_id": Predicted class number
            - "class_name": Predicted class name
            - "confidence": Prediction confidence score
            - "probabilities": Full probability distribution

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
        "class_id": int(class_id),
        "class_name": class_name,
        "confidence": confidence,
        "probabilities": probabilities.tolist(),
        "feature_count": len(features.feature_vector),
        "uzf_path": str(uzf_path)
    }

    return result
