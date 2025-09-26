import argparse
import logging
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from circuit_detector.features import extract_features_from_uzf
from circuit_detector.classifier import (
    train_classifier,
    predict_circuit_class,
    CircuitClassifier
)

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]", level=logging.INFO)


def train_command(args):
    """Train a circuit classifier model."""
    dataset_path = Path(args.dataset_dir)
    model_path = Path(args.output)

    if not dataset_path.exists():
        logging.error(f"Dataset directory not found: {dataset_path}")
        sys.exit(1)

    # Create model directory if it doesn't exist
    model_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Training classifier on dataset: {dataset_path}")

    model_params = {}
    if args.model_params:
        # Parse model parameters from command line (format: key=value,key=value)
        for param in args.model_params.split(','):
            key, value = param.split('=')
            try:
                # Try to convert to number if possible
                model_params[key] = float(value) if '.' in value else int(value)
            except ValueError:
                model_params[key] = value

    classifier = train_classifier(dataset_path, model_params if model_params else None)
    classifier.save(model_path)

    logging.info(f"Model trained and saved to: {model_path}")
    logging.info(f"Model classes: {classifier.classes_}")
    logging.info(f"Number of classes: {classifier.n_classes}")


def predict_command(args):
    """Predict circuit class from UZF file."""
    model_path = Path(args.model)
    uzf_path = Path(args.uzf_file)

    if not model_path.exists():
        logging.error(f"Model file not found: {model_path}")
        sys.exit(1)

    if not uzf_path.exists():
        logging.error(f"UZF file not found: {uzf_path}")
        sys.exit(1)

    logging.info(f"Loading model from: {model_path}")
    classifier = CircuitClassifier.load(model_path)

    logging.info(f"Predicting circuit class for: {uzf_path}")
    result = predict_circuit_class(uzf_path, classifier)

    print("Prediction Results:")
    print(f"  Circuit Class: {result['class_name']} (ID: {result['class_id']})")
    print(f"  Confidence: {result['confidence']:.3f}")

    if args.verbose:
        print("  Class Probabilities:")
        for i, prob in enumerate(result['probabilities']):
            class_name = classifier.classes_[i]
            print(f"    {class_name}: {prob:.3f}")


def evaluate_command(args):
    """Evaluate model performance on test dataset."""
    model_path = Path(args.model)
    test_dir = Path(args.test_dir)

    if not model_path.exists():
        logging.error(f"Model file not found: {model_path}")
        sys.exit(1)

    if not test_dir.exists():
        logging.error(f"Test directory not found: {test_dir}")
        sys.exit(1)

    logging.info(f"Loading model from: {model_path}")
    classifier = CircuitClassifier.load(model_path)

    logging.info(f"Evaluating model on test dataset: {test_dir}")

    # Step 1: Find all UZF files in test dataset
    uzf_files = list(test_dir.rglob("*.uzf"))
    if not uzf_files:
        logging.error(f"No UZF files found in test directory: {test_dir}")
        sys.exit(1)

    logging.info(f"Found {len(uzf_files)} test files")

    # Step 2: Make predictions and collect results
    y_true = []  # True class names
    y_pred = []  # Predicted class names
    y_true_indices = []  # True class indices
    y_pred_indices = []  # Predicted class indices
    failed_files = []

    for i, uzf_file in enumerate(uzf_files):
        try:
            # Extract features and get true class
            features = extract_features_from_uzf(uzf_file)
            true_class = features.class_name

            if not true_class:
                logging.warning(f"Skipping file with no class label: {uzf_file}")
                continue

            # Make prediction
            result = predict_circuit_class(uzf_file, classifier)
            pred_class = result['class_name']

            # Store results
            y_true.append(true_class)
            y_pred.append(pred_class)

            # Convert to indices for scikit-learn metrics
            if true_class in classifier._class_to_index:
                y_true_indices.append(classifier._class_to_index[true_class])
            else:
                logging.warning(f"True class '{true_class}' not in training classes, skipping file: {uzf_file}")
                y_true.pop()  # Remove the last added true class
                y_pred.pop()  # Remove the last added pred class
                continue

            y_pred_indices.append(classifier._class_to_index[pred_class])

            # Progress reporting
            if (i + 1) % 50 == 0:
                logging.info(f"Processed {i + 1}/{len(uzf_files)} files...")

        except Exception as e:
            logging.warning(f"Failed to process {uzf_file}: {e}")
            failed_files.append(uzf_file)
            continue

    if not y_true:
        logging.error("No valid predictions were made")
        sys.exit(1)

    logging.info(f"Successfully processed {len(y_true)} files")
    if failed_files:
        logging.info(f"Failed to process {len(failed_files)} files")

    # Step 3: Calculate metrics
    y_true_indices = np.array(y_true_indices)
    y_pred_indices = np.array(y_pred_indices)

    # Overall accuracy
    accuracy = accuracy_score(y_true_indices, y_pred_indices)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_indices, y_pred_indices, average=None, labels=range(len(classifier.classes_))
    )

    # Macro and weighted averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_indices, y_pred_indices, average='macro'
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_indices, y_pred_indices, average='weighted'
    )

    # Confusion matrix
    cm = confusion_matrix(y_true_indices, y_pred_indices, labels=range(len(classifier.classes_)))

    # Step 4: Display results
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)

    print("\nDataset Summary:")
    print(f"  Total files processed: {len(y_true)}")
    print(f"  Failed files: {len(failed_files)}")
    print(f"  Classes in model: {len(classifier.classes_)}")

    print("\nOverall Performance:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Macro avg Precision: {precision_macro:.4f}")
    print(f"  Macro avg Recall: {recall_macro:.4f}")
    print(f"  Macro avg F1-score: {f1_macro:.4f}")
    print(f"  Weighted avg Precision: {precision_weighted:.4f}")
    print(f"  Weighted avg Recall: {recall_weighted:.4f}")
    print(f"  Weighted avg F1-score: {f1_weighted:.4f}")

    print("\nPer-Class Performance:")
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 60)

    for i, class_name in enumerate(classifier.classes_):
        if i < len(precision):  # Check if we have metrics for this class
            print(f"{class_name:<15} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {support[i]:<10}")
        else:
            print(f"{class_name:<15} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'0':<10}")

    print("\nConfusion Matrix:")
    print("Rows: True labels, Columns: Predicted labels")

    # Print class names as column headers
    header = "        "
    for i, class_name in enumerate(classifier.classes_):
        header += f"{class_name[:6]:<8}"
    print(header)

    # Print confusion matrix with row labels
    for i, class_name in enumerate(classifier.classes_):
        row = f"{class_name[:6]:<8}"
        for j in range(len(classifier.classes_)):
            if i < cm.shape[0] and j < cm.shape[1]:
                row += f"{cm[i, j]:<8}"
            else:
                row += f"{'0':<8}"
        print(row)

    print("="*60)


def extract_features_command(args):
    """Extract and display features from UZF file."""
    uzf_path = Path(args.uzf_file)

    if not uzf_path.exists():
        logging.error(f"UZF file not found: {uzf_path}")
        sys.exit(1)

    logging.info(f"Extracting features from: {uzf_path}")
    features = extract_features_from_uzf(uzf_path)

    features.print_features(verbose=args.verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Circuit Detector CLI for training and inference")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a circuit classifier model")
    train_parser.add_argument("--dataset-dir", default="dataset/",
                              help="Path to dataset directory containing circuit classes (default: dataset/)")
    train_parser.add_argument("--output", default="model/model.pkl",
                              help="Output path for trained model file (default: model/model.pkl)")
    train_parser.add_argument("--model-params",
                              help="Model parameters as comma-separated key=value pairs")
    train_parser.set_defaults(func=train_command)

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict circuit class from UZF file")
    predict_parser.add_argument("--model", default="model/model.pkl",
                                help="Path to trained model file (default: model/model.pkl)")
    predict_parser.add_argument("--uzf-file", required=True,
                                help="Path to UZF file for prediction")
    predict_parser.add_argument("-v", "--verbose", action="store_true",
                                help="Show detailed prediction probabilities")
    predict_parser.set_defaults(func=predict_command)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model performance")
    eval_parser.add_argument("--model", default="model/model.pkl",
                             help="Path to trained model file (default: model/model.pkl)")
    eval_parser.add_argument("--test-dir", required=True,
                             help="Path to test dataset directory")
    eval_parser.set_defaults(func=evaluate_command)

    # Extract features command
    features_parser = subparsers.add_parser("features", help="Extract and display features from UZF file")
    features_parser.add_argument("--uzf-file", required=True,
                                 help="Path to UZF file for feature extraction")
    features_parser.add_argument("-v", "--verbose", action="store_true",
                                 help="Show detailed feature values")
    features_parser.set_defaults(func=extract_features_command)

    arguments = parser.parse_args()

    if not arguments.command:
        parser.print_help()
        sys.exit(1)

    try:
        arguments.func(arguments)
    except Exception as e:
        logging.error(f"Command failed: {e}")
        if logging.getLogger().level == logging.DEBUG:
            raise
        sys.exit(1)
