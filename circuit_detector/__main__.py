import argparse
import logging
import sys
from pathlib import Path

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
        for param in args.model_params.split(","):
            key, value = param.split("=")
            try:
                # Try to convert to number if possible
                model_params[key] = float(value) if "." in value else int(value)
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
        for i, prob in enumerate(result["probabilities"]):
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

    try:
        # Use the classifier's evaluate method
        results = classifier.evaluate(test_dir)

        # Display formatted results
        _display_evaluation_results(results)

    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        sys.exit(1)


def _display_evaluation_results(results: dict):
    """Display evaluation results in a formatted manner."""
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)

    print("\nDataset Summary:")
    print(f"  Total files processed: {results['processed_files']}")
    print(f"  Failed files: {results['failed_files']}")
    print(f"  Classes in model: {len(results['class_names'])}")

    print("\nOverall Performance:")
    print(f"  Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  Macro avg Precision: {results['precision_macro']:.4f}")
    print(f"  Macro avg Recall: {results['recall_macro']:.4f}")
    print(f"  Macro avg F1-score: {results['f1_macro']:.4f}")
    print(f"  Weighted avg Precision: {results['precision_weighted']:.4f}")
    print(f"  Weighted avg Recall: {results['recall_weighted']:.4f}")
    print(f"  Weighted avg F1-score: {results['f1_weighted']:.4f}")

    print("\nPer-Class Performance:")
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 60)

    precision = results["per_class_metrics"]["precision"]
    recall = results["per_class_metrics"]["recall"]
    f1 = results["per_class_metrics"]["f1"]
    support = results["per_class_metrics"]["support"]

    for i, class_name in enumerate(results["class_names"]):
        if i < len(precision):  # Check if we have metrics for this class
            print(f"{class_name:<15} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {support[i]:<10}")
        else:
            print(f"{class_name:<15} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'0':<10}")

    print("\nConfusion Matrix:")
    print("Rows: True labels, Columns: Predicted labels")

    cm = results["confusion_matrix"]
    class_names = results["class_names"]

    # Print class names as column headers
    header = "        "
    for class_name in class_names:
        header += f"{class_name[:6]:<8}"
    print(header)

    # Print confusion matrix with row labels
    for i, class_name in enumerate(class_names):
        row = f"{class_name[:6]:<8}"
        for j in range(len(class_names)):
            if i < len(cm) and j < len(cm[i]):
                row += f"{cm[i][j]:<8}"
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
