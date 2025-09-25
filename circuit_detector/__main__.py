import argparse
import logging
import sys
from pathlib import Path

from circuit_detector.detector import (
    train_classifier, 
    save_model, 
    load_model, 
    predict_circuit_class,
    extract_features_from_uzf
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
    save_model(classifier, model_path)
    
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
    classifier = load_model(model_path)
    
    logging.info(f"Predicting circuit class for: {uzf_path}")
    result = predict_circuit_class(uzf_path, classifier)
    
    print(f"Prediction Results:")
    print(f"  Circuit Class: {result['class_name']} (ID: {result['class_id']})")
    print(f"  Confidence: {result['confidence']:.3f}")
    
    if args.verbose:
        print(f"  Class Probabilities:")
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
    classifier = load_model(model_path)
    
    logging.info(f"Evaluating model on test dataset: {test_dir}")
    
    # TODO: Implement evaluation logic
    # This would involve:
    # 1. Iterating through test dataset
    # 2. Making predictions
    # 3. Calculating accuracy, precision, recall, etc.
    # 4. Generating confusion matrix
    
    print("Evaluation functionality not yet implemented")


def extract_features_command(args):
    """Extract and display features from UZF file."""
    uzf_path = Path(args.uzf_file)
    
    if not uzf_path.exists():
        logging.error(f"UZF file not found: {uzf_path}")
        sys.exit(1)
    
    logging.info(f"Extracting features from: {uzf_path}")
    features = extract_features_from_uzf(uzf_path)
    
    print(f"Feature Extraction Results:")
    print(f"  Feature vector shape: {features.feature_vector.shape}")
    print(f"  Feature names: {features.feature_names}")
    
    if args.verbose:
        print(f"  Feature values:")
        for name, value in zip(features.feature_names, features.feature_vector):
            print(f"    {name}: {value:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Circuit Detector CLI for training and inference")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a circuit classifier model')
    train_parser.add_argument('--dataset-dir', default='dataset/', 
                             help='Path to dataset directory containing circuit classes (default: dataset/)')
    train_parser.add_argument('--output', default='model/model.pt',
                             help='Output path for trained model file (default: model/model.pt)')
    train_parser.add_argument('--model-params', 
                             help='Model parameters as comma-separated key=value pairs')
    train_parser.set_defaults(func=train_command)
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict circuit class from UZF file')
    predict_parser.add_argument('--model', default='model/model.pt',
                               help='Path to trained model file (default: model/model.pt)')
    predict_parser.add_argument('--uzf-file', required=True,
                               help='Path to UZF file for prediction')
    predict_parser.add_argument('-v', '--verbose', action='store_true',
                               help='Show detailed prediction probabilities')
    predict_parser.set_defaults(func=predict_command)
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    eval_parser.add_argument('--model', default='model/model.pt',
                            help='Path to trained model file (default: model/model.pt)')
    eval_parser.add_argument('--test-dir', required=True,
                            help='Path to test dataset directory')
    eval_parser.set_defaults(func=evaluate_command)
    
    # Extract features command
    features_parser = subparsers.add_parser('features', help='Extract and display features from UZF file')
    features_parser.add_argument('--uzf-file', required=True,
                                help='Path to UZF file for feature extraction')
    features_parser.add_argument('-v', '--verbose', action='store_true',
                                help='Show detailed feature values')
    features_parser.set_defaults(func=extract_features_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except Exception as e:
        logging.error(f"Command failed: {e}")
        if logging.getLogger().level == logging.DEBUG:
            raise
        sys.exit(1)