# 4. Circuit detection

## 4.1 Classification

### 4.1.1 Intro

All classes are defined by [../circuit_classes](../circuit_classes) directory. The dataset for training is build upon them as described in previous chapters.

The circuit_detector module has a CLI interface to train, evaluate, and predict the circuit class. An *.uzf file is used as an object for classification. Then uzf file is converted to features with feature extraction function. The feature vector is used for classification.

### 4.1.2 API Overview

The circuit_detector module provides the following main functions:

#### Core Classes

- **CircuitFeatures** - Container for extracted features from I-V curve data
- **CircuitClassifier** - Machine learning classifier for circuit type recognition

#### Main Functions

- **extract_features_from_uzf(uzf_path)** - Converts UZF files to feature objects using EPCore
- **train_classifier(dataset_dir, model_params=None)** - Trains ML model from dataset directory
- **predict_circuit_class(uzf_path, classifier)** - Complete inference pipeline
- **CircuitClassifier.save(model_path)** / **CircuitClassifier.load(model_path)** - Model persistence methods

### 4.1.3 CLI Usage

#### Train a Model

```bash
# Train with custom parameters
python -m circuit_detector train --dataset-dir dataset_train/ --output models/my_model.pkl --model-params "key1=value1,key2=value2"
```

#### Make Predictions

```bash
# Predict using default model model/model.pkl
python -m circuit_detector predict --uzf-file test.uzf

# Predict with custom model and verbose output
python -m circuit_detector predict --model custom_model.pkl --uzf-file test.uzf --verbose
```

#### Evaluate Model Performance

```bash
# Evaluate using default model model/model.pkl
python -m circuit_detector evaluate --test-dir dataset_validate/

# Evaluate with custom model
python -m circuit_detector evaluate --model custom_model.pkl --test-dir dataset_validate/
```

#### Extract Features

```bash
# Extract and display features from UZF file
python -m circuit_detector features --uzf-file test.uzf --verbose
```

### 4.1.4 Typical workflow

   ```bash
   # Generate training dataset
   python -m generate_dataset --dataset-dir dataset_train/
   # Change parameters variation in parameters_variations
   # Done manually. Can be skipped but the evaluation quality will be poor
   # Generate validation dataset
   python -m generate_dataset --dataset-dir dataset_validate/
   # Train a classifier
   python -m circuit_detector train --dataset-dir dataset_train/
   # Evaluate the model
   python -m circuit_detector evaluate --test-dir dataset_validate/
   # Make predictions on a custom *.uzf file
   python -m circuit_detector predict --uzf-file test.uzf
   ```

### 4.1.5 Python API Usage

```python
from circuit_detector.detector import (
    extract_features_from_uzf,
    train_classifier,
    predict_circuit_class,
    CircuitClassifier
)

# Extract features from UZF file
features = extract_features_from_uzf("test.uzf")
features.print(verbose=True)

# Train a classifier
classifier = train_classifier("dataset/", model_params={"n_estimators": 100})

# Save the trained model using instance method
classifier.save("model/trained_classifier.pkl")

# Load a saved model using class method
loaded_classifier = CircuitClassifier.load("model/trained_classifier.pkl")

# Make predictions
result = predict_circuit_class("test.uzf", loaded_classifier)
print(f"Predicted class: {result['class_name']} (confidence: {result['confidence']:.3f})")
```

## 4.2 Circuit parameters estimation

Not implemented yet.
