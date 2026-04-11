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
# Train with builtin parameters
python -m circuit_detector train --dataset-dir dataset_train/ --output model/model.pkl 

# Train with custom parameters
python -m circuit_detector train --dataset-dir dataset_train/ --output model/model.pkl --model-params "n_estimators=20, random_state=42, max_depth=12, min_samples_split=5, min_samples_leaf=2, class_weight=balanced"
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

#### Detect Circuit Parameters

```bash
# Detect parameters from UZF with known class
python -m circuit_detector detect-params --uzf-file test.uzf

# Detect parameters from UZF without class (requires model)
python -m circuit_detector detect-params --uzf-file test.uzf --model model/model.pkl
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
   # Detect circuit parameters from a *.uzf file
   python -m circuit_detector detect-params --uzf-file test.uzf
   ```

### 4.1.5 Python API Usage

#### Training a Classifier

```python
from circuit_detector import (
    extract_features_from_uzf,
    train_classifier,
    CircuitClassifier
)

# Train a classifier from dataset
classifier = train_classifier("dataset_train/", model_params={"n_estimators": 100})

# Save the trained model
classifier.save("model/trained_classifier.pkl")
```

#### Using a Trained Classifier for Detection

```python
from circuit_detector import (
    predict_circuit_class,
    CircuitClassifier
)

# Load a saved model
classifier = CircuitClassifier.load("model/trained_classifier.pkl")

# Predict circuit class
result = predict_circuit_class("test.uzf", classifier)
print(f"Predicted class: {result['class_name']} (confidence: {result['confidence']:.3f})")
```

## 4.2 Circuit parameters estimation

### 4.2.1 Overview

The `detect_parameters` function provides automatic estimation of circuit element values (resistances, capacitances, diode thresholds) from I-V curve features. The detection algorithm is selected automatically based on the circuit class.

### 4.2.2 Supported Circuit Classes

The function supports different circuit class groups with specialized algorithms:

| Group | Circuit Classes | Status | Elements Detected |
|-------|----------------|--------|-------------------|
| **simple_rc** | R, C, RC | Implemented | R (Ohms), C (Farads) |
| **diodes_resistors** | D, nD, DR, nDR, DnDR | Implemented | R (Ohms), Df (V), Dr (V) |
| **unresolvable_rc** | R_C | Not possible | Cannot distinguish parallel R and C |
| **not_yet** | DC, DCR, R_D, nDC, nDCR, R_nD | Not implemented | - |
| **not_yet_complex** | DC(nD_R), DnD_R | Not implemented | - |

**Legend:**

SI units are used

- R: Resistance in Ohms
- C: Capacitance in Farads
- Df: Forward diode threshold voltage in Volts
- Dr: Reverse diode threshold voltage in Volts

### 4.2.3 Detection Algorithms

#### Simple RC Circuits (R, C, RC)

Uses FFT analysis at the fundamental frequency to calculate complex impedance:
- Extracts voltage and current amplitude/phase from FFT
- Computes Z = V/I in complex form
- For R: Extracts real part
- For C: Extracts capacitance from imaginary part
- For RC: Solves parallel impedance equations

#### Diode-Resistor Circuits (D, nD, DR, nDR, DnDR)

Uses first 4 Fourier harmonics (0, 1, 2, 3) to solve nonlinear equations:
- Computes FFT of current signal
- Solves system of nonlinear equations using gradient descent variation

### 4.2.4 API Function

```python
def detect_parameters(circuit_features: CircuitFeatures) -> Dict[str, float]
```

**Args:**
- `circuit_features`: CircuitFeatures object with non-empty class_name

**Returns:**
- Dictionary mapping element names to values (e.g., {"R": 1000.0, "C": 1e-6})

**Raises:**
- `ValueError`: If class_name is empty or not recognized
- `NotImplementedError`: If algorithm not available for the circuit class

### 4.2.5 Python API Usage Examples

#### Example 1: Detect Parameters for RC Circuit

```python
from circuit_detector import (
    extract_features_from_uzf,
    detect_parameters
)

# Extract features from UZF file
features = extract_features_from_uzf("test_rc_circuit.uzf")

# Set the circuit class (from classification or known)
features.class_name = "RC"

# Detect parameters
parameters = detect_parameters(features)
print(f"Resistance: {parameters['R']:.2f} Ohms")
print(f"Capacitance: {parameters['C']:.2e} Farads")
```

#### Example 2: Complete Detection Pipeline

```python
from circuit_detector import (
    extract_features_from_uzf,
    predict_circuit_class,
    detect_parameters,
    CircuitClassifier
)

# Load trained classifier
classifier = CircuitClassifier.load("model/model.pkl")

# Predict circuit class
result = predict_circuit_class("unknown_circuit.uzf", classifier)
print(f"Detected circuit type: {result['class_name']}")
print(f"Classification confidence: {result['confidence']:.3f}")

# Get features with predicted class name
features = result["features"]

# Detect element parameters
try:
    parameters = detect_parameters(features)
    print("\nCircuit parameters:")
    for element, value in parameters.items():
        if element == "R":
            print(f"  Resistance: {value:.2f} Ohms")
        elif element == "C":
            print(f"  Capacitance: {value:.2e} Farads")
        elif element == "Df":
            print(f"  Forward diode threshold: {value:.3f} V")
        elif element == "Dr":
            print(f"  Reverse diode threshold: {value:.3f} V")
except NotImplementedError as e:
    print(f"\nParameter detection not available: {e}")
```

#### Example 3: Batch Processing with Parameter Detection

```python
from pathlib import Path
from circuit_detector import (
    extract_features_from_uzf,
    predict_circuit_class,
    detect_parameters,
    CircuitClassifier
)

# Load classifier
classifier = CircuitClassifier.load("model/model.pkl")

# Process multiple UZF files
uzf_files = Path("test_circuits/").glob("*.uzf")

for uzf_file in uzf_files:
    print(f"\nProcessing: {uzf_file.name}")

    # Classify circuit
    result = predict_circuit_class(uzf_file, classifier)
    print(f"  Class: {result['class_name']} (confidence: {result['confidence']:.3f})")

    # Detect parameters if possible
    try:
        parameters = detect_parameters(result["features"])
        print(f"  Parameters: {parameters}")
    except NotImplementedError:
        print(f"  Parameters: Not available for this circuit class")
    except ValueError as e:
        print(f"  Parameters: Detection failed - {e}")
```

### 4.2.6 CLI Usage

Parameter detection is available through the `detect-params` command:

```bash
# Detect parameters from UZF file with known circuit class
python -m circuit_detector detect-params --uzf-file test.uzf

# Detect parameters from UZF file without class name (will use classifier to predict)
python -m circuit_detector detect-params --uzf-file test.uzf --model model/model.pkl
```

**Note:** If the UZF file contains a circuit class name, it will be used directly. Otherwise, the classifier specified by `--model` will predict the class first, then detect parameters.
