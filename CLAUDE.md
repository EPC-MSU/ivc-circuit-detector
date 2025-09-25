# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python 3.6.8 project for equivalent circuit recognition from I-V curves. The module is designed to be used as an external dependency in other projects while providing complete reproducibility for the recognition models.

## Architecture

### Core Components

- **`circuit_classes/`** - Circuit schema definitions organized by circuit type (R, RC, DR, etc.)
  - Each class folder contains `.cir` (circuit file), `.png` (schema image), and `.sch` (Qucs schema)
- **`generate_dataset/`** - Dataset generation module with main entry point
  - `dataset_generator.py` - Core dataset generation logic
  - `parameters_changer.py` - Component parameter variation handling
  - `simulator_ivc.py` - I-V curve simulation using PySpice
  - `validate_circuit_classes.py` - Circuit class validation
- **`circuit_detector/`** - Detection/recognition module (currently minimal)
- **`dataset/`** - Generated output folder for training data

### Key Dependencies

- **PySpice** - Circuit simulation engine (requires NgSpice shared library)
- **matplotlib** - Plotting and visualization
- **numpy** - Numerical computations
- **EPCore** - Custom library (git+https://github.com/EPC-MSU/EPCore)

## Development Commands

### Environment Setup
```bash
# Create virtual environment (Python 3.6.8 required)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Dataset Generation
```bash
# Generate dataset with PNG images (default output: dataset/)
python -m generate_dataset --image

# Generate dataset without images (default output: dataset/)
python -m generate_dataset

# Generate dataset to custom directory
python -m generate_dataset --dataset-dir custom_dataset/

# Generate dataset with images to custom directory
python -m generate_dataset --image --dataset-dir my_data/
```

### Testing
```bash
# Run all tests
python -m unittest discover tests

# Using tox (supports py34, py36)
tox
```

## Key Configuration Files

- **`generate_dataset/measurement_settings.json`** - Measurement configurations (frequency, voltage, noise settings)
- **`generate_dataset/parameters_variations.json`** - Component parameter variation definitions

## Special Setup Requirements

This project requires manual NgSpice installation:
1. Download `ngspice-34_dll_64.zip` from SourceForge
2. Extract to `venv\Lib\site-packages\PySpice\Spice\NgSpice\Spice64_dll`
3. Create symlink: `mklink ngspice.dll ngspice-34.dll` in the dll-vs folder
4. Fix PySpice bug in `Netlist.py:165` by adding `**` to parameters

## Dataset Structure

Generated datasets follow this hierarchy:
```
dataset/
└── measurement_variant_1/
    └── [CircuitClass]/
        ├── N.cir (circuit file)
        ├── [Class]_params_N.png (I-V curve image)
        └── [Class]_params_N.uzf (EPCore format with curve data)
```

## Noise Generation

The system supports realistic noise injection with configurable SNR ratios and multiple noise copies per circuit configuration.