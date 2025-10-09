"""
Model Training and Validation Tab for the GUI tool.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import os
import threading
import traceback

from tools.base_tab import BaseTab
from tools.gui_utils import safe_get_field, safe_parse_model_params, FieldValidationError, resolve_path

# Global reference to circuit_detector module (set by main GUI)
circuit_detector = None


def set_circuit_detector_module(module):
    """Set the circuit_detector module reference."""
    global circuit_detector
    circuit_detector = module


class TrainTab(BaseTab):
    """Tab for model training and validation."""

    def __init__(self, parent_notebook, root, project_root, log_callback):
        """Initialize the training and validation tab."""
        super().__init__(parent_notebook, "Model Training and Validation", root, project_root, log_callback)

        # Training variables
        self.train_dataset_var = None
        self.train_model_file_var = None
        self.model_params_var = None

        # Validation variables
        self.val_dataset_var = None
        self.val_model_file_var = None

        # Create the tab UI
        self.create_tab()

    def create_tab(self):
        """Create the training and validation tab UI."""
        # Create main container with two columns
        main_container = ttk.Frame(self.frame)
        main_container.pack(fill="both", expand=True, padx=5, pady=5)

        # Left column for training
        left_frame = ttk.Frame(main_container)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))

        # Right column for validation
        right_frame = ttk.Frame(main_container)
        right_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))

        # Create training section
        self._create_training_section(left_frame)

        # Create validation section
        self._create_validation_section(right_frame)

    def _create_training_section(self, parent):
        """Create the model training section."""
        training_frame = ttk.LabelFrame(parent, text="Model Training")
        training_frame.pack(fill="both", expand=True, pady=(0, 5))

        # Train dataset directory
        train_dir_frame = ttk.Frame(training_frame)
        train_dir_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(train_dir_frame, text="Training Dataset:").pack(anchor="w")
        train_entry_frame = ttk.Frame(train_dir_frame)
        train_entry_frame.pack(fill="x", pady=(2, 0))

        self.train_dataset_var = tk.StringVar(value="dataset_train")
        train_dataset_entry = ttk.Entry(train_entry_frame, textvariable=self.train_dataset_var)
        train_dataset_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

        ttk.Button(train_entry_frame, text="Browse", command=self._browse_train_dataset).pack(side="right")

        # Output model file path
        train_model_frame = ttk.Frame(training_frame)
        train_model_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(train_model_frame, text="Output Model File:").pack(anchor="w")
        train_model_entry_frame = ttk.Frame(train_model_frame)
        train_model_entry_frame.pack(fill="x", pady=(2, 0))

        self.train_model_file_var = tk.StringVar(value="model/model.pkl")
        train_model_file_entry = ttk.Entry(train_model_entry_frame, textvariable=self.train_model_file_var)
        train_model_file_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

        ttk.Button(train_model_entry_frame, text="Save As...", command=self._browse_train_model_file).pack(side="right")

        # Warning label for model overwriting
        warning_label = ttk.Label(train_model_frame,
                                  text="⚠ Warning: Existing model file will be overwritten",
                                  font=("Arial", 8), foreground="orange")
        warning_label.pack(anchor="w", pady=(2, 0))

        # Model parameters
        params_frame = ttk.Frame(training_frame)
        params_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(params_frame, text="Model Parameters:").pack(anchor="w")
        self.model_params_var = tk.StringVar(value="")
        model_params_entry = ttk.Entry(params_frame, textvariable=self.model_params_var)
        model_params_entry.pack(fill="x", pady=(2, 0))

        # Help text for parameters
        help_label = ttk.Label(params_frame,
                               text="Format: key=value,key=value\n(e.g., n_estimators=200,max_depth=15)",
                               font=("Arial", 8), foreground="gray")
        help_label.pack(anchor="w", pady=(2, 0))

        # Train button
        train_button_frame = ttk.Frame(training_frame)
        train_button_frame.pack(fill="x", padx=5, pady=10)

        ttk.Button(train_button_frame, text="Train Model", command=self.train_model).pack(anchor="center")

    def _create_validation_section(self, parent):
        """Create the model validation section."""
        validation_frame = ttk.LabelFrame(parent, text="Model Validation")
        validation_frame.pack(fill="both", expand=True, pady=(0, 5))

        # Validation dataset directory
        val_dir_frame = ttk.Frame(validation_frame)
        val_dir_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(val_dir_frame, text="Validation Dataset:").pack(anchor="w")
        val_entry_frame = ttk.Frame(val_dir_frame)
        val_entry_frame.pack(fill="x", pady=(2, 0))

        self.val_dataset_var = tk.StringVar(value="dataset_validate")
        val_dataset_entry = ttk.Entry(val_entry_frame, textvariable=self.val_dataset_var)
        val_dataset_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

        ttk.Button(val_entry_frame, text="Browse", command=self._browse_val_dataset).pack(side="right")

        # Model file for validation
        val_model_frame = ttk.Frame(validation_frame)
        val_model_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(val_model_frame, text="Model File to Validate:").pack(anchor="w")
        val_model_entry_frame = ttk.Frame(val_model_frame)
        val_model_entry_frame.pack(fill="x", pady=(2, 0))

        self.val_model_file_var = tk.StringVar(value="model/model.pkl")
        val_model_file_entry = ttk.Entry(val_model_entry_frame, textvariable=self.val_model_file_var)
        val_model_file_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

        ttk.Button(val_model_entry_frame, text="Browse", command=self._browse_val_model_file).pack(side="right")

        # Validation button
        val_button_frame = ttk.Frame(validation_frame)
        val_button_frame.pack(fill="x", padx=5, pady=10)

        ttk.Button(val_button_frame, text="Validate Model", command=self.validate_model).pack(anchor="center")

    def _browse_train_dataset(self):
        """Browse for training dataset directory"""
        self.browse_directory("Select Training Dataset Directory", self.train_dataset_var)

    def _browse_val_dataset(self):
        """Browse for validation dataset directory"""
        self.browse_directory("Select Validation Dataset Directory", self.val_dataset_var)

    def _browse_train_model_file(self):
        """Browse for training model output file path"""
        self.browse_file(
            "Select Output Model File Location",
            self.train_model_file_var,
            [("Pickle files", "*.pkl"), ("All files", "*.*")],
            mode="save"
        )

    def _browse_val_model_file(self):
        """Browse for validation model file path (existing file)"""
        self.browse_file(
            "Select Existing Model File for Validation",
            self.val_model_file_var,
            [("Pickle files", "*.pkl"), ("All files", "*.*")],
            mode="open"
        )

    def train_model(self):
        """Execute model training using circuit_detector API"""

        # Run training in a separate thread to avoid blocking the GUI
        def training_thread():
            try:
                # Get and validate all training fields
                train_dataset = safe_get_field(self.train_dataset_var, "training dataset directory")
                model_file = safe_get_field(self.train_model_file_var, "output model file path")
                model_params_str = safe_get_field(self.model_params_var, "model parameters", allow_empty=True)
                model_params = safe_parse_model_params(model_params_str)

                self.log("Starting model training...")
                self.log(f"Dataset: {train_dataset}")
                self.log(f"Output model: {model_file}")
                if model_params:
                    self.log(f"Parameters: {model_params}")

                # Change to project root directory
                original_cwd = os.getcwd()
                os.chdir(self.project_root)

                try:
                    # Call the training API directly
                    dataset_path = resolve_path(train_dataset, self.project_root)

                    self.log("Training classifier...")
                    classifier, X, y, feature_names = circuit_detector.train_classifier(
                        dataset_path,
                        model_params if model_params else None,
                        return_training_data=True
                    )

                    # Save the model
                    model_path = resolve_path(model_file, self.project_root)

                    self.log("Saving model...")
                    classifier.save(model_path)

                    self.log("Model trained and saved successfully!")
                    self.log(f"Model classes: {classifier.classes_}")
                    self.log(f"Number of classes: {classifier.n_classes}")

                    # Calculate permutation importance
                    self.log("\nCalculating feature importances (permutation)...")
                    from sklearn.inspection import permutation_importance
                    perm_importance = permutation_importance(
                        classifier.model, X, y,
                        n_repeats=10,
                        random_state=42,
                        n_jobs=-1
                    )

                    self.log("\nFeature Importances (Permutation):")
                    # Sort features by importance in descending order
                    sorted_indices = perm_importance.importances_mean.argsort()[::-1]
                    for idx in sorted_indices:
                        if feature_names and idx < len(feature_names):
                            self.log(f"  {feature_names[idx]}: {perm_importance.importances_mean[idx]:.6f} +/- {perm_importance.importances_std[idx]:.6f}")
                        else:
                            self.log(f"  Feature {idx}: {perm_importance.importances_mean[idx]:.6f} +/- {perm_importance.importances_std[idx]:.6f}")

                    # Show success message in main thread
                    self.root.after(0, lambda: messagebox.showinfo("Success", "Model training completed successfully!"))

                except Exception as e:
                    error_msg = f"Training failed: {str(e)}"
                    self.log(error_msg)
                    self.log(f"Error details: {traceback.format_exc()}")
                    # Show error message in main thread
                    self.root.after(0, lambda: messagebox.showerror("Error", error_msg))

                finally:
                    os.chdir(original_cwd)

            except FieldValidationError as e:
                messagebox.showerror("Error", str(e))
                return

            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                self.log(error_msg)
                self.root.after(0, lambda: messagebox.showerror("Error", error_msg))

        # Start training in background thread
        thread = threading.Thread(target=training_thread, daemon=True)
        thread.start()

    def validate_model(self):
        """Execute model validation using CircuitClassifier API"""

        # Run validation in a separate thread to avoid blocking the GUI
        def validation_thread():
            try:
                # Get and validate all validation fields
                val_dataset = safe_get_field(self.val_dataset_var, "validation dataset directory")
                model_file = safe_get_field(self.val_model_file_var, "model file path")

                self.log("Starting model validation...")
                self.log(f"Model: {model_file}")
                self.log(f"Validation dataset: {val_dataset}")

                # Change to project root directory
                original_cwd = os.getcwd()
                os.chdir(self.project_root)

                try:
                    # Load the model
                    model_path = resolve_path(model_file, self.project_root)

                    self.log("Loading model...")
                    classifier = circuit_detector.CircuitClassifier.load(model_path)

                    # Evaluate the model
                    dataset_path = resolve_path(val_dataset, self.project_root)

                    self.log("Evaluating model...")
                    results = classifier.evaluate(dataset_path)

                    # Display results using unified function
                    circuit_detector.CircuitClassifier.display_evaluation_results(results, self.log)
                    self.log("Model validation completed successfully!")

                    # Show success message in main thread
                    self.root.after(0,
                                    lambda: messagebox.showinfo("Success", "Model validation completed successfully!"))

                except Exception as e:
                    error_msg = f"Validation failed: {str(e)}"
                    self.log(error_msg)
                    self.log(f"Error details: {traceback.format_exc()}")
                    # Show error message in main thread
                    self.root.after(0, lambda: messagebox.showerror("Error", error_msg))

                finally:
                    os.chdir(original_cwd)

            except FieldValidationError as e:
                messagebox.showerror("Error", str(e))
                return

            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                self.log(error_msg)
                self.root.after(0, lambda: messagebox.showerror("Error", error_msg))

        # Start validation in background thread
        thread = threading.Thread(target=validation_thread, daemon=True)
        thread.start()
