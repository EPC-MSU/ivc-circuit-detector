"""
UZF Testing Tab for the GUI tool.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import os
import threading
import traceback

from tools.base_tab import BaseTab
from tools.gui_utils import safe_get_field, FieldValidationError, resolve_path

# Global reference to circuit_detector module (set by main GUI)
circuit_detector = None


def set_circuit_detector_module(module):
    """Set the circuit_detector module reference."""
    global circuit_detector
    circuit_detector = module


class UZFTestingTab(BaseTab):
    """Tab for UZF file testing and parameter detection."""

    def __init__(self, parent_notebook, root, project_root, log_callback):
        """Initialize the UZF testing tab."""
        super().__init__(parent_notebook, "UZF Testing", root, project_root, log_callback)

        # UI variables
        self.uzf_test_model_var = None
        self.uzf_test_file_var = None

        # Create the tab UI
        self.create_tab()

    def create_tab(self):
        """Create the UZF testing tab UI."""
        # Create main container
        main_container = ttk.Frame(self.frame)
        main_container.pack(fill="both", expand=True, padx=5, pady=5)

        # SETTINGS SECTION
        settings_frame = ttk.LabelFrame(main_container, text="Settings")
        settings_frame.pack(fill="x", padx=5, pady=5)

        # Model File
        model_entry_frame = ttk.Frame(settings_frame)
        model_entry_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(model_entry_frame, text="Model File:").pack(anchor="w")
        model_input_frame = ttk.Frame(model_entry_frame)
        model_input_frame.pack(fill="x", pady=(2, 0))

        self.uzf_test_model_var = tk.StringVar(value="model/model.pkl")
        uzf_test_model_entry = ttk.Entry(model_input_frame, textvariable=self.uzf_test_model_var)
        uzf_test_model_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

        ttk.Button(model_input_frame, text="Browse", command=self._browse_uzf_test_model_file).pack(side="right")

        # UZF File
        uzf_entry_frame = ttk.Frame(settings_frame)
        uzf_entry_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(uzf_entry_frame, text="UZF File:").pack(anchor="w")
        uzf_input_frame = ttk.Frame(uzf_entry_frame)
        uzf_input_frame.pack(fill="x", pady=(2, 0))

        self.uzf_test_file_var = tk.StringVar(value="")
        uzf_test_file_entry = ttk.Entry(uzf_input_frame, textvariable=self.uzf_test_file_var)
        uzf_test_file_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

        ttk.Button(uzf_input_frame, text="Browse", command=self._browse_uzf_test_file).pack(side="right")

        # RECOGNIZE BUTTON
        recognize_button_frame = ttk.Frame(main_container)
        recognize_button_frame.pack(fill="x", padx=5, pady=10)

        ttk.Button(recognize_button_frame, text="Recognize", command=self.recognize_uzf).pack(anchor="center")

    def _browse_uzf_test_model_file(self):
        """Browse for UZF test model file path (existing file)"""
        self.browse_file(
            "Select Model File for UZF Testing",
            self.uzf_test_model_var,
            [("Pickle files", "*.pkl"), ("All files", "*.*")],
            mode="open"
        )

    def _browse_uzf_test_file(self):
        """Browse for UZF file to test"""
        self.browse_file(
            "Select UZF File for Testing",
            self.uzf_test_file_var,
            [("UZF files", "*.uzf"), ("All files", "*.*")],
            mode="open"
        )

    def recognize_uzf(self):
        """Execute UZF recognition using circuit_detector API"""

        # Run recognition in a separate thread to avoid blocking the GUI
        def recognition_thread():
            try:
                # Get and validate fields
                model_file = safe_get_field(self.uzf_test_model_var, "model file path")
                uzf_file = safe_get_field(self.uzf_test_file_var, "UZF file path")

                self.log("Starting UZF recognition...")
                self.log(f"Model: {model_file}")
                self.log(f"UZF File: {uzf_file}")
                self.log("")

                # Change to project root directory
                original_cwd = os.getcwd()
                os.chdir(self.project_root)

                try:
                    # Resolve paths
                    model_path = resolve_path(model_file, self.project_root)
                    uzf_path = resolve_path(uzf_file, self.project_root)

                    # Check if files exist
                    if not model_path.exists():
                        raise FileNotFoundError(f"Model file not found: {model_path}")
                    if not uzf_path.exists():
                        raise FileNotFoundError(f"UZF file not found: {uzf_path}")

                    # Extract features from UZF
                    self.log("Extracting features from UZF file...")
                    from circuit_detector.features import extract_features_from_uzf
                    from circuit_detector.classifier import predict_circuit_class, CircuitClassifier
                    from circuit_detector.regression import detect_parameters

                    features = extract_features_from_uzf(uzf_path)

                    # If class_name is empty, use classifier to predict it
                    if not features.class_name:
                        self.log("No class name in UZF file, loading classifier...")
                        classifier = CircuitClassifier.load(model_path)

                        self.log("Predicting circuit class...")
                        result = predict_circuit_class(uzf_path, classifier)
                        features = result["features"]

                        self.log(f"Detected Circuit Class: {features.class_name}")
                        self.log(f"Confidence: {result['confidence']:.3f}")
                        self.log("")
                        self.log("Class Probabilities:")
                        for i, prob in enumerate(result["probabilities"]):
                            class_name = classifier.classes_[i]
                            self.log(f"  {class_name}: {prob:.3f}")
                    else:
                        self.log(f"Circuit Class from UZF: {features.class_name}")

                    self.log("")
                    self.log(f"Detecting parameters for circuit class: {features.class_name}")

                    # Detect parameters
                    try:
                        parameters = detect_parameters(features)

                        self.log("")
                        self.log("Detected Parameters:")
                        for element_name, value in parameters.items():
                            self.log(f"  {element_name}: {value}")

                        self.log("")
                        self.log("Recognition completed successfully!")

                        # Show success message in main thread
                        self.root.after(0, lambda: messagebox.showinfo("Success", "UZF recognition completed!"))

                    except (ValueError, NotImplementedError) as e:
                        error_msg = f"Parameter detection failed: {str(e)}"
                        self.log("")
                        self.log(error_msg)
                        # Show error message in main thread
                        self.root.after(0, lambda: messagebox.showerror("Error", error_msg))

                except Exception as e:
                    error_msg = f"Recognition failed: {str(e)}"
                    self.log("")
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
                self.log("")
                self.log(error_msg)
                self.root.after(0, lambda: messagebox.showerror("Error", error_msg))

        # Start recognition in background thread
        thread = threading.Thread(target=recognition_thread, daemon=True)
        thread.start()
