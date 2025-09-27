#!/usr/bin/env python
"""
GUI tool for dataset generation and management using tkinter.
Started from project root directory.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
import sys
import subprocess
from pathlib import Path
import threading
import traceback

# Import circuit_detector module API
train_classifier = None
CircuitClassifier = None


def import_circuit_detector():
    """Import circuit_detector module with proper path handling"""
    global train_classifier, CircuitClassifier

    try:
        # Add the project root to Python path if not already there
        project_root = Path(__file__).parent.parent
        print(f"Project root: {project_root}")
        print(f"Circuit detector path: {project_root / 'circuit_detector'}")
        print(f"Circuit detector exists: {(project_root / 'circuit_detector').exists()}")

        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
            print(f"Added to sys.path: {project_root}")

        from circuit_detector.classifier import train_classifier, CircuitClassifier
        print("Successfully imported circuit_detector module")
        return True
    except ImportError as e:
        print(f"Warning: Could not import circuit_detector module: {e}")
        print(f"Python path: {sys.path[:3]}...")  # Show first few paths
        return False
    except Exception as e:
        print(f"Unexpected error importing circuit_detector: {e}")
        return False


class DatasetGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Circuit Dataset Tool")
        self.root.geometry("800x600")

        # Set project root path
        self.project_root = Path(__file__).parent.parent
        self.parameters_file = self.project_root / "generate_dataset" / "parameters_variations.json"

        # Try to import circuit_detector module
        self.circuit_detector_available = import_circuit_detector()

        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Create tabs
        self.create_dataset_tab()
        self.create_train_tab()
        self.create_filter_tab()

        # Load initial parameters
        self.load_parameters()

    def create_dataset_tab(self):
        """Create the dataset generation tab"""
        self.dataset_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.dataset_frame, text="Dataset Generation")

        # Parameters section
        params_label = ttk.Label(self.dataset_frame, text="Component Parameters", font=("Arial", 12, "bold"))
        params_label.pack(anchor="w", padx=5, pady=(5, 0))

        # Create frame for parameters with scrollbar
        params_container = ttk.Frame(self.dataset_frame)
        params_container.pack(fill="both", expand=True, padx=5, pady=5)

        # Canvas and scrollbar for parameters
        self.params_canvas = tk.Canvas(params_container, height=200)
        params_scrollbar = ttk.Scrollbar(params_container, orient="vertical", command=self.params_canvas.yview)
        self.params_scrollable_frame = ttk.Frame(self.params_canvas)

        self.params_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.params_canvas.configure(scrollregion=self.params_canvas.bbox("all"))
        )

        self.params_canvas.create_window((0, 0), window=self.params_scrollable_frame, anchor="nw")
        self.params_canvas.configure(yscrollcommand=params_scrollbar.set)

        self.params_canvas.pack(side="left", fill="both", expand=True)
        params_scrollbar.pack(side="right", fill="y")

        # Storage for parameter entry widgets
        self.param_entries = {}

        # Settings section
        settings_frame = ttk.LabelFrame(self.dataset_frame, text="Generation Settings")
        settings_frame.pack(fill="x", padx=5, pady=5)

        # Dataset directory
        dir_frame = ttk.Frame(settings_frame)
        dir_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(dir_frame, text="Dataset Directory:").pack(side="left")
        self.dataset_dir_var = tk.StringVar(value="dataset")
        self.dataset_dir_entry = ttk.Entry(dir_frame, textvariable=self.dataset_dir_var, width=30)
        self.dataset_dir_entry.pack(side="left", padx=(10, 5))

        ttk.Button(dir_frame, text="Browse", command=self.browse_dataset_dir).pack(side="left")

        # Image generation checkbox
        self.image_var = tk.BooleanVar(value=True)
        image_checkbox = ttk.Checkbutton(
            settings_frame,
            text="Generate images (PNG files)",
            variable=self.image_var
        )
        image_checkbox.pack(anchor="w", padx=5, pady=5)

        # Buttons
        button_frame = ttk.Frame(self.dataset_frame)
        button_frame.pack(fill="x", padx=5, pady=10)

        ttk.Button(button_frame, text="Apply Parameters", command=self.apply_parameters).pack(side="left", padx=(0, 10))
        ttk.Button(button_frame, text="Generate Dataset", command=self.generate_dataset).pack(side="left")

        # Status text
        self.status_text = tk.Text(self.dataset_frame, height=8, state="disabled")
        self.status_text.pack(fill="x", padx=5, pady=5)

        status_scrollbar = ttk.Scrollbar(self.dataset_frame, orient="vertical", command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=status_scrollbar.set)

    def create_train_tab(self):
        """Create the model training and Validation tab"""
        self.train_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.train_frame, text="Model Training and Validation")

        # Create main container with two columns
        main_container = ttk.Frame(self.train_frame)
        main_container.pack(fill="both", expand=True, padx=5, pady=5)

        # Left column for training
        left_frame = ttk.Frame(main_container)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))

        # Right column for validation
        right_frame = ttk.Frame(main_container)
        right_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))

        # MODEL TRAINING SECTION (Left)
        training_frame = ttk.LabelFrame(left_frame, text="Model Training")
        training_frame.pack(fill="both", expand=True, pady=(0, 5))

        # Train dataset directory
        train_dir_frame = ttk.Frame(training_frame)
        train_dir_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(train_dir_frame, text="Training Dataset:").pack(anchor="w")
        train_entry_frame = ttk.Frame(train_dir_frame)
        train_entry_frame.pack(fill="x", pady=(2, 0))

        self.train_dataset_var = tk.StringVar(value="dataset_train")
        self.train_dataset_entry = ttk.Entry(train_entry_frame, textvariable=self.train_dataset_var)
        self.train_dataset_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

        ttk.Button(train_entry_frame, text="Browse", command=self.browse_train_dataset).pack(side="right")

        # Output model file path
        train_model_frame = ttk.Frame(training_frame)
        train_model_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(train_model_frame, text="Output Model File:").pack(anchor="w")
        train_model_entry_frame = ttk.Frame(train_model_frame)
        train_model_entry_frame.pack(fill="x", pady=(2, 0))

        self.train_model_file_var = tk.StringVar(value="model/model.pkl")
        self.train_model_file_entry = ttk.Entry(train_model_entry_frame, textvariable=self.train_model_file_var)
        self.train_model_file_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

        ttk.Button(train_model_entry_frame, text="Save As...", command=self.browse_train_model_file).pack(side="right")

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
        self.model_params_entry = ttk.Entry(params_frame, textvariable=self.model_params_var)
        self.model_params_entry.pack(fill="x", pady=(2, 0))

        # Help text for parameters
        help_label = ttk.Label(params_frame,
                               text="Format: key=value,key=value\n(e.g., n_estimators=200,max_depth=15)",
                               font=("Arial", 8), foreground="gray")
        help_label.pack(anchor="w", pady=(2, 0))

        # Train button
        train_button_frame = ttk.Frame(training_frame)
        train_button_frame.pack(fill="x", padx=5, pady=10)

        ttk.Button(train_button_frame, text="Train Model", command=self.train_model).pack(anchor="center")

        # MODEL VALIDATION SECTION (Right)
        validation_frame = ttk.LabelFrame(right_frame, text="Model Validation")
        validation_frame.pack(fill="both", expand=True, pady=(0, 5))

        # Validation dataset directory
        val_dir_frame = ttk.Frame(validation_frame)
        val_dir_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(val_dir_frame, text="Validation Dataset:").pack(anchor="w")
        val_entry_frame = ttk.Frame(val_dir_frame)
        val_entry_frame.pack(fill="x", pady=(2, 0))

        self.val_dataset_var = tk.StringVar(value="dataset_validate")
        self.val_dataset_entry = ttk.Entry(val_entry_frame, textvariable=self.val_dataset_var)
        self.val_dataset_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

        ttk.Button(val_entry_frame, text="Browse", command=self.browse_val_dataset).pack(side="right")

        # Model file for validation
        val_model_frame = ttk.Frame(validation_frame)
        val_model_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(val_model_frame, text="Model File to Validate:").pack(anchor="w")
        val_model_entry_frame = ttk.Frame(val_model_frame)
        val_model_entry_frame.pack(fill="x", pady=(2, 0))

        self.val_model_file_var = tk.StringVar(value="model/model.pkl")
        self.val_model_file_entry = ttk.Entry(val_model_entry_frame, textvariable=self.val_model_file_var)
        self.val_model_file_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

        ttk.Button(val_model_entry_frame, text="Browse", command=self.browse_val_model_file).pack(side="right")

        # Validation button
        val_button_frame = ttk.Frame(validation_frame)
        val_button_frame.pack(fill="x", padx=5, pady=10)

        ttk.Button(val_button_frame, text="Validate Model", command=self.validate_model).pack(anchor="center")

        # Results/Status text (spans full width at bottom)
        results_frame = ttk.LabelFrame(self.train_frame, text="Training/Validation Results")
        results_frame.pack(fill="both", expand=True, padx=5, pady=(5, 0))

        self.train_results_text = tk.Text(results_frame, height=12, state="disabled", wrap=tk.WORD)
        train_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.train_results_text.yview)
        self.train_results_text.configure(yscrollcommand=train_scrollbar.set)

        self.train_results_text.pack(side="left", fill="both", expand=True)
        train_scrollbar.pack(side="right", fill="y")

    def create_filter_tab(self):
        """Create the dataset filtering tab (empty for now)"""
        self.filter_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.filter_frame, text="Dataset Filtering")

        placeholder_label = ttk.Label(self.filter_frame,
                                      text="Dataset filtering functionality will be implemented here")
        placeholder_label.pack(expand=True)

    def load_parameters(self):
        """Load parameters from parameters_variations.json"""
        try:
            with open(self.parameters_file, "r", encoding="utf-8") as f:
                self.parameters_data = json.load(f)

            self.create_parameter_widgets()
            self.log_status("Parameters loaded successfully")

        except FileNotFoundError:
            messagebox.showerror("Error", f"Parameters file not found: {self.parameters_file}")
        except json.JSONDecodeError as e:
            messagebox.showerror("Error", f"Invalid JSON in parameters file: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading parameters: {e}")

    def create_parameter_widgets(self):
        """Create input widgets for each parameter"""
        # Clear existing widgets
        for widget in self.params_scrollable_frame.winfo_children():
            widget.destroy()
        self.param_entries.clear()

        elements = self.parameters_data.get("elements", {})

        for element_type, params_list in elements.items():
            # Element type label
            type_label = ttk.Label(self.params_scrollable_frame, text=f"{element_type}:", font=("Arial", 10, "bold"))
            type_label.pack(anchor="w", padx=5, pady=(10, 5))

            for i, param in enumerate(params_list):
                param_frame = ttk.Frame(self.params_scrollable_frame)
                param_frame.pack(fill="x", padx=20, pady=2)

                # Parameter name and units
                name = param.get("_name", f"Parameter {i + 1}")
                units = param.get("_units", "")
                label_text = f"{name} ({units}):" if units else f"{name}:"

                ttk.Label(param_frame, text=label_text, width=25).pack(side="left")

                # Get nominal values as comma-separated string
                nominal = param.get("nominal", {})
                if nominal.get("type") == "list":
                    values = nominal.get("value", [])
                    value_str = ", ".join(map(str, values))
                elif nominal.get("type") == "constant":
                    value_str = str(nominal.get("value", ""))
                else:
                    value_str = ""

                # Entry widget
                entry_var = tk.StringVar(value=value_str)
                entry = ttk.Entry(param_frame, textvariable=entry_var, width=50)
                entry.pack(side="left", padx=(10, 0))

                # Store reference
                self.param_entries[(element_type, i)] = {
                    "var": entry_var,
                    "entry": entry,
                    "original_param": param
                }

    def apply_parameters(self):
        """Apply parameter changes and save to parameters_variations.json"""
        try:
            # Update parameters data with new values
            elements = self.parameters_data.get("elements", {})

            for (element_type, param_index), entry_data in self.param_entries.items():
                value_str = entry_data["var"].get().strip()

                if not value_str:
                    continue

                # Parse comma-separated values
                try:
                    if "," in value_str:
                        # List of values
                        values = [float(v.strip()) for v in value_str.split(",") if v.strip()]
                        elements[element_type][param_index]["nominal"] = {
                            "type": "list",
                            "value": values
                        }
                    else:
                        # Single constant value
                        value = float(value_str)
                        elements[element_type][param_index]["nominal"] = {
                            "type": "constant",
                            "value": value
                        }
                except ValueError as e:
                    messagebox.showerror(f"Error {e}",
                                         f"Invalid value for {element_type} parameter {param_index + 1}: {value_str}")
                    return

            # Save updated parameters
            with open(self.parameters_file, "w", encoding="utf-8") as f:
                json.dump(self.parameters_data, f, indent=2, ensure_ascii=False)

            self.log_status("Parameters saved successfully")
            messagebox.showinfo("Success", "Parameters have been applied and saved!")

        except Exception as e:
            messagebox.showerror("Error", f"Error saving parameters: {e}")

    def browse_dataset_dir(self):
        """Browse for dataset directory"""
        directory = filedialog.askdirectory(initialdir=self.project_root)
        if directory:
            # Make path relative to project root if possible
            try:
                rel_path = Path(directory).relative_to(self.project_root)
                self.dataset_dir_var.set(str(rel_path))
            except ValueError:
                self.dataset_dir_var.set(directory)

    def generate_dataset(self):
        """Execute dataset generation"""
        try:
            # Build command
            cmd = [sys.executable, "-m", "generate_dataset"]

            # Add dataset directory if not default
            dataset_dir = self.dataset_dir_var.get().strip()
            if dataset_dir and dataset_dir != "dataset":
                cmd.extend(["--dataset-dir", dataset_dir])

            # Add image flag if enabled
            if self.image_var.get():
                cmd.append("--image")

            self.log_status("Starting dataset generation...")
            self.log_status(f"Command: {' '.join(cmd)}")

            # Change to project root directory
            original_cwd = os.getcwd()
            os.chdir(self.project_root)

            try:
                # Run command
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )

                # Read output in real-time
                while True:
                    output = process.stdout.readline()
                    if output == "" and process.poll() is not None:
                        break
                    if output:
                        self.log_status(output.strip())
                        self.root.update_idletasks()

                # Check return code
                return_code = process.poll()
                if return_code == 0:
                    self.log_status("Dataset generation completed successfully!")
                    messagebox.showinfo("Success", "Dataset generation completed!")
                else:
                    self.log_status(f"Dataset generation failed with return code {return_code}")
                    messagebox.showerror("Error", "Dataset generation failed. Check the status log for details.")

            finally:
                os.chdir(original_cwd)

        except Exception as e:
            self.log_status(f"Error: {e}")
            messagebox.showerror("Error", f"Failed to start dataset generation: {e}")

    def browse_train_dataset(self):
        """Browse for training dataset directory"""
        directory = filedialog.askdirectory(initialdir=self.project_root, title="Select Training Dataset Directory")
        if directory:
            try:
                rel_path = Path(directory).relative_to(self.project_root)
                self.train_dataset_var.set(str(rel_path))
            except ValueError:
                self.train_dataset_var.set(directory)

    def browse_val_dataset(self):
        """Browse for validation dataset directory"""
        directory = filedialog.askdirectory(initialdir=self.project_root, title="Select Validation Dataset Directory")
        if directory:
            try:
                rel_path = Path(directory).relative_to(self.project_root)
                self.val_dataset_var.set(str(rel_path))
            except ValueError:
                self.val_dataset_var.set(directory)

    def browse_train_model_file(self):
        """Browse for training model output file path"""
        file_path = filedialog.asksaveasfilename(
            initialdir=self.project_root,
            title="Select Output Model File Location",
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if file_path:
            try:
                rel_path = Path(file_path).relative_to(self.project_root)
                self.train_model_file_var.set(str(rel_path))
            except ValueError:
                self.train_model_file_var.set(file_path)

    def browse_val_model_file(self):
        """Browse for validation model file path (existing file)"""
        file_path = filedialog.askopenfilename(
            initialdir=self.project_root,
            title="Select Existing Model File for Validation",
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if file_path:
            try:
                rel_path = Path(file_path).relative_to(self.project_root)
                self.val_model_file_var.set(str(rel_path))
            except ValueError:
                self.val_model_file_var.set(file_path)

    def train_model(self):
        """Execute model training using circuit_detector API"""
        if not self.circuit_detector_available or train_classifier is None:
            # Try importing again
            if not import_circuit_detector():
                messagebox.showerror("Error",
                                     "Circuit detector module not available.\n\n"
                                     "Please ensure you are running from the project root directory\n"
                                     "and that the circuit_detector module is properly installed.")
                return

        # Get parameters
        train_dataset = self.train_dataset_var.get().strip()
        model_file = self.train_model_file_var.get().strip()
        model_params_str = self.model_params_var.get().strip()

        if not train_dataset:
            messagebox.showerror("Error", "Please specify training dataset directory")
            return

        if not model_file:
            messagebox.showerror("Error", "Please specify output model file path")
            return

        # Parse model parameters
        model_params = {}
        if model_params_str:
            try:
                for param in model_params_str.split(","):
                    key, value = param.split("=")
                    key = key.strip()
                    value = value.strip()
                    try:
                        # Try to convert to number if possible
                        model_params[key] = float(value) if "." in value else int(value)
                    except ValueError:
                        model_params[key] = value
            except Exception as e:
                messagebox.showerror("Error", f"Invalid model parameters format: {e}")
                return

        self.log_train_results("Starting model training...")
        self.log_train_results(f"Dataset: {train_dataset}")
        self.log_train_results(f"Output model: {model_file}")
        if model_params:
            self.log_train_results(f"Parameters: {model_params}")

        # Run training in a separate thread to avoid blocking the GUI
        def training_thread():
            try:
                # Change to project root directory
                original_cwd = os.getcwd()
                os.chdir(self.project_root)

                try:
                    # Call the training API directly
                    dataset_path = Path(train_dataset)
                    if not dataset_path.is_absolute():
                        dataset_path = self.project_root / dataset_path

                    self.log_train_results("Training classifier...")
                    classifier = train_classifier(dataset_path, model_params if model_params else None)

                    # Save the model
                    model_path = Path(model_file)
                    if not model_path.is_absolute():
                        model_path = self.project_root / model_path

                    self.log_train_results("Saving model...")
                    classifier.save(model_path)

                    self.log_train_results("Model trained and saved successfully!")
                    self.log_train_results(f"Model classes: {classifier.classes_}")
                    self.log_train_results(f"Number of classes: {classifier.n_classes}")

                    # Show success message in main thread
                    self.root.after(0, lambda: messagebox.showinfo("Success", "Model training completed successfully!"))

                except Exception as e:
                    error_msg = f"Training failed: {str(e)}"
                    self.log_train_results(error_msg)
                    self.log_train_results(f"Error details: {traceback.format_exc()}")
                    # Show error message in main thread
                    self.root.after(0, lambda: messagebox.showerror("Error", error_msg))

                finally:
                    os.chdir(original_cwd)

            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                self.log_train_results(error_msg)
                self.root.after(0, lambda: messagebox.showerror("Error", error_msg))

        # Start training in background thread
        thread = threading.Thread(target=training_thread, daemon=True)
        thread.start()

    def validate_model(self):
        """Execute model validation using CircuitClassifier API"""
        if not self.circuit_detector_available or CircuitClassifier is None:
            # Try importing again
            if not import_circuit_detector():
                messagebox.showerror("Error",
                                     "Circuit detector module not available.\n\n"
                                     "Please ensure you are running from the project root directory\n"
                                     "and that the circuit_detector module is properly installed.")
                return

        # Get parameters
        val_dataset = self.val_dataset_var.get().strip()
        model_file = self.val_model_file_var.get().strip()

        if not val_dataset:
            messagebox.showerror("Error", "Please specify validation dataset directory")
            return

        if not model_file:
            messagebox.showerror("Error", "Please specify model file path")
            return

        self.log_train_results("Starting model validation...")
        self.log_train_results(f"Model: {model_file}")
        self.log_train_results(f"Validation dataset: {val_dataset}")

        # Run validation in a separate thread to avoid blocking the GUI
        def validation_thread():
            try:
                # Change to project root directory
                original_cwd = os.getcwd()
                os.chdir(self.project_root)

                try:
                    # Load the model
                    model_path = Path(model_file)
                    if not model_path.is_absolute():
                        model_path = self.project_root / model_path

                    self.log_train_results("Loading model...")
                    classifier = CircuitClassifier.load(model_path)

                    # Evaluate the model
                    dataset_path = Path(val_dataset)
                    if not dataset_path.is_absolute():
                        dataset_path = self.project_root / dataset_path

                    self.log_train_results("Evaluating model...")
                    results = classifier.evaluate(dataset_path)

                    # Display results
                    self.log_train_results("\n" + "=" * 60)
                    self.log_train_results("MODEL EVALUATION RESULTS")
                    self.log_train_results("=" * 60)

                    self.log_train_results("\nDataset Summary:")
                    self.log_train_results(f"  Total files processed: {results['processed_files']}")
                    self.log_train_results(f"  Failed files: {results['failed_files']}")
                    self.log_train_results(f"  Classes in model: {len(results['class_names'])}")

                    self.log_train_results("\nOverall Performance:")
                    self.log_train_results(f"✓ Accuracy: {results['accuracy']:.4f} ({results['accuracy'] * 100:.2f}%)")
                    self.log_train_results(f"✓ Macro avg Precision: {results['precision_macro']:.4f}")
                    self.log_train_results(f"✓ Macro avg Recall: {results['recall_macro']:.4f}")
                    self.log_train_results(f"  Macro avg F1-score: {results['f1_macro']:.4f}")
                    self.log_train_results(f"  Weighted avg Precision: {results['precision_weighted']:.4f}")
                    self.log_train_results(f"  Weighted avg Recall: {results['recall_weighted']:.4f}")
                    self.log_train_results(f"  Weighted avg F1-score: {results['f1_weighted']:.4f}")

                    self.log_train_results("\nPer-Class Performance:")
                    self.log_train_results(
                        f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
                    self.log_train_results("-" * 60)

                    precision = results["per_class_metrics"]["precision"]
                    recall = results["per_class_metrics"]["recall"]
                    f1 = results["per_class_metrics"]["f1"]
                    support = results["per_class_metrics"]["support"]

                    for i, class_name in enumerate(results["class_names"]):
                        if i < len(precision):
                            self.log_train_results(
                                f"{class_name:<15} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {support[i]:<10}")
                        else:
                            self.log_train_results(f"{class_name:<15} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'0':<10}")

                    self.log_train_results("=" * 60)
                    self.log_train_results("Model validation completed successfully!")

                    # Show success message in main thread
                    self.root.after(0,
                                    lambda: messagebox.showinfo("Success", "Model validation completed successfully!"))

                except Exception as e:
                    error_msg = f"Validation failed: {str(e)}"
                    self.log_train_results(error_msg)
                    self.log_train_results(f"Error details: {traceback.format_exc()}")
                    # Show error message in main thread
                    self.root.after(0, lambda: messagebox.showerror("Error", error_msg))

                finally:
                    os.chdir(original_cwd)

            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                self.log_train_results(error_msg)
                self.root.after(0, lambda: messagebox.showerror("Error", error_msg))

        # Start validation in background thread
        thread = threading.Thread(target=validation_thread, daemon=True)
        thread.start()

    def log_train_results(self, message):
        """Add message to training results log"""
        self.train_results_text.config(state="normal")
        self.train_results_text.insert(tk.END, f"{message}\n")
        self.train_results_text.see(tk.END)
        self.train_results_text.config(state="disabled")

    def log_status(self, message):
        """Add message to status log"""
        self.status_text.config(state="normal")
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)
        self.status_text.config(state="disabled")


def main():
    # Check if we're in the right directory
    if not Path("generate_dataset").exists():
        messagebox.showerror(
            "Error",
            "Please run this tool from the project root directory.\n"
            "The 'generate_dataset' module was not found."
        )
        return

    root = tk.Tk()
    DatasetGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
