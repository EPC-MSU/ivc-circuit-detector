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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from epcore.filemanager.ufiv import load_board_from_ufiv


global circuit_detector


def complex_import():
    global circuit_detector
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(parent_dir)
    import circuit_detector


complex_import()


class FieldValidationError(Exception):
    """Custom exception for field validation errors."""
    pass


class DatasetGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Circuit Dataset Tool")
        self.root.geometry("800x600")

        # Set project root path
        self.project_root = Path(__file__).parent.parent
        self.parameters_file = self.project_root / "generate_dataset" / "parameters_variations.json"

        # Filtering state variables
        self.current_classifier = None
        self.uzf_files = []
        self.current_file_index = 0
        self.filter_canvas = None
        self.filter_figure = None

        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Create tabs
        self.create_dataset_tab()
        self.create_train_tab()
        self.create_filter_tab()

        # Load initial parameters
        self.load_parameters()

    def _safe_get_field(self, field_var, field_name, allow_empty=False):
        """
        Safely get and validate a field value from tkinter StringVar.

        Args:
            field_var: tkinter StringVar containing the field value
            field_name: Human-readable name for error messages
            allow_empty: If True, empty values are allowed

        Returns:
            str: The field value if valid

        Raises:
            FieldValidationError: If field validation fails
        """
        try:
            value = field_var.get().strip()
            if not allow_empty and not value:
                raise FieldValidationError(f"Please specify {field_name}")
            return value
        except Exception as e:
            if isinstance(e, FieldValidationError):
                raise
            raise FieldValidationError(f"Error reading {field_name}: {e}")

    def _safe_parse_model_params(self, params_str):
        """
        Safely parse model parameters from string format.

        Args:
            params_str: String in format "key=value,key=value"

        Returns:
            dict: Parsed parameters

        Raises:
            FieldValidationError: If parameter parsing fails
        """
        if not params_str:
            return {}

        model_params = {}
        try:
            for param in params_str.split(","):
                key, value = param.split("=")
                key = key.strip()
                value = value.strip()
                try:
                    # Try to convert to number if possible
                    model_params[key] = float(value) if "." in value else int(value)
                except ValueError:
                    model_params[key] = value
            return model_params
        except Exception as e:
            raise FieldValidationError(f"Invalid model parameters format: {e}")

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

        # Disable filtering checkbox
        self.disable_filtering_var = tk.BooleanVar(value=False)
        disable_filtering_checkbox = ttk.Checkbutton(
            settings_frame,
            text="Disable boundary condition filtering",
            variable=self.disable_filtering_var
        )
        disable_filtering_checkbox.pack(anchor="w", padx=5, pady=5)

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
        """Create the dataset filtering tab"""
        self.filter_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.filter_frame, text="Dataset Filtering")

        # Create main container with scrollable content
        main_container = ttk.Frame(self.filter_frame)
        main_container.pack(fill="both", expand=True, padx=5, pady=5)

        # SETTINGS SECTION (combines Model File, Dataset Folder, and Filter Settings)
        settings_frame = ttk.LabelFrame(main_container, text="Settings")
        settings_frame.pack(fill="x", padx=5, pady=5)

        # Model File
        model_entry_frame = ttk.Frame(settings_frame)
        model_entry_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(model_entry_frame, text="Model File:").pack(side="left")
        self.filter_model_var = tk.StringVar(value="model/model.pkl")
        self.filter_model_entry = ttk.Entry(model_entry_frame, textvariable=self.filter_model_var, width=40)
        self.filter_model_entry.pack(side="left", padx=(10, 5), fill="x", expand=True)

        ttk.Button(model_entry_frame, text="Browse", command=self.browse_filter_model_file).pack(side="left")

        # Dataset Folder
        dataset_entry_frame = ttk.Frame(settings_frame)
        dataset_entry_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(dataset_entry_frame, text="Dataset Folder:").pack(side="left")
        self.filter_dataset_var = tk.StringVar(value="dataset_train")
        self.filter_dataset_entry = ttk.Entry(dataset_entry_frame, textvariable=self.filter_dataset_var, width=40)
        self.filter_dataset_entry.pack(side="left", padx=(10, 5), fill="x", expand=True)

        ttk.Button(dataset_entry_frame, text="Browse", command=self.browse_filter_dataset_folder).pack(side="left")

        # Class filtering settings
        class_filter_frame = ttk.Frame(settings_frame)
        class_filter_frame.pack(fill="x", padx=5, pady=5)
        self.class_mismatch_var = tk.BooleanVar(value=True)
        self.class_mismatch_checkbox = ttk.Checkbutton(
            class_filter_frame,
            text="Class mismatch",
            variable=self.class_mismatch_var,
            command=self.on_class_mismatch_changed
        )
        self.class_mismatch_checkbox.pack(side="left", padx=5, pady=5)

        ttk.Label(class_filter_frame, text="Minimal confidence level (%):").pack(side="left")
        self.confidence_var = tk.DoubleVar(value=80.0)
        self.confidence_scale = ttk.Scale(
            class_filter_frame,
            from_=0.0,
            to=100.0,
            orient="horizontal",
            variable=self.confidence_var,
            length=200
        )
        self.confidence_scale.state(["disabled"])
        self.confidence_scale.pack(side="left", padx=(10, 5))

        self.confidence_label = ttk.Label(class_filter_frame, text="80.0%")
        self.confidence_label.pack(side="left", padx=5)

        # Update label when slider changes
        self.confidence_var.trace("w", self.update_confidence_label)

        # START FILTERING BUTTON
        start_button_frame = ttk.Frame(main_container)
        start_button_frame.pack(fill="x", padx=5, pady=10)

        ttk.Button(start_button_frame, text="Start Filtering", command=self.start_filtering).pack(anchor="center")

        # Info frame for actual and predicted class
        info_frame = ttk.Frame(main_container)
        info_frame.pack(fill="x", padx=5, pady=5)

        self.actual_class_label = ttk.Label(info_frame, text="Actual Class: N/A", font=("Arial", 12))
        self.actual_class_label.pack(side="left", padx=(0, 20))

        self.predicted_class_label = ttk.Label(info_frame, text="Predicted Class: N/A", font=("Arial", 12))
        self.predicted_class_label.pack(side="left")

        # CONTROL BUTTONS (always visible below Circuit Analysis section)
        control_frame = ttk.Frame(main_container)
        control_frame.pack(fill="x", padx=5, pady=10)

        self.delete_button = ttk.Button(control_frame, text="Delete", command=self.delete_current_file)
        self.delete_button.pack(side="left", padx=(0, 10))
        self.next_button = ttk.Button(control_frame, text="Next", command=self.next_file)
        self.next_button.pack(side="left")

        # Progress info
        self.progress_label = ttk.Label(control_frame, text="Progress: N/A")
        self.progress_label.pack(side="right")

        # GRAPHICAL REPRESENTATION SECTION
        graph_frame = ttk.LabelFrame(main_container, text="Circuit Analysis")
        graph_frame.pack(fill="x", expand=True, padx=5, pady=5)

        # Create matplotlib figure and canvas
        self.filter_figure = Figure(figsize=(10, 6), dpi=80)
        self.filter_canvas = FigureCanvasTkAgg(self.filter_figure, master=graph_frame)
        self.filter_canvas.get_tk_widget().pack(fill="x", expand=True, padx=5, pady=5)

        # Clear the initial plot (after all widgets are created)
        self.clear_plot()

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
                    # Parse as list (single values become single-element lists)
                    values = [float(v.strip()) for v in value_str.split(",") if v.strip()]
                    elements[element_type][param_index]["nominal"] = {
                        "type": "list",
                        "value": values
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

            # Add disable-filtering flag if enabled
            if self.disable_filtering_var.get():
                cmd.append("--disable-filtering")

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

        # Run training in a separate thread to avoid blocking the GUI
        def training_thread():
            try:
                # Get and validate all training fields
                train_dataset = self._safe_get_field(self.train_dataset_var, "training dataset directory")
                model_file = self._safe_get_field(self.train_model_file_var, "output model file path")
                model_params_str = self._safe_get_field(self.model_params_var, "model parameters", allow_empty=True)
                model_params = self._safe_parse_model_params(model_params_str)

                self.log_train_results("Starting model training...")
                self.log_train_results(f"Dataset: {train_dataset}")
                self.log_train_results(f"Output model: {model_file}")
                if model_params:
                    self.log_train_results(f"Parameters: {model_params}")

                # Change to project root directory
                original_cwd = os.getcwd()
                os.chdir(self.project_root)

                try:
                    # Call the training API directly
                    dataset_path = Path(train_dataset)
                    if not dataset_path.is_absolute():
                        dataset_path = self.project_root / dataset_path

                    self.log_train_results("Training classifier...")
                    classifier = circuit_detector.train_classifier(dataset_path, model_params if model_params else None)

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

            except FieldValidationError as e:
                messagebox.showerror("Error", str(e))
                return

            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                self.log_train_results(error_msg)
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
                val_dataset = self._safe_get_field(self.val_dataset_var, "validation dataset directory")
                model_file = self._safe_get_field(self.val_model_file_var, "model file path")

                self.log_train_results("Starting model validation...")
                self.log_train_results(f"Model: {model_file}")
                self.log_train_results(f"Validation dataset: {val_dataset}")

                # Change to project root directory
                original_cwd = os.getcwd()
                os.chdir(self.project_root)

                try:
                    # Load the model
                    model_path = Path(model_file)
                    if not model_path.is_absolute():
                        model_path = self.project_root / model_path

                    self.log_train_results("Loading model...")
                    classifier = circuit_detector.CircuitClassifier.load(model_path)

                    # Evaluate the model
                    dataset_path = Path(val_dataset)
                    if not dataset_path.is_absolute():
                        dataset_path = self.project_root / dataset_path

                    self.log_train_results("Evaluating model...")
                    results = classifier.evaluate(dataset_path)

                    # Display results using unified function
                    circuit_detector.CircuitClassifier.display_evaluation_results(results, self.log_train_results)
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

            except FieldValidationError as e:
                messagebox.showerror("Error", str(e))
                return

            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                self.log_train_results(error_msg)
                self.root.after(0, lambda: messagebox.showerror("Error", error_msg))

        # Start validation in background thread
        thread = threading.Thread(target=validation_thread, daemon=True)
        thread.start()

    def browse_filter_model_file(self):
        """Browse for filter model file path (existing file)"""
        file_path = filedialog.askopenfilename(
            initialdir=self.project_root,
            title="Select Model File for Filtering",
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if file_path:
            try:
                rel_path = Path(file_path).relative_to(self.project_root)
                self.filter_model_var.set(str(rel_path))
            except ValueError:
                self.filter_model_var.set(file_path)

    def browse_filter_dataset_folder(self):
        """Browse for filter dataset directory"""
        directory = filedialog.askdirectory(
            initialdir=self.project_root,
            title="Select Dataset Directory for Filtering"
        )
        if directory:
            try:
                rel_path = Path(directory).relative_to(self.project_root)
                self.filter_dataset_var.set(str(rel_path))
            except ValueError:
                self.filter_dataset_var.set(directory)

    def on_class_mismatch_changed(self):
        """Handle class mismatch checkbox state change"""
        if self.class_mismatch_var.get():
            self.confidence_scale.state(["disabled"])
        else:
            self.confidence_scale.state(["!disabled"])

    def update_confidence_label(self):
        """Update confidence level label when slider changes"""
        self.confidence_label.config(text=f"{self.confidence_var.get():.1f}%")

    def start_filtering(self):
        """Initialize filtering process"""

        # Get and validate all filtering fields and load the classifier
        try:
            model_file = self._safe_get_field(self.filter_model_var, "model file path")
            dataset_folder = self._safe_get_field(self.filter_dataset_var, "dataset folder path")

            model_path = Path(model_file)
            if not model_path.is_absolute():
                model_path = self.project_root / model_path

            self.current_classifier = circuit_detector.CircuitClassifier.load(model_path)
        except FieldValidationError as e:
            messagebox.showerror("Error", str(e))
            return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            return

        # Find all UZF files in the dataset
        dataset_path = Path(dataset_folder)
        if not dataset_path.is_absolute():
            dataset_path = self.project_root / dataset_path

        if not dataset_path.exists():
            messagebox.showerror("Error", f"Dataset folder does not exist: {dataset_path}")
            return

        # Find UZF files recursively
        self.uzf_files = list(dataset_path.rglob("*.uzf"))

        if not self.uzf_files:
            messagebox.showwarning("Warning", f"No UZF files found in {dataset_path}")
            return

        self.current_file_index = 0
        self.process_next_file()

    def process_next_file(self):
        """Process the next file in the list"""
        while self.current_file_index < len(self.uzf_files):
            current_file = self.uzf_files[self.current_file_index]

            try:
                # Load UZF file
                measurement = load_board_from_ufiv(str(current_file))
                comment = measurement.elements[0].pins[0].comment
                measurement_obj = measurement.elements[0].pins[0].measurements[0]
                iv_curve = measurement_obj.ivc

                voltages = np.array(iv_curve.voltages)
                currents = np.array(iv_curve.currents)

                # Predict using the classifier
                from circuit_detector.features import CircuitFeatures
                features = CircuitFeatures(comment, measurement_obj.settings, voltages, currents)

                # Extract actual class from CircuitFeatures class_name
                actual_class = features.class_name if features.class_name else "Unknown"

                # Get prediction probabilities
                probabilities = self.current_classifier.predict_proba(features)
                predicted_class_idx = np.argmax(probabilities)
                predicted_class = self.current_classifier.classes_[predicted_class_idx]
                confidence = probabilities[predicted_class_idx] * 100

                # Check if file should be skipped based on filter criteria
                should_skip = False

                if self.class_mismatch_var.get():
                    # Class mismatch mode: skip correctly recognized classes
                    if actual_class == predicted_class:
                        should_skip = True
                else:
                    # Confidence threshold mode: skip classes above threshold
                    if confidence >= self.confidence_var.get():
                        should_skip = True

                if should_skip:
                    self.current_file_index += 1
                    continue

                # Display this file
                self.display_current_file(voltages,
                                          currents,
                                          actual_class,
                                          predicted_class,
                                          confidence,
                                          measurement_obj.settings
                                          )
                return

            except Exception as e:
                print(f"Error processing {current_file}: {e}")
                self.current_file_index += 1
                continue

        # All files processed
        messagebox.showinfo("Complete", "All files have been processed!")
        self.clear_plot()

    def display_current_file(self, voltages, currents, actual_class, predicted_class, confidence, measurement_settings):
        """Display the current file's I-V curve and information"""
        # Clear previous plot
        self.filter_figure.clear()

        # Create subplot
        ax = self.filter_figure.add_subplot(111)
        ax.grid(True)
        ax.plot(voltages, currents, "b-", linewidth=2)
        ax.set_xlabel("Voltage [V]")
        ax.set_ylabel("Current [A]")
        ax.set_title("IV-Characteristic")

        # Set fixed axis ranges based on measurement parameters (same as save_plot function)
        max_voltage = measurement_settings.max_voltage
        internal_resistance = measurement_settings.internal_resistance

        # X-axis: 20% wider than [-max_voltage, max_voltage]
        x_range = max_voltage * 1.2
        ax.set_xlim(-x_range, x_range)

        # Y-axis: max_voltage / internal_resistance * 1.2, centered at 0
        y_max = (max_voltage / internal_resistance) * 1.2
        ax.set_ylim(-y_max, y_max)

        # Update canvas
        self.filter_canvas.draw()

        # Update info labels
        self.actual_class_label.config(text=f"Actual Class: {actual_class}")
        self.predicted_class_label.config(text=f"Predicted Class: {predicted_class} ({confidence:.1f}%)")

        # Update progress
        self.progress_label.config(text=f"Progress: {self.current_file_index + 1}/{len(self.uzf_files)}")

    def clear_plot(self):
        """Clear the plot and info labels"""
        if self.filter_figure:
            self.filter_figure.clear()
            ax = self.filter_figure.add_subplot(111)
            ax.text(0.5, 0.5, "No data to display", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            self.filter_canvas.draw()

        self.actual_class_label.config(text="Actual Class: N/A")
        self.predicted_class_label.config(text="Predicted Class: N/A")
        self.progress_label.config(text="Progress: N/A")

    def delete_current_file(self):
        """Delete current UZF file and corresponding PNG file if it exists"""
        if self.current_file_index >= len(self.uzf_files):
            return

        current_file = self.uzf_files[self.current_file_index]

        try:
            # Delete UZF file
            current_file.unlink()

            # Try to delete corresponding PNG file
            png_file = current_file.with_suffix(".png")
            if png_file.exists():
                png_file.unlink()

            print(f"Deleted: {current_file}")
            if png_file.exists():
                print(f"Deleted: {png_file}")

            # Remove from list and process next
            self.uzf_files.pop(self.current_file_index)

            # Don't increment index since we removed an item
            self.process_next_file()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete file: {e}")

    def next_file(self):
        """Skip to next file without deleting"""
        self.current_file_index += 1
        self.process_next_file()

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
