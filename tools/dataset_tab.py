"""
Dataset Generation Tab for the GUI tool.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import threading
import traceback
import io
from contextlib import redirect_stdout

from tools.base_tab import BaseTab

# Global reference to generate_dataset module (set by main GUI)
generate_dataset_module = None


def set_generate_dataset_module(module):
    """Set the generate_dataset module reference."""
    global generate_dataset_module
    generate_dataset_module = module


class DatasetTab(BaseTab):
    """Tab for dataset generation and parameter management."""

    def __init__(self, parent_notebook, root, project_root, log_callback):
        """Initialize the dataset generation tab."""
        super().__init__(parent_notebook, "Dataset Generation", root, project_root, log_callback)

        # Parameters file path
        self.parameters_file = self.project_root / "generate_dataset" / "parameters_variations.json"
        self.parameters_data = None
        self.param_entries = {}

        # UI elements
        self.params_canvas = None
        self.params_scrollable_frame = None
        self.dataset_dir_var = None
        self.image_var = None
        self.disable_filtering_var = None

        # Create the tab UI
        self.create_tab()

    def create_tab(self):
        """Create the dataset generation tab UI."""
        # Parameters section
        params_label = ttk.Label(self.frame, text="Component Parameters", font=("Arial", 12, "bold"))
        params_label.pack(anchor="w", padx=5, pady=(5, 0))

        # Create frame for parameters with scrollbar
        params_container = ttk.Frame(self.frame)
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

        # Settings section
        settings_frame = ttk.LabelFrame(self.frame, text="Generation Settings")
        settings_frame.pack(fill="x", padx=5, pady=5)

        # Dataset directory
        dir_frame = ttk.Frame(settings_frame)
        dir_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(dir_frame, text="Dataset Directory:").pack(side="left")
        self.dataset_dir_var = tk.StringVar(value="dataset")
        dataset_dir_entry = ttk.Entry(dir_frame, textvariable=self.dataset_dir_var, width=30)
        dataset_dir_entry.pack(side="left", padx=(10, 5))

        ttk.Button(dir_frame, text="Browse", command=self._browse_dataset_dir).pack(side="left")

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
        button_frame = ttk.Frame(self.frame)
        button_frame.pack(fill="x", padx=5, pady=10)

        ttk.Button(button_frame, text="Apply Parameters", command=self.apply_parameters).pack(side="left", padx=(0, 10))
        ttk.Button(button_frame, text="Generate Dataset", command=self.generate_dataset).pack(side="left")

    def load_parameters(self):
        """Load parameters from parameters_variations.json"""
        try:
            with open(self.parameters_file, "r", encoding="utf-8") as f:
                self.parameters_data = json.load(f)

            self.create_parameter_widgets()
            self.log("Parameters loaded successfully")

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

            self.log("Parameters saved successfully")
            messagebox.showinfo("Success", "Parameters have been applied and saved!")

        except Exception as e:
            messagebox.showerror("Error", f"Error saving parameters: {e}")

    def _browse_dataset_dir(self):
        """Browse for dataset directory"""
        self.browse_directory("Select Dataset Directory", self.dataset_dir_var)

    def generate_dataset(self):
        """Execute dataset generation using API"""

        # Run generation in a separate thread to avoid blocking the GUI
        def generation_thread():
            try:
                # Get parameters
                dataset_dir = self.dataset_dir_var.get().strip()
                save_png = self.image_var.get()
                disable_filtering = self.disable_filtering_var.get()

                self.log("Starting dataset generation...")
                self.log(f"Dataset directory: {dataset_dir if dataset_dir else 'dataset'}")
                self.log(f"Generate images: {save_png}")
                self.log(f"Disable filtering: {disable_filtering}")
                self.log("")

                # Change to project root directory
                original_cwd = os.getcwd()
                os.chdir(self.project_root)

                try:
                    # Redirect stdout to capture print statements from generate_dataset
                    output_buffer = io.StringIO()

                    with redirect_stdout(output_buffer):
                        # Call the generate_dataset API
                        generate_dataset_module.generate_dataset(
                            save_png=save_png,
                            dataset_dir=dataset_dir if dataset_dir else None,
                            disable_filtering=disable_filtering
                        )

                    # Get all output and log it
                    output = output_buffer.getvalue()
                    for line in output.split("\n"):
                        if line.strip():
                            self.log(line)

                    self.log("")
                    self.log("Dataset generation completed successfully!")

                    # Show success message in main thread
                    self.root.after(0, lambda: messagebox.showinfo("Success", "Dataset generation completed!"))

                except Exception as e:
                    error_msg = f"Dataset generation failed: {str(e)}"
                    self.log("")
                    self.log(error_msg)
                    self.log(f"Error details: {traceback.format_exc()}")
                    # Show error message in main thread
                    self.root.after(0, lambda: messagebox.showerror("Error", error_msg))

                finally:
                    os.chdir(original_cwd)

            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                self.log("")
                self.log(error_msg)
                self.root.after(0, lambda: messagebox.showerror("Error", error_msg))

        # Start generation in background thread
        thread = threading.Thread(target=generation_thread, daemon=True)
        thread.start()
