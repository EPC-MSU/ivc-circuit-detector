"""
Dataset Filtering Tab for the GUI tool.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from epcore.filemanager.ufiv import load_board_from_ufiv

from tools.base_tab import BaseTab
from tools.gui_utils import safe_get_field, FieldValidationError, resolve_path

# Global reference to circuit_detector module (set by main GUI)
circuit_detector = None


def set_circuit_detector_module(module):
    """Set the circuit_detector module reference."""
    global circuit_detector
    circuit_detector = module


class FilterTab(BaseTab):
    """Tab for dataset filtering and cleaning."""

    def __init__(self, parent_notebook, root, project_root, log_callback):
        """Initialize the dataset filtering tab."""
        super().__init__(parent_notebook, "Dataset Filtering", root, project_root, log_callback)

        # Filtering state
        self.current_classifier = None
        self.uzf_files = []
        self.current_file_index = 0

        # UI variables
        self.filter_model_var = None
        self.filter_dataset_var = None
        self.class_mismatch_var = None
        self.confidence_var = None
        self.confidence_scale = None
        self.confidence_label = None
        self.actual_class_label = None
        self.predicted_class_label = None
        self.progress_label = None

        # Matplotlib components
        self.filter_figure = None
        self.filter_canvas = None

        # Create the tab UI
        self.create_tab()

    def create_tab(self):
        """Create the dataset filtering tab UI."""
        # Create main container
        main_container = ttk.Frame(self.frame)
        main_container.pack(fill="both", expand=True, padx=5, pady=5)

        # SETTINGS SECTION
        settings_frame = ttk.LabelFrame(main_container, text="Settings")
        settings_frame.pack(fill="x", padx=5, pady=5)

        # Model File
        model_entry_frame = ttk.Frame(settings_frame)
        model_entry_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(model_entry_frame, text="Model File:").pack(side="left")
        self.filter_model_var = tk.StringVar(value="model/model.pkl")
        filter_model_entry = ttk.Entry(model_entry_frame, textvariable=self.filter_model_var, width=40)
        filter_model_entry.pack(side="left", padx=(10, 5), fill="x", expand=True)

        ttk.Button(model_entry_frame, text="Browse", command=self._browse_filter_model_file).pack(side="left")

        # Dataset Folder
        dataset_entry_frame = ttk.Frame(settings_frame)
        dataset_entry_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(dataset_entry_frame, text="Dataset Folder:").pack(side="left")
        self.filter_dataset_var = tk.StringVar(value="dataset_train")
        filter_dataset_entry = ttk.Entry(dataset_entry_frame, textvariable=self.filter_dataset_var, width=40)
        filter_dataset_entry.pack(side="left", padx=(10, 5), fill="x", expand=True)

        ttk.Button(dataset_entry_frame, text="Browse", command=self._browse_filter_dataset_folder).pack(side="left")

        # Class filtering settings
        class_filter_frame = ttk.Frame(settings_frame)
        class_filter_frame.pack(fill="x", padx=5, pady=5)
        self.class_mismatch_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            class_filter_frame,
            text="Class mismatch",
            variable=self.class_mismatch_var,
            command=self._on_class_mismatch_changed
        ).pack(side="left", padx=5, pady=5)

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
        self.confidence_var.trace("w", self._update_confidence_label)

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

        # CONTROL BUTTONS
        control_frame = ttk.Frame(main_container)
        control_frame.pack(fill="x", padx=5, pady=10)

        ttk.Button(control_frame, text="Delete", command=self.delete_current_file).pack(side="left", padx=(0, 10))
        ttk.Button(control_frame, text="Next", command=self.next_file).pack(side="left")

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

        # Clear the initial plot
        self.clear_plot()

    def _browse_filter_model_file(self):
        """Browse for filter model file path (existing file)"""
        self.browse_file(
            "Select Model File for Filtering",
            self.filter_model_var,
            [("Pickle files", "*.pkl"), ("All files", "*.*")],
            mode="open"
        )

    def _browse_filter_dataset_folder(self):
        """Browse for filter dataset directory"""
        self.browse_directory("Select Dataset Directory for Filtering", self.filter_dataset_var)

    def _on_class_mismatch_changed(self):
        """Handle class mismatch checkbox state change"""
        if self.class_mismatch_var.get():
            self.confidence_scale.state(["disabled"])
        else:
            self.confidence_scale.state(["!disabled"])

    def _update_confidence_label(self, *args):
        """Update confidence level label when slider changes"""
        self.confidence_label.config(text=f"{self.confidence_var.get():.1f}%")

    def start_filtering(self):
        """Initialize filtering process"""
        try:
            model_file = safe_get_field(self.filter_model_var, "model file path")
            dataset_folder = safe_get_field(self.filter_dataset_var, "dataset folder path")

            model_path = resolve_path(model_file, self.project_root)
            self.current_classifier = circuit_detector.CircuitClassifier.load(model_path)
        except FieldValidationError as e:
            messagebox.showerror("Error", str(e))
            return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            return

        # Find all UZF files in the dataset
        dataset_path = resolve_path(dataset_folder, self.project_root)

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

        # Set fixed axis ranges based on measurement parameters
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
