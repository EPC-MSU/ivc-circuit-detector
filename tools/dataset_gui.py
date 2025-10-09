#!/usr/bin/env python
"""
GUI tool for dataset generation and management using tkinter.
Started from project root directory.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys
from pathlib import Path

# Import tab modules and set their module references
from tools import dataset_tab, train_tab, filter_tab, uzf_testing_tab


global circuit_detector
global generate_dataset_module


def complex_import():
    global circuit_detector
    global generate_dataset_module
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(parent_dir)
    import circuit_detector
    from generate_dataset import dataset_generator as generate_dataset_module


complex_import()

dataset_tab.set_generate_dataset_module(generate_dataset_module)
train_tab.set_circuit_detector_module(circuit_detector)
filter_tab.set_circuit_detector_module(circuit_detector)
uzf_testing_tab.set_circuit_detector_module(circuit_detector)


class DatasetGUI:
    """Main GUI application for dataset management."""

    def __init__(self, root):
        """Initialize the main GUI window."""
        self.root = root
        self.root.title("Circuit Dataset Tool")
        self.root.geometry("800x600")

        # Set project root path
        self.project_root = Path(__file__).parent.parent

        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Create text widgets for logging (one per tab that needs logging)
        self.status_text = None  # Dataset tab log
        self.train_results_text = None  # Train tab log
        self.uzf_test_results_text = None  # UZF testing tab log

        # Create tabs
        self.create_tabs()

    def create_tabs(self):
        """Create all GUI tabs."""
        # Create Dataset Generation tab
        self.dataset_tab = dataset_tab.DatasetTab(
            self.notebook,
            self.root,
            self.project_root,
            self.log_dataset_status
        )

        # Create status text widget for dataset tab
        self.status_text = tk.Text(self.dataset_tab.frame, height=8, state="disabled")
        self.status_text.pack(fill="x", padx=5, pady=5)

        status_scrollbar = ttk.Scrollbar(self.dataset_tab.frame, orient="vertical", command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=status_scrollbar.set)

        # Load parameters for dataset tab
        self.dataset_tab.load_parameters()

        # Create Train/Validation tab
        self.train_tab = train_tab.TrainTab(
            self.notebook,
            self.root,
            self.project_root,
            self.log_train_results
        )

        # Create results text widget for train tab
        results_frame = ttk.LabelFrame(self.train_tab.frame, text="Training/Validation Results")
        results_frame.pack(fill="both", expand=True, padx=5, pady=(5, 0))

        self.train_results_text = tk.Text(results_frame, height=12, state="disabled", wrap=tk.WORD)
        train_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.train_results_text.yview)
        self.train_results_text.configure(yscrollcommand=train_scrollbar.set)

        self.train_results_text.pack(side="left", fill="both", expand=True)
        train_scrollbar.pack(side="right", fill="y")

        # Create Filter tab (no additional log widget needed - uses its own display)
        self.filter_tab = filter_tab.FilterTab(
            self.notebook,
            self.root,
            self.project_root,
            None  # Filter tab doesn't use text logging
        )

        # Create UZF Testing tab
        self.uzf_testing_tab = uzf_testing_tab.UZFTestingTab(
            self.notebook,
            self.root,
            self.project_root,
            self.log_uzf_test_results
        )

        # Create results text widget for UZF testing tab
        results_frame = ttk.LabelFrame(self.uzf_testing_tab.frame, text="Recognition Results")
        results_frame.pack(fill="both", expand=True, padx=5, pady=(5, 0))

        self.uzf_test_results_text = tk.Text(results_frame, height=20, state="disabled", wrap=tk.WORD)
        uzf_test_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.uzf_test_results_text.yview)
        self.uzf_test_results_text.configure(yscrollcommand=uzf_test_scrollbar.set)

        self.uzf_test_results_text.pack(side="left", fill="both", expand=True)
        uzf_test_scrollbar.pack(side="right", fill="y")

    def log_dataset_status(self, message):
        """Add message to dataset status log"""
        self.status_text.config(state="normal")
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)
        self.status_text.config(state="disabled")

    def log_train_results(self, message):
        """Add message to training results log"""
        self.train_results_text.config(state="normal")
        self.train_results_text.insert(tk.END, f"{message}\n")
        self.train_results_text.see(tk.END)
        self.train_results_text.config(state="disabled")

    def log_uzf_test_results(self, message):
        """Add message to UZF test results log"""
        self.uzf_test_results_text.config(state="normal")
        self.uzf_test_results_text.insert(tk.END, f"{message}\n")
        self.uzf_test_results_text.see(tk.END)
        self.uzf_test_results_text.config(state="disabled")


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
