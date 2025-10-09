"""
Base class for GUI tabs in the dataset tool.
"""

from tkinter import ttk, filedialog
from pathlib import Path
from tools.gui_utils import get_relative_path


class BaseTab:
    """Base class for all GUI tabs providing common functionality."""

    def __init__(self, parent_notebook, tab_name, root, project_root, log_callback):
        """
        Initialize base tab.

        Args:
            parent_notebook: Parent ttk.Notebook widget
            tab_name: Name to display on the tab
            root: Root tkinter window
            project_root: Path to project root directory
            log_callback: Callback function for logging messages (tab-specific)
        """
        self.notebook = parent_notebook
        self.root = root
        self.project_root = project_root
        self.log_callback = log_callback

        # Create the tab frame
        self.frame = ttk.Frame(parent_notebook)
        parent_notebook.add(self.frame, text=tab_name)

    def log(self, message):
        """
        Log a message using the tab's log callback.

        Args:
            message: Message to log
        """
        if self.log_callback:
            self.log_callback(message)

    def browse_file(self, title, var, filetypes, mode="open"):
        """
        Browse for a file and update a StringVar.

        Args:
            title: Dialog title
            var: tkinter StringVar to update
            filetypes: List of (description, pattern) tuples
            mode: "open" or "save"
        """
        if mode == "open":
            file_path = filedialog.askopenfilename(
                initialdir=self.project_root,
                title=title,
                filetypes=filetypes
            )
        else:  # save
            file_path = filedialog.asksaveasfilename(
                initialdir=self.project_root,
                title=title,
                filetypes=filetypes
            )

        if file_path:
            rel_path = get_relative_path(file_path, self.project_root)
            var.set(rel_path)

    def browse_directory(self, title, var):
        """
        Browse for a directory and update a StringVar.

        Args:
            title: Dialog title
            var: tkinter StringVar to update
        """
        directory = filedialog.askdirectory(
            initialdir=self.project_root,
            title=title
        )

        if directory:
            rel_path = get_relative_path(directory, self.project_root)
            var.set(rel_path)

    def create_tab(self):
        """
        Create the tab's UI elements.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement create_tab()")
