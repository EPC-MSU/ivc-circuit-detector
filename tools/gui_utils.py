"""
Shared utilities for the dataset GUI application.
These utilities are kept separate because they are stateless and can ve reused outside the base_tab class
"""

from pathlib import Path


class FieldValidationError(Exception):
    """Custom exception for field validation errors."""
    pass


def safe_get_field(field_var, field_name, allow_empty=False):
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


def safe_parse_model_params(params_str):
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


def get_relative_path(file_path, project_root):
    """
    Get relative path from project root if possible, otherwise return absolute path.

    Args:
        file_path: Path to convert
        project_root: Project root directory

    Returns:
        str: Relative path if possible, otherwise absolute path
    """
    try:
        rel_path = Path(file_path).relative_to(project_root)
        return str(rel_path)
    except ValueError:
        return file_path


def resolve_path(path_str, project_root):
    """
    Resolve a path string relative to project root if not absolute.

    Args:
        path_str: Path string to resolve
        project_root: Project root directory

    Returns:
        Path: Resolved absolute path
    """
    path = Path(path_str)
    if not path.is_absolute():
        path = project_root / path
    return path
