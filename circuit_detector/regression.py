"""
Circuit Parameter Detection Module

This module provides functionality for detecting circuit element parameters
from I-V curve features. Different algorithms are applied based on circuit class groups.
"""

from typing import Dict
import numpy as np
from circuit_detector.features import CircuitFeatures


# Define circuit class groups for different parameter detection algorithms
CIRCUIT_CLASS_GROUPS = {
    "simple_rc": ["R", "C", "RC"],
    "unresolvable_rc": ["R_C"],
    "not_yet": ["D", "DC", "DCR", "DR", "R_D", "nD", "nDC", "nDCR", "nDR", "R_nD"],
    "not_yet_complex": ["DC(nD_R)", "DnD_R"],
}


def detect_parameters(circuit_features: CircuitFeatures) -> Dict[str, float]:
    """
    Detect circuit element parameters from circuit features.

    This function takes a CircuitFeatures object with a non-empty class_name
    and returns a dictionary mapping element names to their estimated values.
    The detection algorithm used depends on which circuit class group the
    class_name belongs to.

    Args:
        circuit_features: CircuitFeatures object with non-empty class_name

    Returns:
        Dictionary with element names as keys and parameter values as values
        Example: {"R1": 1000.0, "C1": 1e-6}

    Raises:
        ValueError: If class_name is empty or not recognized
    """
    class_name = circuit_features.class_name

    if not class_name:
        raise ValueError("CircuitFeatures object must have a non-empty class_name")

    # Determine which group the circuit class belongs to
    group = _get_circuit_group(class_name)

    if group is None:
        raise ValueError(f"Circuit class '{class_name}' is not a member of any known circuit group")

    # Apply the appropriate algorithm based on the group
    if group == "simple_rc":
        return _detect_simple_rc_parameters(circuit_features)
    elif group == "unresolvable_rc":
        raise NotImplementedError(f"Parameter detection algorithm for group '{group}' is not yet implemented")
    elif group == "not_yet":
        raise NotImplementedError(f"Parameter detection algorithm for group '{group}' is not yet implemented")
    elif group == "not_yet_complex":
        raise NotImplementedError(f"Parameter detection algorithm for group '{group}' is not yet implemented")

    raise NotImplementedError(f"Unknown group '{group}'")


def _get_circuit_group(class_name: str) -> str:
    """
    Determine which circuit group a class belongs to.

    Args:
        class_name: Circuit class name

    Returns:
        Group name if found, None otherwise
    """
    for group_name, classes in CIRCUIT_CLASS_GROUPS.items():
        if class_name in classes:
            return group_name
    return None


def _detect_simple_rc_parameters(circuit_features: CircuitFeatures) -> Dict[str, float]:
    """
    Detect parameters for simple R, C, and RC circuits.

    This algorithm uses the FFT characteristics (amplitude and phase) of the voltage
    and current signals at the fundamental frequency to calculate the complex impedance
    of the circuit, then extracts resistance and capacitance values.

    For an RC circuit with sinusoidal excitation:
    - Complex impedance: Z = V/I (in complex form using amplitude and phase)
    - For pure R: Z = R (purely real)
    - For pure C: Z = 1/(jωC) = -j/(ωC) (purely imaginary)
    - For parallel RC: Z = R || (1/(jωC)) = R/(1 + jωRC)
      - |Z|² = R²/(1 + ω²R²C²)
      - Re(Z) = R/(1 + ω²R²C²)
      - Im(Z) = -ωR²C/(1 + ω²R²C²)

    Args:
        circuit_features: CircuitFeatures object

    Returns:
        Dictionary with element parameters (R in Ohms, C in Farads)
    """
    # Extract FFT features for the fundamental frequency (freq1)
    features_dict = circuit_features._features

    # Get voltage and current FFT amplitude and phase for fundamental frequency
    v_amplitude = features_dict["voltage_fft_freq1_amplitude"]
    v_phase = features_dict["voltage_fft_freq1_phase [rad]"]
    i_amplitude = features_dict["current_fft_freq1_amplitude"]
    i_phase = features_dict["current_fft_freq1_phase [rad]"]

    # Get measurement parameters
    probe_frequency = features_dict["probe_signal_frequency [Hz]"]
    max_voltage = features_dict["max_voltage [V]"]
    internal_resistance = features_dict["internal_resistance [Ohm]"]

    # Calculate angular frequency
    omega = 2 * np.pi * probe_frequency

    # Convert normalized FFT amplitudes back to actual values
    # (normalized by max_voltage and max_current during feature extraction)
    max_current = max_voltage / internal_resistance
    v_actual = v_amplitude * max_voltage
    i_actual = i_amplitude * max_current

    # Calculate complex voltage and current using Euler's formula: A*e^(jφ) = A*cos(φ) + j*A*sin(φ)
    v_complex = v_actual * np.exp(1j * v_phase)
    i_complex = i_actual * np.exp(1j * i_phase)

    # Calculate complex impedance: Z = V/I
    if np.abs(i_complex) < 1e-12:
        raise ValueError("Current amplitude is too small to calculate impedance reliably")

    z_complex = v_complex / i_complex

    # Extract real and imaginary parts
    z_real = np.real(z_complex)  # Resistance component
    z_imag = np.imag(z_complex)  # Reactive component

    # Initialize result dictionary
    result = {}

    class_name = circuit_features.class_name

    if class_name == "R":
        # Pure resistor: Z = R
        result["R"] = float(np.abs(z_real))

    elif class_name == "C":
        # Pure capacitor: Z = 1/(jωC), so C = 1/(ω*|Z_imag|)
        if np.abs(z_imag) < 1e-12:
            raise ValueError("Imaginary impedance is too small to calculate capacitance")
        # For capacitor, impedance is negative imaginary: -j/(ωC)
        capacitance = -1.0 / (omega * z_imag)
        result["C"] = float(np.abs(capacitance))

    elif class_name == "RC":
        # Parallel R||C circuit: Z = R/(1 + jωRC)
        # From Z_complex, we can extract R and C
        # |Z|² = R²/(1 + ω²R²C²)
        # Re(Z) = R/(1 + ω²R²C²)
        # Im(Z) = -ωR²C/(1 + ω²R²C²)

        # From the ratio: Im(Z)/Re(Z) = -ωRC
        if np.abs(z_real) < 1e-12:
            raise ValueError("Real impedance is too small to calculate parameters")

        # Calculate RC time constant from phase relationship
        rc_product = -z_imag / (omega * z_real)

        # From Re(Z) = R/(1 + ω²R²C²), and knowing ω²R²C² = (ωRC)²
        # Re(Z) * (1 + (ωRC)²) = R
        resistance = z_real * (1 + (omega * rc_product) ** 2)

        # Calculate capacitance from RC product
        if np.abs(resistance) < 1e-12:
            raise ValueError("Calculated resistance is too small")
        capacitance = rc_product / resistance

        result["R"] = float(np.abs(resistance))
        result["C"] = float(np.abs(capacitance))

    return result
