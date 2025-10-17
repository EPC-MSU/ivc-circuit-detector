"""
Circuit Parameter Detection Module

This module provides functionality for detecting circuit element parameters
from I-V curve features. Different algorithms are applied based on circuit class groups.
"""

from typing import Dict
import numpy as np
from scipy.optimize import root
from circuit_detector.features import CircuitFeatures


# Define circuit class groups for different parameter detection algorithms
CIRCUIT_CLASS_GROUPS = {
    "simple_rc": ["R", "C", "RC"],
    "unresolvable_rc": ["R_C"],
    "diodes_resistors": ["D", "nD", "DR", "nDR", "DnDR"],
    "not_yet": ["DC", "DCR", "R_D", "nDC", "nDCR", "R_nD"],
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

    if group == "":
        raise ValueError(f"Circuit class '{class_name}' is not a member of any known circuit group")

    # Apply the appropriate algorithm based on the group
    if group == "simple_rc":
        return _detect_simple_rc_parameters(circuit_features)
    elif group == "diodes_resistors":
        return _detect_diodes_resistors_parameters(circuit_features)
    elif group == "unresolvable_rc":
        raise NotImplementedError(
            f"Parameter detection algorithm for group '{group}' seems to be impossible to implement."
        )
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
        Group name if found, empty string otherwise
    """
    for group_name, classes in CIRCUIT_CLASS_GROUPS.items():
        if class_name in classes:
            return group_name
    return ""


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
        result["R"] = float(z_real)

    elif class_name == "C":
        # Pure capacitor: Z = 1/(jωC), so C = 1/(ω*|Z_imag|)
        if np.abs(z_imag) < 1e-12:
            raise ValueError("Imaginary impedance is too small to calculate capacitance")
        # For capacitor, impedance is negative imaginary: -j/(ωC)
        capacitance = -1.0 / (omega * z_imag)
        result["C"] = float(capacitance)

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

        result["R"] = float(resistance)
        result["C"] = float(capacitance)

    return result


def _sigmoid(x: float) -> float:
    """
    Numerically stable sigmoid function.

    Args:
        x: Input value

    Returns:
        Sigmoid output in range (0, 1)
    """
    if x > 0:
        return 1 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (np.exp(x) + 1)


def _f(y: float, r: float) -> tuple:
    """
    Calculate harmonic components f_0, f_1, f_2, f_3 for normalized diode threshold.

    Args:
        y: Normalized diode threshold voltage (must be in [0, 1])
        r: Ratio R/(R+R_int)

    Returns:
        Tuple of (f_0, f_1, f_2, f_3) harmonic components
    """
    if y < 0 or y > 1:
        raise ValueError("y must be in range [0, 1]")

    sqrt_term = np.sqrt(1 - y ** 2)
    arccos_term = np.arccos(y)
    sqrt_cubed = sqrt_term ** 3

    f0 = (r / np.pi) * (sqrt_term - y * arccos_term)
    f1 = (r / (2 * np.pi)) * (y * sqrt_term - arccos_term)
    f2 = -(r / (3 * np.pi)) * sqrt_cubed
    f3 = (r / (3 * np.pi)) * y * sqrt_cubed

    return f0, f1, f2, f3


def _calculate_harmonics(r: float, y: float, z: float) -> np.ndarray:
    """
    Calculate theoretical Fourier harmonics (0, 1, 2, 3) for diode-resistor circuit.

    The harmonics are returned in dimensionless form (normalized by V_amplitude/R_int).
    This allows them to be directly compared with normalized FFT components.

    Args:
        r: Ratio R/(R+R_int)
        y: Normalized forward diode threshold (V_threshold_forward / (V_amplitude * r))
        z: Normalized reverse diode threshold (V_threshold_reverse / (V_amplitude * r))

    Returns:
        Numpy array of 4 harmonics [h0, h1, h2, h3] in dimensionless form
    """
    fy0, fy1, fy2, fy3 = _f(y, r)
    fz0, fz1, fz2, fz3 = _f(z, r)

    h0 = fy0 - fz0
    h1 = fy1 + fz1 - (1 - r) / 2
    h2 = fy2 - fz2
    h3 = fy3 + fz3

    return np.array([h0, h1, h2, h3])


def _extract_harmonics(circuit_features: CircuitFeatures) -> list:
    """
    Calculate harmonics (1, 2, 3) directly from current data via FFT with phase correction.

    Computes FFT of normalized current from raw measurement data, then extracts the
    harmonics with phase correction. The measured phases may be shifted by an unknown
    phase offset due to the measurement setup, so phase differences are calculated
    relative to the first harmonic phase to determine the common phase shift.

    Args:
        circuit_features: CircuitFeatures object containing current data and measurement parameters

    Returns:
        List of three harmonic values [h1_imag, h2_real, h3_imag] where:
        - h1_imag: imaginary part of 1st harmonic (sine component)
        - h2_real: real part of 2nd harmonic (cosine component)
        - h3_imag: imaginary part of 3rd harmonic (sine component)
    """
    try:
        # Compute FFT of not normalized current
        fft_current = np.fft.fft(circuit_features.currents) / len(circuit_features.currents)

        # Extract components for harmonics 1, 2, 3
        fft_h0 = fft_current[0]
        fft_h1 = fft_current[1]
        fft_h2 = fft_current[2]
        fft_h3 = fft_current[3]

        # Get phases of each harmonic
        h1_phase = np.angle(fft_h1)
        h2_phase = np.angle(fft_h2)
        h3_phase = np.angle(fft_h3)

        # Calculate phase offset relative to h1 phase
        # This accounts for unknown phase shift in the measurement system
        phase_offset = h1_phase + np.pi / 2

        # Apply phase correction by subtracting the offset which grows for every harmonic (see FFT formulas)
        corrected_h1_phase = h1_phase - phase_offset
        corrected_h2_phase = h2_phase - 2 * phase_offset
        corrected_h3_phase = h3_phase - 3 * phase_offset

        # Extract real and imaginary components with corrected phases
        h1_amplitude = np.abs(fft_h1)
        h2_amplitude = np.abs(fft_h2)
        h3_amplitude = np.abs(fft_h3)

        h0_real = np.real(fft_h0)
        h1_imag = h1_amplitude * np.sin(corrected_h1_phase)
        h2_real = h2_amplitude * np.cos(corrected_h2_phase)
        h3_imag = h3_amplitude * np.sin(corrected_h3_phase)

    except (KeyError, AttributeError) as e:
        raise ValueError(f"CircuitFeatures missing required data for FFT calculation: {e}")

    return [h0_real, h1_imag, h2_real, h3_imag]


def _detect_diodes_resistors_parameters(circuit_features: CircuitFeatures) -> Dict[str, float]:
    """
    Detect parameters for diode-resistor circuits (D, nD, DR, nDR, DnDR).

    This algorithm uses the first 4 Fourier harmonics (0, 1, 2, 3) of the current signal
    to determine the series resistance (R), forward diode threshold voltage, and reverse
    diode threshold voltage (if present).

    The algorithm solves a system of 3 nonlinear equations using scipy.optimize.root:
    - Uses harmonics 1, 2, and 3 as independent equations
    - Parameterizes r, y, z using sigmoid functions to constrain solutions to valid ranges
    - Returns calculated R and diode threshold voltages

    Args:
        circuit_features: CircuitFeatures object with current FFT harmonics and circuit parameters

    Returns:
        Dictionary with element parameters:
        - "R": Series resistance in Ohms
        - "V_threshold_forward": Forward diode threshold in Volts (if detectable)
        - "V_threshold_reverse": Reverse diode threshold in Volts (if detectable)

    Raises:
        ValueError: If required features are missing or solution fails to converge
    """
    features_dict = circuit_features._features

    # Extract measurement parameters
    try:
        voltage_amplitude = features_dict["max_voltage [V]"]
        r_int = features_dict["internal_resistance [Ohm]"]
    except KeyError:
        raise ValueError("CircuitFeatures missing voltage_amplitude or internal_resistance")

    # Add three other harmonics
    h_experimental = _extract_harmonics(circuit_features)

    # Define system of nonlinear equations
    def _equations(params: np.ndarray, h_exp: np.ndarray) -> list:
        """
        System of 3 equations for solving r, y, z parameters.

        Uses harmonics 1, 2, 3 as independent equations.
        """
        proto_r, proto_y, proto_z = params

        # Convert sigmoid parameters to physical parameters
        r = _sigmoid(proto_r)
        y = _sigmoid(proto_y)
        z = _sigmoid(proto_z)

        # Calculate theoretical harmonics (in dimensionless form)
        h_theory = _calculate_harmonics(r, y, z) * voltage_amplitude / r_int

        # Return residuals for harmonics 1, 2, 3
        return [h_theory[1] - h_exp[1], h_theory[2] - h_exp[2], h_theory[3] - h_exp[3]]

    # Initial guess (sigmoid parameters): [0, 0, 0] maps to [0.5, 0.5, 0.5]
    initial_guess = np.array([0.0, 0.0, 0.0])

    # Solve the nonlinear system
    try:
        solution = root(_equations, initial_guess, args=(h_experimental,), tol=1e-6)

        # if not solution.success:
        #    raise ValueError(f"Solver failed to converge: {solution.message}")

        proto_r_solved, proto_y_solved, proto_z_solved = solution.x
    except Exception as e:
        raise ValueError(f"Failed to solve nonlinear system: {str(e)}")

    # Convert solution back to physical parameters
    r_solution = _sigmoid(proto_r_solved)
    y_solution = _sigmoid(proto_y_solved)
    z_solution = _sigmoid(proto_z_solved)

    # Initialize result dictionary
    result = {}

    # Calculate R from r ratio: r = R / (R + R_int)
    # Solving: R = r * (R + R_int) => R = r * R + r * R_int => R(1 - r) = r * R_int => R = r * R_int / (1 - r)
    if r_solution < 0.999:
        r_calculated = (r_solution / (1 - r_solution)) * r_int
        result["R"] = float(r_calculated)
    else:
        result["R"] = float(np.inf)

    # Calculate V_threshold_forward from y: y = V_threshold_forward / (V_amplitude * r)
    if y_solution < 0.999:
        v_threshold_forward = y_solution * r_solution * voltage_amplitude
        result["Df"] = float(v_threshold_forward)
    else:
        # Forward diode disconnected
        result["Df"] = float(np.inf)

    # Calculate V_threshold_reverse from z: z = V_threshold_reverse / (V_amplitude * r)
    if z_solution < 0.999:
        v_threshold_reverse = z_solution * r_solution * voltage_amplitude
        result["Dr"] = float(v_threshold_reverse)
    else:
        # Reverse diode disconnected
        result["Dr"] = float(np.inf)

    return result
