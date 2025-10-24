import matplotlib.pyplot as plt
import numpy as np
from epcore.elements import Board
from epcore.filemanager import save_board_to_ufiv
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PySpice.Spice.Parser import Circuit
import matplotlib
import random
import re


# The matplotlib backend that can be used with threading
matplotlib.use("agg")


class SimulatorIVC:
    def __init__(self, measurement_variant):
        self.measurement_settings = measurement_variant["measurement_settings"]
        self.noise_settings = measurement_variant["noise_settings"]

    @staticmethod
    def _parse_spice_value(value_str):
        """Parse SPICE value string with units (e.g., '1u', '100n', '4.7k')

        Returns value in base SI units (Farads for capacitance, Ohms for resistance)
        """
        if isinstance(value_str, (int, float)):
            return float(value_str)

        value_str = str(value_str).strip()

        # SPICE unit multipliers
        units = {
            "f": 1e-15,  # femto
            "p": 1e-12,  # pico
            "n": 1e-9,   # nano
            "u": 1e-6,   # micro
            "m": 1e-3,   # milli
            "k": 1e3,    # kilo
            "meg": 1e6,  # mega
            "g": 1e9,    # giga
        }

        # Try to match number + optional unit
        match = re.match(r"([+-]?(?:\d+\.?\d*|\.\d+))\s*([a-zA-Z]*)", value_str)

        if not match:
            raise ValueError(f"Cannot parse value: {value_str}")

        number_str, unit_str = match.groups()
        number = float(number_str)
        unit_str = unit_str.lower()

        if unit_str in units:
            return number * units[unit_str]
        elif unit_str == "":
            return number
        else:
            raise ValueError(f"Unknown unit: {unit_str}")

    @staticmethod
    def _get_circuit_rc_time_constant(circuit: Circuit, internal_resistance):
        """Calculate RC time constant for the circuit

        Extracts the single capacitance from the circuit's plot_title and calculates
        the RC time constant based on internal measurement resistance.
        Circuits must have 0 or 1 capacitor. Multiple capacitors are not supported.
        plot_title format: "Class: [ClassName]\nC1(1u)\nR1(1k)\n..."

        Args:
            circuit: PySpice Circuit object with plot_title attribute
            internal_resistance: Measurement internal resistance in Ohms

        Returns:
            RC time constant in seconds. Returns 0 if no capacitors found.

        Raises:
            ValueError: If more than one capacitor is found in the circuit
        """
        try:
            plot_title = circuit.plot_title
            # Find all capacitor entries: match lines like "C1(1u)" or "C2(100n)"
            # Pattern: C followed by number(s), then parentheses with value
            capacitor_pattern = r"C\d+\(([^)]+)\)"
            matches = re.findall(capacitor_pattern, plot_title)

            if len(matches) == 0:
                # No capacitors found
                return 0.0
            elif len(matches) == 1:
                # Exactly one capacitor - this is the expected case
                try:
                    capacitance_str = matches[0].strip()
                    capacitance_f = SimulatorIVC._parse_spice_value(capacitance_str)
                except (ValueError, AttributeError) as e:
                    raise ValueError(f"Failed to parse capacitance value '{capacitance_str}': {e}")
            else:
                # More than one capacitor - not supported
                raise ValueError(
                    f"Circuit has {len(matches)} capacitors: {matches}. "
                    "Multiple capacitors are not supported in this approach."
                )

        except (AttributeError, TypeError) as e:
            # If plot_title doesn't exist or can't be parsed, return 0
            return 0.0

        # RC time constant = R * C
        rc_time_constant = internal_resistance * capacitance_f
        return rc_time_constant

    def get_ivc(self, circuit: Circuit):
        rms_voltage = self.measurement_settings["max_voltage"] / np.sqrt(2)

        # ssr = simulator sampling rate
        ssr = self.measurement_settings["sampling_rate"]

        internal_resistance = self.measurement_settings["internal_resistance"]
        circuit.R("cs", "input", "input_dummy", internal_resistance)
        # circuit.C("coaxial_probes", circuit.gnd, "input_dummy", 204*10**-12)  # 28*10**-12
        circuit.AcLine("Current", circuit.gnd, "input_dummy", rms_voltage=rms_voltage,
                       frequency=self.measurement_settings["probe_signal_frequency"])

        period = 1 / self.measurement_settings["probe_signal_frequency"]

        # Calculate RC time constant for the circuit
        rc_time_constant = self._get_circuit_rc_time_constant(circuit, internal_resistance)

        # Scale precharge delay by RC time constant
        # precharge_delay_RC_times setting is in units of RC time constants
        # If no capacitors, RC = 0, so precharge = 0 regardless of setting
        precharge_delay_seconds = self.measurement_settings["precharge_delay_RC_times"] * rc_time_constant

        # Add random phase offset to precharge delay for phase randomization
        random_phase_offset = random.uniform(0, 1) * period
        total_precharge_delay = precharge_delay_seconds + random_phase_offset

        step_time = period / (ssr / self.measurement_settings["probe_signal_frequency"])
        end_time = total_precharge_delay + period

        simulator = circuit.simulator()
        analysis = simulator.transient(step_time=step_time, end_time=end_time)

        need_points = int(ssr / self.measurement_settings["probe_signal_frequency"]) - 1

        voltages = analysis.input[-need_points:].as_ndarray()
        currents = analysis.VCurrent[-need_points:].as_ndarray()

        voltages = np.append(voltages, voltages[0])  # Close points circle
        currents = np.append(currents, currents[0])  # Close points circle

        expected_points = int(
            self.measurement_settings["sampling_rate"] / self.measurement_settings["probe_signal_frequency"]
        )
        assert len(voltages) == expected_points, f"Expected {expected_points} voltage points, got {len(voltages)}"
        assert len(currents) == expected_points, f"Expected {expected_points} current points, got {len(currents)}"
        return voltages, currents

    def save_ivc(self, title, analysis, path):
        voltages, currents = analysis
        measurement = {"measurement_settings": self.measurement_settings,
                       "currents": list(currents),
                       "voltages": list(voltages),
                       "is_reference": True}

        board = {"version": "1.1.2",
                 "elements": [{"pins": [{"iv_curves": [measurement],
                                         "x": 0,
                                         "y": 0,
                                         "comment": title.replace("\n", " ")
                                         }]}]}
        epcore_board = Board.create_from_json(board)
        save_board_to_ufiv(path, epcore_board)

    def save_plot(self, title, analysis, path, scheme_png_path, save_png=False):
        if not save_png:
            return
        fig, ax = plt.subplots(1, figsize=(8, 4))
        plt.subplots_adjust(right=0.5, bottom=0.15)
        ax.grid()
        voltages, currents = analysis
        ax.plot(voltages, currents)
        ax.set_xlabel("Voltage [V]")
        ax.set_ylabel("Current [A]")

        # Set fixed axis ranges based on measurement parameters
        max_voltage = self.measurement_settings["max_voltage"]
        internal_resistance = self.measurement_settings["internal_resistance"]

        # X-axis: 20% wider than [-max_voltage, max_voltage]
        x_range = max_voltage * 1.2
        ax.set_xlim(-x_range, x_range)

        # Y-axis: max_voltage / internal_resistance * 1.2, centered at 0
        y_max = (max_voltage / internal_resistance) * 1.2
        ax.set_ylim(-y_max, y_max)

        arr_img = plt.imread(scheme_png_path)
        im = OffsetImage(arr_img, zoom=.5)
        ab = AnnotationBbox(im, (1, 0), xycoords="axes fraction", box_alignment=(-0.68, 0.1))
        ax.add_artist(ab)
        ax.set_title("IV-Characteristic")

        plt.figtext(0.62, 0.45, s=title)
        sett = "[Measurements settings]\n"
        for sett_name, sett_value in self.measurement_settings.items():
            sett += f"{sett_name}: {sett_value}\n"
        # sett += "[Simulator settings]\n"
        # for sett_name, sett_value in self.simulator_settings.items():
        #     sett += f"{sett_name}: {sett_value}\n"
        plt.figtext(0.62, 0.65, s=sett)

        plt.savefig(path, dpi=100)
        plt.clf()
        plt.close("all")

    def compare_ivc(self, ivc1, ivc2):
        """
        Compare two I-V curves and return a difference value from 0 to 1.
        Uses EPCore's IVCComparator for accurate I-V curve comparison.

        :param ivc1: First I-V curve (voltages, currents) tuple
        :param ivc2: Second I-V curve (voltages, currents) tuple
        :return: Difference value from 0 to 1 (0 means identical, 1 means completely different)
        """
        from epcore.measurementmanager import IVCComparator
        from epcore.elements.measurement import IVCurve

        # Extract voltages and currents from tuples
        voltages1, currents1 = ivc1
        voltages2, currents2 = ivc2

        # Convert to lists if they're numpy arrays
        if hasattr(voltages1, "tolist"):
            voltages1 = voltages1.tolist()
            currents1 = currents1.tolist()
        if hasattr(voltages2, "tolist"):
            voltages2 = voltages2.tolist()
            currents2 = currents2.tolist()

        # Create IVCurve objects
        curve1 = IVCurve(currents=currents1, voltages=voltages1)
        curve2 = IVCurve(currents=currents2, voltages=voltages2)

        # Create comparator and configure it
        comparator = IVCComparator()

        # Get noise settings from measurement variant
        # horizontal_noise = voltage noise, vertical_noise = current noise
        noise_settings = getattr(self, "noise_settings", {})
        min_var_voltage = noise_settings.get("horizontal_noise", 0.005)  # Default 5mV
        min_var_current = noise_settings.get("vertical_noise", 5e-6)     # Default 5µA

        comparator.set_min_ivc(min_var_voltage, min_var_current)

        # Compare the curves
        difference = comparator.compare_ivc(curve1, curve2)

        # Ensure the result is in [0, 1] range
        return max(0.0, min(1.0, difference))

    @staticmethod
    def add_noise(analysis, noise_settings):
        # We calculate noise independently for current and voltage based on RMS values and the same SNR
        voltages, currents = analysis
        # avg_v_db = 10 * np.log10(np.mean(np.array(voltages, dtype=float) ** 2))
        # avg_v_noise_db = avg_v_db - SNR
        # v_noise = np.random.normal(0, np.sqrt(10 ** (avg_v_noise_db / 10)), len(voltages))
        # voltages = np.array(voltages, dtype=float) + v_noise
        #
        # avg_i_db = 10 * np.log10(np.mean(np.array(currents, dtype=float) ** 2))
        # avg_i_noise_db = avg_i_db - SNR
        # i_noise = np.random.normal(0, np.sqrt(10 ** (avg_i_noise_db / 10)), len(currents))
        # currents = np.array(currents, dtype=float) + i_noise

        i_noise = np.random.normal(0, noise_settings["vertical_noise"], len(currents))
        v_noise = np.random.normal(0, noise_settings["horizontal_noise"], len(voltages))

        voltages = np.array(voltages, dtype=float) + v_noise
        currents = np.array(currents, dtype=float) + i_noise
        return voltages, currents
