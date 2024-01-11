import matplotlib.pyplot as plt
import numpy as np
from epcore.elements import Board
from epcore.filemanager import save_board_to_ufiv
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PySpice.Spice.Parser import Circuit

UFIV_VERSION = "1.1.2"


class SimulatorIVC:
    def __init__(self, measurement_variant):
        self.measurement_settings = measurement_variant['measurement_settings']
        self.simulator_settings = measurement_variant['simulator_settings']
        self.snr = 40

    def get_ivc(self, circuit: Circuit):
        points = self.simulator_settings['points_per_cycle']
        period = 1 / self.measurement_settings['probe_signal_frequency']
        rms_voltage = self.measurement_settings['max_voltage'] / np.sqrt(2)
        step_time = period / points
        end_time = period * self.simulator_settings['cycles']
        skip_points = points * self.simulator_settings['skip_cycles']

        circuit.R('cs', 'input', 'input_dummy', 0)  # Rcs
        circuit.AcLine('Current', circuit.gnd, 'input_dummy',
                       rms_voltage=rms_voltage,
                       frequency=self.measurement_settings['probe_signal_frequency'])

        simulator = circuit.simulator()
        analysis = simulator.transient(step_time=step_time, end_time=end_time)

        analysis.input_dummy = analysis.input_dummy[skip_points:]
        analysis.VCurrent = analysis.VCurrent[skip_points:]
        return analysis

    def save_ivc(self, circuit, analysis, path):
        currents = list(analysis.VCurrent.as_ndarray())
        voltages = list(analysis.input_dummy.as_ndarray())
        measurement = {'measurement_settings': self.measurement_settings,
                       'comment': circuit.plot_title.replace('\n', ' '),
                       'currents': currents,
                       'voltages': voltages}

        # TODO: Fix epcore, actually PCB not saved into ufiv
        board = {'version': UFIV_VERSION,
                 "PCB": {"pcb_name": "myclass", "comment": "super_comment"},
                 'elements': [{'pins': [{'iv_curves': [measurement], 'x': 0, 'y': 0}]}]}
        epcore_board = Board.create_from_json(board)
        save_board_to_ufiv(path, epcore_board)

    def save_plot(self, circuit, analysis, path, png_path, plot_measurements_settings=True):
        fig, ax = plt.subplots(1, figsize=(8, 4))
        plt.subplots_adjust(right=0.5, bottom=0.15)
        ax.grid()
        ax.plot(analysis.input_dummy, analysis.VCurrent)
        ax.set_xlabel('Voltage [V]')
        ax.set_ylabel('Current [A]')

        arr_img = plt.imread(png_path)
        im = OffsetImage(arr_img, zoom=.5)
        ab = AnnotationBbox(im, (1, 0), xycoords='axes fraction', box_alignment=(-0.68, 0.1))
        ax.add_artist(ab)
        ax.set_title('IV-Characteristic')

        plt.figtext(0.62, 0.45, s=circuit.plot_title)
        if plot_measurements_settings:
            sett = '[Measurements settings]\n'
            for sett_name, sett_value in self.measurement_settings.items():
                sett += f'{sett_name}: {sett_value}\n'
            # sett += '[Simulator settings]\n'
            # for sett_name, sett_value in self.simulator_settings.items():
            #     sett += f'{sett_name}: {sett_value}\n'
            plt.figtext(0.62, 0.65, s=sett)

        plt.savefig(path, dpi=100)
        plt.clf()
        plt.close('all')

    def add_noise(self, analysis):
        avg_v_db = 10 * np.log10(np.mean(np.array(analysis.input_dummy, dtype=float) ** 2))
        avg_v_noise_db = avg_v_db - self.snr
        v_noise = np.random.normal(0, np.sqrt(10 ** (avg_v_noise_db / 10)), len(analysis.input_dummy))
        analysis.input_dummy = np.array(analysis.input_dummy, dtype=float) + v_noise

        avg_i_db = 10 * np.log10(np.mean(np.array(analysis.VCurrent, dtype=float) ** 2))
        avg_i_noise_db = avg_i_db - self.snr
        i_noise = np.random.normal(0, np.sqrt(10 ** (avg_i_noise_db / 10)), len(analysis.VCurrent))
        analysis.VCurrent = np.array(analysis.VCurrent, dtype=float) + i_noise
        return analysis
