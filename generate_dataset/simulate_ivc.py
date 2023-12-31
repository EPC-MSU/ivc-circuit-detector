import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PySpice.Spice.Parser import Circuit
from epcore.elements import Board
from epcore.filemanager import save_board_to_ufiv


class SimulatorIVC:
    def __init__(self, measurement_settings_json):
        self.measurement_settings_json = measurement_settings_json
        self.probe_signal_frequency = measurement_settings_json['probe_signal_frequency']
        self.max_voltage = measurement_settings_json['max_voltage']
        self.precharge_delay = measurement_settings_json['precharge_delay']
        self.sampling_rate = measurement_settings_json['sampling_rate']
        self.internal_resistance = measurement_settings_json['internal_resistance']
        self.num_cycles = 1
        self.SNR = 40

    def get_ivc(self, circuit: Circuit):
        lendata = 100

        period = 1 / self.probe_signal_frequency
        rms_voltage = self.max_voltage / np.sqrt(2)
        circuit.R('cs', 'input', 'input_dummy', 0)  # Rcs
        circuit.AcLine('Current', circuit.gnd, 'input_dummy',
                       rms_voltage=rms_voltage,
                       frequency=self.probe_signal_frequency)
        simulator = circuit.simulator()
        analysis = simulator.transient(step_time=period / lendata,
                                       end_time=period * self.num_cycles)
        analysis.input_dummy = analysis.input_dummy[len(analysis.input_dummy) - lendata:len(analysis.input_dummy)]
        analysis.VCurrent = analysis.VCurrent[len(analysis.VCurrent) - lendata:len(analysis.VCurrent)]
        return analysis

    def save_ivc(self, circuit, analysis, path):
        currents = list(analysis.VCurrent.as_ndarray())
        voltages = list(analysis.input_dummy.as_ndarray())
        measurement = {'measurement_settings': self.measurement_settings_json,
                       'comment': circuit.plot_title.replace('\n', ' '),
                       'currents': currents,
                       'voltages': voltages}

        # TODO: Fix epcore, actually PCB not saved into ufiv
        board = {'version': "1.1.2",
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
            sett += f'probe_signal_frequency: {self.probe_signal_frequency}\n'
            sett += f'max_voltage: {self.max_voltage}\n'
            sett += f'precharge_delay: {self.precharge_delay}\n'
            sett += f'sampling_rate: {self.sampling_rate}\n'
            sett += f'internal_resistance: {self.sampling_rate}\n'
            plt.figtext(0.62, 0.65, s=sett)

        plt.savefig(path, dpi=100)
        plt.clf()
        plt.close('all')

    @staticmethod
    def add_noise(analysis, SNR=40):
        avg_V_db = 10 * np.log10(np.mean(np.array(analysis.input_dummy, dtype=float) ** 2))
        avg_Vnoise_db = avg_V_db - SNR
        Vnoise = np.random.normal(0, np.sqrt(10 ** (avg_Vnoise_db / 10)), len(analysis.input_dummy))
        analysis.input_dummy = np.array(analysis.input_dummy, dtype=float) + Vnoise
        avg_I_db = 10 * np.log10(np.mean(np.array(analysis.VCurrent, dtype=float) ** 2))
        avg_Inoise_db = avg_I_db - SNR
        Inoise = np.random.normal(0, np.sqrt(10 ** (avg_Inoise_db / 10)), len(analysis.VCurrent))
        analysis.VCurrent = np.array(analysis.VCurrent, dtype=float) + Inoise
        return analysis
