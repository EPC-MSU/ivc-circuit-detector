import matplotlib.pyplot as plt
import numpy as np
from epcore.elements import Board
from epcore.filemanager import save_board_to_ufiv
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PySpice.Spice.Parser import Circuit


class SimulatorIVC:
    def __init__(self, measurement_variant):
        self.measurement_settings = measurement_variant['measurement_settings']

    def get_ivc(self, circuit: Circuit):
        rms_voltage = self.measurement_settings['max_voltage'] / np.sqrt(2)

        # ssr = simulator sampling rate
        ssr = self.measurement_settings['sampling_rate']

        # TODO: Dirty quality increase, but set-up EPLab for new settings is worse
        if ssr == 2000000:
            ssr = 10000000

        ssr *= 2  # Increase quality

        circuit.R('cs', 'input', 'input_dummy', self.measurement_settings['internal_resistance'])
        # circuit.C('coaxial_probes', circuit.gnd, 'input_dummy', 204*10**-12)  # 28*10**-12
        circuit.AcLine('Current', circuit.gnd, 'input_dummy', rms_voltage=rms_voltage,
                       frequency=self.measurement_settings['probe_signal_frequency'])

        period = 1 / self.measurement_settings['probe_signal_frequency']

        step_time = period / (ssr / self.measurement_settings['probe_signal_frequency'])
        end_time = period + self.measurement_settings['precharge_delay']

        simulator = circuit.simulator()
        analysis = simulator.transient(step_time=step_time, end_time=end_time)

        need_points = int(ssr / self.measurement_settings['probe_signal_frequency']) - 1

        voltages = analysis.input[-need_points:].as_ndarray()
        currents = analysis.VCurrent[-need_points:].as_ndarray()

        voltages = np.append(voltages, voltages[0])  # Close points circle
        currents = np.append(currents, currents[0])  # Close points circle

        assert len(voltages) == 200
        assert len(currents) == 200
        return voltages, currents

    def save_ivc(self, title, analysis, path):
        voltages, currents = analysis
        measurement = {'measurement_settings': self.measurement_settings,
                       'currents': list(currents),
                       'voltages': list(voltages),
                       'is_reference': True}

        board = {'version': "1.1.2",
                 'elements': [{'pins': [{'iv_curves': [measurement],
                                         'x': 0,
                                         'y': 0,
                                         'comment': title.replace('\n', ' ')
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
        ax.set_xlabel('Voltage [V]')
        ax.set_ylabel('Current [A]')

        arr_img = plt.imread(scheme_png_path)
        im = OffsetImage(arr_img, zoom=.5)
        ab = AnnotationBbox(im, (1, 0), xycoords='axes fraction', box_alignment=(-0.68, 0.1))
        ax.add_artist(ab)
        ax.set_title('IV-Characteristic')

        plt.figtext(0.62, 0.45, s=title)
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

        i_noise = np.random.normal(0, noise_settings['vertical_noise'], len(currents))
        v_noise = np.random.normal(0, noise_settings['horizontal_noise'], len(voltages))

        voltages = np.array(voltages, dtype=float) + v_noise
        currents = np.array(currents, dtype=float) + i_noise
        return voltages, currents
