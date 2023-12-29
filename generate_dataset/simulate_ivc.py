

import csv
import numpy as np
import matplotlib.pyplot as plt
from PySpice.Spice.Parser import Circuit


class SimulatorIVC:
    def __init__(self,
                 probe_signal_frequency,
                 max_voltage,
                 precharge_delay,
                 sampling_rate,
                 internal_resistance):
        self.probe_signal_frequency = probe_signal_frequency
        self.max_voltage = max_voltage
        # self.precharge_delay = precharge_delay
        # self.sampling_rate = sampling_rate
        # self.internal_resistance = internal_resistance
        self.num_cycles = 1

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

    @staticmethod
    def save_ivc(circuit, analysis, path):
        with open(path, 'w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=';')
            csv_writer.writerow(analysis.input_dummy)
            csv_writer.writerow(analysis.VCurrent)

    @staticmethod
    def save_plot(circuit, analysis, path):
        plt.figure(1, (10, 10))
        plt.grid()
        plt.plot(analysis.input_dummy, analysis.VCurrent)
        plt.xlabel('Напряжение [В]')
        plt.ylabel('Сила тока [А]')
        plt.figtext(0.5, 0.8, s=circuit.title)
        plt.savefig(path, dpi=200)
        plt.clf()

    # def add_noise(self):
    #     # Расчитываем шум независмо для тока и напряжения исходя из среднеквадратичных значений и одинакового SNR
    #     avg_V_db = 10 * np.log10(np.mean(np.array(analysis.input_dummy, dtype=float) ** 2))
    #     avg_Vnoise_db = avg_V_db - input_data.SNR
    #     Vnoise = np.random.normal(0, np.sqrt(10 ** (avg_Vnoise_db / 10)), len(analysis.input_dummy))
    #     analysis.input_dummy = np.array(analysis.input_dummy, dtype=float) + Vnoise
    #     avg_I_db = 10 * np.log10(np.mean(np.array(analysis.VCurrent, dtype=float) ** 2))
    #     avg_Inoise_db = avg_I_db - input_data.SNR
    #     Inoise = np.random.normal(0, np.sqrt(10 ** (avg_Inoise_db / 10)), len(analysis.VCurrent))
    #     analysis.VCurrent = np.array(analysis.VCurrent, dtype=float) + Inoise



class Init_Data:
    F: float
    V: float
    Rcs: float = 0.0
    SNR: float = 40.0


def LoadFile(path):
    parser = MySpiceParser(path=path)
    circuit = parser.build_circuit()
    return circuit


def SaveFile(analysis, path):
    with open(path, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=';')
        csv_writer.writerow(analysis.input_dummy)
        csv_writer.writerow(analysis.VCurrent)
    return


def CreateCVC(circuit, input_data, lendata, cycle=1):
    # lendata не может принимать значения меньше 59
    # if lendata>86:
    #    lendata = lendata - 8
    # else:
    #    lendata = lendata - 9
    period = 1 / input_data.F
    rms_voltage = input_data.V / math.sqrt(2)
    circuit.R('cs', 'input', 'input_dummy', input_data.Rcs)
    circuit.AcLine('Current', circuit.gnd, 'input_dummy', rms_voltage=rms_voltage, frequency=input_data.F)
    simulator = circuit.simulator()
    analysis = simulator.transient(step_time=period / lendata, end_time=period * cycle)
    analysis.input_dummy = analysis.input_dummy[len(analysis.input_dummy)-lendata:len(analysis.input_dummy)]
    analysis.VCurrent = analysis.VCurrent[len(analysis.VCurrent)-lendata:len(analysis.VCurrent)]
# Расчитываем шум независмо для тока и напряжения исходя из среднеквадратичных значений и одинакового SNR
    avg_V_db = 10 * np.log10(np.mean(np.array(analysis.input_dummy, dtype=float) ** 2))
    avg_Vnoise_db = avg_V_db - input_data.SNR
    Vnoise = np.random.normal(0, np.sqrt(10 ** (avg_Vnoise_db / 10)), len(analysis.input_dummy))
    analysis.input_dummy = np.array(analysis.input_dummy, dtype=float) + Vnoise
    avg_I_db = 10 * np.log10(np.mean(np.array(analysis.VCurrent, dtype=float) ** 2))
    avg_Inoise_db = avg_I_db - input_data.SNR
    Inoise = np.random.normal(0, np.sqrt(10 ** (avg_Inoise_db / 10)), len(analysis.VCurrent))
    analysis.VCurrent = np.array(analysis.VCurrent, dtype=float) + Inoise
    return analysis
