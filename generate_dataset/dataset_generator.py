import os
import glob

from generate_dataset.parameters_changer import ParametersChanger
from generate_dataset.simulate_ivc import SimulatorIVC

GENERATE_SETTINGS_PATH = 'generate_dataset\\parameters_variations.json'


def generate_dataset():
    # changer = ParametersChanger('circuit_classes\\DR_R\\DR_R.cir', GENERATE_SETTINGS_PATH)
    # changer.generate_circuits()
    # path = os.path.join('dataset', 'measurement_default', 'DR')
    # changer.dump_circuits_on_disk(path)

    folders = glob.glob("circuit_classes/*")
    for measurements_settings in ['measurement_none']:
        for folder in folders:
            top, cls = os.path.split(folder)
            cir_path = os.path.join(folder, cls + '.cir')
            png_path = os.path.join(folder, cls + '.png')
            changer = ParametersChanger(cir_path, GENERATE_SETTINGS_PATH)
            changer.generate_circuits()
            path = os.path.join('dataset', measurements_settings, cls)
            changer.dump_circuits_on_disk(path)

            simulator = SimulatorIVC(1000, 0.3, 0, 0, 0)
            for i, circuit in enumerate(changer.circuits):
                print(path, i)
                analysis = simulator.get_ivc(circuit)
                fname = os.path.join(path, f'{i}.csv')
                simulator.save_ivc(circuit, analysis, fname)

                pname = os.path.join(path, f'{i}.png')
                simulator.save_plot(circuit, analysis, pname, png_path)



generate_dataset()
