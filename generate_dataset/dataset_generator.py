import json
import os
import glob

from generate_dataset.parameters_changer import ParametersChanger
from generate_dataset.simulate_ivc import SimulatorIVC

GENERATE_SETTINGS_PATH = 'generate_dataset\\parameters_variations.json'
MEASUREMENTS_SETTINGS_PATH = 'generate_dataset\\measurement_settings.json'


def generate_dataset(save_png=False):
    # # Example how to create .cir-files for one circuit class
    # changer = ParametersChanger('circuit_classes\\DR_R\\DR_R.cir', GENERATE_SETTINGS_PATH)
    # changer.generate_circuits()
    # path = os.path.join('dataset', 'measurement_default', 'DR')
    # changer.dump_circuits_on_disk(path)

    with open(MEASUREMENTS_SETTINGS_PATH, 'r') as f:
        measurements_settings = json.load(f)

    folders = glob.glob("circuit_classes/*")
    for measurement in measurements_settings['measurement_variants']:
        for circuit_class_path in folders:
            _, cls = os.path.split(circuit_class_path)
            path = os.path.join('dataset', measurement['name'], cls)
            cir_path = os.path.join(circuit_class_path, cls + '.cir')
            png_path = os.path.join(circuit_class_path, cls + '.png')

            changer = ParametersChanger(cir_path, GENERATE_SETTINGS_PATH)
            changer.generate_circuits()
            changer.dump_circuits_on_disk(path)

            simulator = SimulatorIVC(measurement['measurement_settings'])
            for i, circuit in enumerate(changer.circuits):
                print(path, i)
                analysis = simulator.get_ivc(circuit)
                cname = os.path.join(path, f'{i}.uzf')
                simulator.save_ivc(circuit, analysis, cname)

                if save_png:
                    pname = os.path.join(path, f'{i}.png')
                    simulator.save_plot(circuit, analysis, pname, png_path)
