import glob
import json
import os

from generate_dataset.parameters_changer import ParametersChanger
from generate_dataset.simulator_ivc import SimulatorIVC

BASE_CLASSES_FOLDER = "circuit_classes"
PARAMETERS_SETTINGS_PATH = 'generate_dataset\\parameters_variations.json'
MEASUREMENTS_SETTINGS_PATH = 'generate_dataset\\measurement_settings.json'
DATASET_FOLDER = 'dataset'


def generate_dataset(save_png=False):
    with open(PARAMETERS_SETTINGS_PATH, 'r') as f:
        parameters_settings = json.load(f)

    with open(MEASUREMENTS_SETTINGS_PATH, 'r') as f:
        measurements_settings = json.load(f)

    classes_folders = glob.glob(os.path.join(BASE_CLASSES_FOLDER, "*"))

    for meas_variant in measurements_settings['variants']:
        for circuit_class_folder in classes_folders:
            _, cls = os.path.split(circuit_class_folder)
            cir_path = os.path.join(circuit_class_folder, cls + '.cir')
            png_path = os.path.join(circuit_class_folder, cls + '.png')
            output_path = os.path.join(DATASET_FOLDER, meas_variant['name'], cls)

            changer = ParametersChanger(cir_path, parameters_settings)
            changer.generate_circuits()
            changer.dump_circuits_on_disk(output_path)

            simulator = SimulatorIVC(meas_variant)
            for i, circuit in enumerate(changer.circuits):
                print(output_path, i)
                analysis = simulator.get_ivc(circuit)
                cname = os.path.join(output_path, f'{i}.uzf')
                simulator.save_ivc(circuit, analysis, cname)

                if save_png:
                    pname = os.path.join(output_path, f'{i}.png')
                    simulator.save_plot(circuit, analysis, pname, png_path)
