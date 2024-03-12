import glob
import json
import os

from generate_dataset.parameters_changer import ParametersChanger
from generate_dataset.simulator_ivc import SimulatorIVC

BASE_CLASSES_FOLDER = "circuit_classes"
PARAMETERS_SETTINGS_PATH = 'generate_dataset\\parameters_variations.json'
MEASUREMENTS_SETTINGS_PATH = 'generate_dataset\\measurement_settings.json'
DATASET_FOLDER = 'dataset'


def generate_dataset(save_png=False, debug=False):
    with open(PARAMETERS_SETTINGS_PATH, 'r') as f:
        parameters_settings = json.load(f)

    with open(MEASUREMENTS_SETTINGS_PATH, 'r') as f:
        measurements_settings = json.load(f)

    classes_folders = glob.glob(os.path.join(BASE_CLASSES_FOLDER, "*"))

    for measurement_variant in measurements_settings['variants']:
        for circuit_class_folder in classes_folders:
            _, cls = os.path.split(circuit_class_folder)
            cir_path = os.path.join(circuit_class_folder, cls + '.cir')
            scheme_png_path = os.path.join(circuit_class_folder, cls + '.png')
            output_path = os.path.join(DATASET_FOLDER, measurement_variant['name'])

            changer = ParametersChanger(cir_path, parameters_settings)
            changer.generate_circuits(debug)
            changer.dump_circuits_on_disk(output_path)

            simulator = SimulatorIVC(measurement_variant)

            for i, circuit in enumerate(changer.circuits):
                print(output_path, i)
                analysis = simulator.get_ivc(circuit)
                if measurement_variant['noise_settings']['without_noise']:
                    uzf_name = os.path.join(output_path, f'{cls}_params{i}.uzf')
                    png_name = os.path.join(output_path, f'{cls}_params{i}.png')
                    simulator.save_ivc(circuit.plot_title, analysis, uzf_name)
                    simulator.save_plot(circuit.plot_title, analysis, png_name, scheme_png_path, save_png=save_png)

                for noise_number in range(measurement_variant['noise_settings']['with_noise']):
                    analysis = simulator.add_noise(analysis, measurement_variant['noise_settings']['SNR'])

                    uzf_name = os.path.join(output_path, f'{cls}_params{i}_noise{noise_number}.uzf')
                    png_name = os.path.join(output_path, f'{cls}_params{i}_noise{noise_number}.png')
                    simulator.save_ivc(circuit.plot_title, analysis, uzf_name)
                    simulator.save_plot(circuit.plot_title, analysis, png_name, scheme_png_path, save_png=save_png)
