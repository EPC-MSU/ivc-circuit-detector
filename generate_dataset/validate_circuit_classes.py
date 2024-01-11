import glob
import json
import os.path

from generate_dataset.dataset_generator import MEASUREMENTS_SETTINGS_PATH


def validate_circuit_classes():
    folders = glob.glob("circuit_classes/*")
    for folder in folders:
        top, cls = os.path.split(folder)

        cir = os.path.join(folder, cls + '.cir')
        png = os.path.join(folder, cls + '.png')
        sch = os.path.join(folder, cls + '.sch')

        if not os.path.isfile(cir):
            raise FileNotFoundError(cir)
        if not os.path.isfile(png):
            raise FileNotFoundError(png)
        if not os.path.isfile(sch):
            raise FileNotFoundError(sch)

        with open(cir, 'r') as f:
            text = ''.join(f.readlines())
        if 'input' not in text:
            raise ValueError(f'Label "input" doesn\'t exist in "{cir}". Read documentation at 2.2.1(6)')


def validate_measurement_settings():
    with open(MEASUREMENTS_SETTINGS_PATH, 'r') as f:
        measurements_settings = json.load(f)

    for measurement_variant in measurements_settings['variants']:
        name = measurement_variant['name']
        if not measurement_variant['noise_settings']['without_noise'] or \
                not len(measurement_variant['noise_settings']['with_noise']) > 0:
            raise ValueError(f'Both noise generation variants disabled for {name} in {MEASUREMENTS_SETTINGS_PATH}'
                             f'\n"without_noise" can\'t be false while "with_noise" equal 0')
