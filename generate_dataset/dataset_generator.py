import os
import glob

from generate_dataset.parameters_changer import ParametersChanger

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
            changer = ParametersChanger(cir_path, GENERATE_SETTINGS_PATH)
            changer.generate_circuits()
            path = os.path.join('dataset', measurements_settings, cls)
            changer.dump_circuits_on_disk(path)


generate_dataset()
