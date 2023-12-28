import os
import glob

from generate_dataset.parameters_changer import ParametersChanger


def generate_dataset():
    folders = glob.glob("circuit_classes/*")
    for folder in folders:
        top, cls = os.path.split(folder)
        cir_path = os.path.join(folder, cls + '.cir')
        changer = ParametersChanger(cir_path, 'generate_dataset\\parameters_variations.json')
        changer.generate_all_circuits()
        path = os.path.join('dataset', 'measurement_default', cls)
        changer.dump_circuits_on_disk(path)


generate_dataset()
