import glob
import os.path


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
