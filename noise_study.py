import glob
from epcore.filemanager import load_board_from_ufiv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def check_file(file, board):
    f, s, v = file.split('\\')[1].replace('.uzf', '').split('_')
    f = f.replace('Hz', '')
    v = v.replace('V', '')
    ms = board['elements'][0]['pins'][0]['iv_curves'][0]['measurement_settings']

    sens = {'high': 47500, 'middle': 4750, 'low': 475}
    try:
        assert sens[s] == ms['internal_resistance']
        assert int(f) == int(ms['probe_signal_frequency'])
        assert round(float(v), ndigits=2) == round(float(ms['max_voltage']), ndigits=2)
    except AssertionError as err:
        print(file.split('\\')[1])
        raise err


files = glob.glob("dataset_noise_study/*uzf")
for file in files:
    board = load_board_from_ufiv(file).to_json()

    check_file(file, board)

    stds = []
    colors = ['red']
    for j, color in zip(range(len(board['elements'][0]['pins'])), mcolors.TABLEAU_COLORS):
        v = np.array(board['elements'][0]['pins'][j]['iv_curves'][0]['voltages'])
        i = np.array(board['elements'][0]['pins'][j]['iv_curves'][0]['currents'])
        stds.append(np.std(i))

        v = (v - v.min()) / (v.max() - v.min())
        i = (i - i.min()) / (i.max() - i.min())

        plt.plot(range(len(v)), v, c=color)
        plt.plot(range(len(i)), i, c=color, linestyle='--')
    print(np.mean(stds), '\t', file.split('\\')[1])

    plt.title(file.split('\\')[1])
    plt.ylabel('Normed values')
    plt.xlabel('V (solid line), I(dashed line)')
    plt.show()
