# Cut screenshots from EPLab
# import numpy as np
# import os
# import glob
# import cv2
#
#
# files = glob.glob('C:\\dev\\ivc-circuit-detector\\dataset_human_compare\\original\\*\\*.png')
#
# for file in files:
#     img = cv2.imread(file)
#     img = img[100:900, 120:1600, :]
#     filename = file.replace("original", "cutted")
#     if not os.path.exists(os.path.split(filename)[0]):
#         os.makedirs(os.path.split(filename)[0])
#     cv2.imwrite(filename, img)
import glob

from epcore.elements import Board
from epcore.filemanager import save_board_to_ufiv, load_board_from_ufiv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import glob
import os

PATH = "dataset/measurement_1kHz_middle_5v/*.uzf"
files = glob.glob(PATH)
fig = plt.figure()
axe = fig.gca(polar=True)

for file in files:
    board = load_board_from_ufiv(file)
    a = board.to_json()
    v = np.array(a['elements'][0]['pins'][0]['iv_curves'][0]['voltages'])[:100]
    i = np.array(a['elements'][0]['pins'][0]['iv_curves'][0]['currents'])[:100]

    v = (v - v.min())/(v.max() - v.min())
    i = (i - i.min()) / (i.max() - i.min())

    w = a['elements'][0]['pins'][0]['iv_curves'][0]['measurement_settings']['probe_signal_frequency']
    dt = np.linspace(start=0, stop=(1 / w), num=len(v))

    A = np.sqrt(v ** 2 + i ** 2)

    thetas = np.linspace(0, 2 * np.pi, 100)
    axe.plot(thetas, A, label=os.path.basename(file))

    # plt.plot(dt, v)
    # plt.plot()
angle = np.deg2rad(67.5)
plt.legend(loc="lower left")
plt.show()


# files = glob.glob(PATH)
# for file in files:
#     board = load_board_from_ufiv(file).to_json()
#     v = np.array(board['elements'][0]['pins'][0]['iv_curves'][0]['voltages'])[:100]
#     i = np.array(board['elements'][0]['pins'][0]['iv_curves'][0]['currents'])[:100]
#
#     # v = (v - v.min())/(v.max() - v.min())
#     # i = (i - i.min()) / (i.max() - i.min())
#
#     w = board['elements'][0]['pins'][0]['iv_curves'][0]['measurement_settings']['probe_signal_frequency']
#     dt = np.linspace(start=0, stop=(1 / w), num=len(v))
#
#     phi = np.linspace(0, 2, num=100)
#
#     plt.plot(phi, i, label=os.path.basename(file))
#     # plt.plot(dt, v, label=os.path.basename(file))
#
#     # A = np.sqrt(v ** 2 + i ** 2)
#     # plt.plot(dt, A, label=os.path.basename(file))
#
#
# plt.ylabel('I, amps')
# gc = plt.gca()
# gc.xaxis.set_major_formatter(plt.FormatStrFormatter('%g $\pi$'))
# gc.xaxis.set_major_locator(MultipleLocator(base=0.25))
# plt.grid()
# plt.title(PATH.split('/')[1])
# plt.xlabel('period')
# plt.legend()
# plt.show()
