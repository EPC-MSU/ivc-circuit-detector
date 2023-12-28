import json
import logging

from PySpice.Logging import Logging
from PySpice.Spice.Parser import SpiceParser
from PySpice.Spice.Netlist import DeviceModel

Logging.setup_logging(logging_level=logging.ERROR)


parser = SpiceParser(path=r'C:\dev\ivc-circuit-detector\circuit_classes\D_R\D_R.cir')
# parser = SpiceParser(path='dataset/test.cir')
circuit = parser.build_circuit()

# circuit['R1'].resistance = '0.0001K'
# circuit['D1'].model = 'DMOD_LOL'

a = list(circuit.models)[0]
new_params = {}
for key in a.parameters:
    if key not in ['Is']:
        new_params[key] = a[key]
        continue

    if key == 'Is':
        new_params[key] = 1e-42

new_model = DeviceModel(a.name, a.model_type, **new_params)
circuit._models['DMOD_D1'] = new_model

#
# # print(circuit['D1'].model)
# # print(circuit)
# # circuit.R1.resistance = '20K'
# # circuit.D1.area = 2.0
# # print(circuit.D1)
# # print(circuit.raw_spice)
# with open('dataset/test.cir', 'w+') as f:
#     f.write(str(circuit))
#
# # print(circuit)
# # print(circuit['R1'])
#
#
# class ParametersChanger:
#     def __init__(self, cir_path, generation_parameters_json_path):
#         self.parser = SpiceParser(path=cir_path)
#         with open(generation_parameters_json_path, 'r') as f:
#             self.gen_params = json.load(f)
#         self.base_circuit = parser.build_circuit()
#         self.generated_circuits = []
#
#     def generate_all_circuits(self):
#         print(self.gen_params)
#
#
# changer = ParametersChanger('circuit_classes\\C_R\\C_R.cir', 'generate_dataset\\generation_parameters.json')
# changer.generate_all_circuits()
