import logging

from PySpice.Logging import Logging
from PySpice.Spice.Parser import SpiceParser
from PySpice.Spice.Netlist import DeviceModel

Logging.setup_logging(logging_level=logging.ERROR)


parser = SpiceParser(path=r'C:\dev\ivc-circuit-detector\circuit_classes\D_R\D_R.cir')
# parser = SpiceParser(path='dataset/test.cir')
circuit = parser.build_circuit()

# circuit['R1'].resistance = '0.0001K'
# circuit.models['DMOD_D1'].param('VJ', 42)
# circuit.model('DMOD_LOL', 'D', Af=1, Bv=10, Cj0=1e-14, Eg=1.11, Fc=0.5, Ibv=0.001, Is=1e-10, Kf=0, M=0.5, N=1, Rs=0, Tcv=0, Tm1=0, Tm2=0, Tnom=26.85, Trs=0, Tt=0, Ttt1=0, Ttt2=0, Vj=0.7, Xti=3)

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


# print(circuit['D1'].model)
# print(circuit)
# circuit.R1.resistance = '20K'
# circuit.D1.area = 2.0
# print(circuit.D1)
# print(circuit.raw_spice)
with open('dataset/test.cir', 'w+') as f:
    f.write(str(circuit))

# print(circuit)
# print(circuit['R1'])
