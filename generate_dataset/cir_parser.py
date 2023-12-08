import logging

from PySpice.Logging import Logging
from PySpice.Spice.Parser import SpiceParser

Logging.setup_logging(logging_level=logging.ERROR)


parser = SpiceParser(path=r'C:\dev\ivc-circuit-detector\circuit_classes\D_R\D_R.cir')
circuit = parser.build_circuit()

el = circuit['R1']
circuit.R1.resistance = '2K'

print(circuit)
print(circuit['R1'])
