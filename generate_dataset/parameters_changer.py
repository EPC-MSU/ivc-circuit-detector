import json
import logging

import numpy as np
from PySpice.Logging import Logging
from PySpice.Spice.Parser import SpiceParser
from PySpice.Spice.Netlist import DeviceModel

Logging.setup_logging(logging_level=logging.ERROR)


class UnknownElementToVariate(ValueError):
    pass

class UnknownIntervalType(ValueError):
    pass


class ParametersChanger:
    def __init__(self, cir_path, generation_parameters_json_path):
        self.parser = SpiceParser(path=cir_path)
        with open(generation_parameters_json_path, 'r') as f:
            self.gen_params = json.load(f)
        self.base_circuit = self.parser.build_circuit()
        self.generated_circuits = []

    def generate_all_circuits(self):
        existed_elements = self._find_variate_parameters()
        self._generate_intervals(existed_elements)

    def _find_variate_parameters(self):
        elem_names = [x for x in list(self.base_circuit.element_names) if x not in ['Print', 'print']]
        existed_elements = {}
        for elem_name in elem_names:
            if elem_name[0] not in self.gen_params['elements']:
                raise UnknownElementToVariate(f'Element {elem_name} is unknown for parameters variation')
            # Create a dict like 'R1' = {<Resistance variate settings>}
            existed_elements[elem_name] = self.gen_params['elements'][elem_name[0]]
        return existed_elements

    def _generate_intervals(self, existed_elements):
        for k, params in existed_elements.items():
            for param in params:
                print(param['name'], self._interval_description_to_interval_points(param['nominal']))

    @staticmethod
    def _interval_description_to_interval_points(nominal):
        if nominal['type'] == 'constant':
            return [nominal['value']]
        elif nominal['type'] == 'uniform_interval':
            return list(np.linspace(nominal['interval'][0],
                                    nominal['interval'][1],
                                    nominal['interval_points'],
                                    endpoint=True))
        elif nominal['type'] == 'exponential_interval':
            return list(np.geomspace(nominal['interval'][0],
                                    nominal['interval'][1],
                                    nominal['interval_points'],
                                    endpoint=True))
        else:
            raise UnknownIntervalType(f'Interval {nominal["type"]} unknown')


changer = ParametersChanger('circuit_classes\\C_R\\C_R.cir', 'generate_dataset\\parameters_variations.json')
changer.generate_all_circuits()
