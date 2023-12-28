import copy
import json
import logging
import itertools

import numpy as np
from PySpice.Logging import Logging
from PySpice.Spice.Parser import SpiceParser
from PySpice.Spice.Netlist import Circuit, DeviceModel
from PySpice.Spice.Parser import Model
from PySpice.Spice.Parser import Element

Logging.setup_logging(logging_level=logging.ERROR)


class UnknownElementToVariate(ValueError):
    pass


class UnknownIntervalType(ValueError):
    pass


class CleanSpiceParser(SpiceParser):
    @staticmethod
    def _build_circuit(circuit, statements, ground):
        for statement in statements:
            if isinstance(statement, Element):
                if statement.name != 'rint':
                    statement.build(circuit, ground)
            elif isinstance(statement, Model):
                statement.build(circuit)



# circuit = Circuit(self.base_circuit.title,
#                   self.base_circuit._ground,
#                   self.base_circuit._global_nodes)
# circuit._nodes = self.base_circuit._nodes
# circuit._includes = self.base_circuit._includes
# circuit._libs = self.base_circuit._libs
# circuit._elements = self.base_circuit._elements
# circuit._models = self.base_circuit._models
# circuit._parameters = self.base_circuit._parameters


class ParametersChanger:
    def __init__(self, cir_path, generation_parameters_json_path):
        self.parser = CleanSpiceParser(path=cir_path)
        with open(generation_parameters_json_path, 'r') as f:
            self.gen_params = json.load(f)
        self.base_circuit = self.parser.build_circuit()
        self.generated_circuits = []

    def generate_all_circuits(self):
        existed_elements = self._find_variate_parameters()
        self._generate_intervals(existed_elements)
        named_param_sets = self._get_params_sets(existed_elements)
        self._generate_circuit_with_params_set(named_param_sets)

    def _generate_circuit_with_params_set(self, named_param_sets):
        for named_param_set in named_param_sets:
            self.generated_circuits.append(self._params_set_to_circuit(named_param_set))

    def _params_set_to_circuit(self, params_set):
        circuit = self.base_circuit.clone()
        for el_name, el_params in params_set.items():
            # Some crutch or define what element has DeviceModel and what hasn't
            if el_params[0]['cir_key'] is not None:
                # Has DeviceModel (D, transistors etc)


                dev_model = list(circuit.models)[0]
                new_params = {}
                for key in dev_model.parameters:

                    if key not in ['Is']:
                        new_params[key] = a[key]
                        continue

                    if key == 'Is':
                        new_params[key] = 1e-42

                new_model = DeviceModel(dev_model.name, dev_model.model_type, **new_params)
                circuit._models['DMOD_D1'] = new_model

            else:
                # Hasn't DeviceModel (R, L, C, etc)
                # Only one named class attribute, so set this attribute to value at list[0] + unit letter
                attr_names = {'R': 'resistance',
                              'C': 'capacitance_expression',
                              'L': 'inductance_expression'}
                setattr(circuit[el_name],
                        attr_names[el_name[0]],
                        str(el_params[0]['value']) + str(el_params[0]['cir_unit']))

        with open('dataset\\test.cir', 'w+') as f:
            f.write(str(circuit))

    @staticmethod
    def _get_params_sets(existed_elements):
        _all_intervals = []
        params_keys = []

        # Create a list of intervals lists
        for k, params in existed_elements.items():
            for i, param in enumerate(params):
                _all_intervals.append(param['interval'])
                del param['interval']
                del param['nominal']
                params_keys.append({k: param}) #.copy()

        # Generate all possible params combinations
        params_sets = list(itertools.product(*_all_intervals))

        # Create a named params dict from every combination
        named_param_sets = []
        for params_set in params_sets:
            named_param_set = list(params_keys.copy())
            for i, param_value in enumerate(params_set):
                named_param_set[i][tuple(named_param_set[i].keys())[0]]['value'] = param_value

            # Reshape from `[R1, D1_p1, D1_p2]` to `[R1, D1[p1, p2]]`
            reshaped_named_param_set = {}
            for named_param in copy.deepcopy(named_param_set):
                k, v = named_param.popitem()
                if k in reshaped_named_param_set.keys():
                    reshaped_named_param_set[k].append(v)
                else:
                    reshaped_named_param_set[k] = [v]
            named_param_sets.append(reshaped_named_param_set)
        return named_param_sets

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
            for i, param in enumerate(params):
                existed_elements[k][i]['interval'] = self._description_to_interval_points(param['nominal'])
        return existed_elements

    @staticmethod
    def _description_to_interval_points(nominal):
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


changer = ParametersChanger('circuit_classes\\D_R\\D_R.cir', 'generate_dataset\\parameters_variations.json')
changer.generate_all_circuits()