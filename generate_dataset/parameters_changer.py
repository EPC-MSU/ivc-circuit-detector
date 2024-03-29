import copy
import itertools
import logging
import os

import numpy as np
from PySpice.Logging import Logging
from PySpice.Spice.Netlist import DeviceModel
from PySpice.Spice.Parser import Element, Model, SpiceParser

Logging.setup_logging(logging_level=logging.ERROR)


class UnknownElementToVariate(ValueError):
    pass


class UnknownIntervalType(ValueError):
    pass


class CleanSpiceParser(SpiceParser):
    # PySpice have a lot of bugs, this a class for fix some parser bugs without change PySpice
    # Unfortunately PySpice is dead and not accept new pull requests
    @staticmethod
    def _build_circuit(circuit, statements, ground):
        for statement in statements:
            if isinstance(statement, Element):
                if statement.name != 'rint':  # Print. Letter 'P' detected as other instruction
                    statement.build(circuit, ground)
            elif isinstance(statement, Model):
                statement.build(circuit)


class ParametersChanger:
    def __init__(self, base_cir_path, params_settings):
        """
        Class for iterating through all combinations of circuit parameters
        and creating .cir files with new parameters.

        :param base_cir_path: Path to base .cir file (circuit_classes/X/X.cir)
        :param params_settings: Dict with parameters intervals
        """
        self.circuits = []
        self.base_cir_path = base_cir_path
        self.circuit_class = os.path.splitext(os.path.basename(self.base_cir_path))[0]
        self._base_circuit = CleanSpiceParser(path=base_cir_path).build_circuit()
        self._params_settings = params_settings
        self._settings = self._generate_intervals(self._filter_settings(self._params_settings))
        self._assist_settings = self._make_assist_settings()

    def generate_circuits(self, debug) -> None:
        """
        Generate all possible parameters combinations according to settings in
        `self.params_settings_path` for specific circuit from `self.base_cir_path`

        Generated circuits you can find in `self.circuits`

        For save to disk use .dump_circuits_on_disk() method
        """
        params_combinations = self._get_params_combinations(self._settings)
        for params_combination in params_combinations:
            circuit = self._params_combination_to_circuit(params_combination)
            if not debug:
                self.circuits.append(circuit)
            else:  # Debug filter - select only cooper plate parameters
                if circuit.plot_title == 'Class: [DC]\nD_1N4148_1(Tnom=26.85)\nC1(10N)':
                    self.circuits.append(circuit)
                if circuit.plot_title == 'Class: [DC(D_R)]\nR1(1K)\nD_1N4148_2' \
                                         '(Tnom=26.85)\nC1(10N)\nD_1N4148_1(Tnom=26.85)':
                    self.circuits.append(circuit)
                if circuit.plot_title == 'Class: [DCR]\nD_1N4148_1(Tnom=26.85)\nC1(10N)\nR1(0.1000K)':
                    self.circuits.append(circuit)
                if circuit.plot_title == 'Class: [DD_R]\nD_1N4148_1(Tnom=26.85)\nR1(0.1000K)\nD_1N4148_2(Tnom=26.85)':
                    self.circuits.append(circuit)
                if circuit.plot_title == 'Class: [DR]\nD_1N4148_1(Tnom=26.85)\nR1(10K)':
                    self.circuits.append(circuit)
                if circuit.plot_title == 'Class: [D_C]\nC1(10N)\nD_1N4148_1(Tnom=26.85)':
                    self.circuits.append(circuit)
                if circuit.plot_title == 'Class: [R]\nR1(1K)':
                    self.circuits.append(circuit)
                if circuit.plot_title == 'Class: [RC]\nC1(10N)\nR1(10K)':
                    self.circuits.append(circuit)
                if circuit.plot_title == 'Class: [R_D]\nD_1N4148_1(Tnom=26.85)\nR1(1K)':
                    self.circuits.append(circuit)
                if circuit.plot_title == 'Class: [R_C]\nR1(0.5100K)\nC1(10N)':
                    self.circuits.append(circuit)

    def dump_circuits_on_disk(self, base_folder) -> None:
        """
        Method dumps `self.circuits` to disk as .cir-files
        :param base_folder: Folder to save .cir files. (If not exist - it's ok)
        """
        os.makedirs(base_folder, exist_ok=True)
        for i, circuit in enumerate(self.circuits):
            with open(os.path.join(base_folder, f'{i}.cir'), 'w+') as f:
                f.write(str(circuit))

    def _params_combination_to_circuit(self, params_combination):
        # Make from combination-dict a circuit with this params.
        circuit = self._base_circuit.clone()  # TODO: Fix clone without change PySpice
        circuit.plot_title = self._circuit_plot_title(params_combination)
        for el_name, el_params in params_combination.items():
            # Some crutch or define what element has DeviceModel and what hasn't
            if el_params[0]['cir_key'] is not None:
                # Has DeviceModel(s) (D, transistors etc)
                for dev_model in list(circuit.models):
                    new_params = {}
                    for key in dev_model.parameters:
                        if key not in [el_param['cir_key'] for el_param in el_params]:
                            new_params[key] = dev_model[key]
                            continue
                        param_key = next(item for item in el_params if item["cir_key"] == key)
                        new_params[key] = param_key['value']
                    new_model = DeviceModel(dev_model.name, dev_model.model_type, **new_params)
                    circuit._models[dev_model.name] = new_model
            else:
                # Hasn't DeviceModel (R, L, C, etc)
                # Only one named class attribute, so set this attribute to value at list[0] + unit letter
                attr_names = {'R': 'resistance',
                              'C': 'capacitance_expression',
                              'L': 'inductance_expression'}
                setattr(circuit[el_name],
                        attr_names[el_name[0]],
                        str(el_params[0]['value']) + str(el_params[0]['cir_unit']))
        return circuit

    def _get_params_combinations(self, settings):
        # Get all params intervals. Make list of intervals lists (need for itertools.product)
        _all_intervals = []
        for params in settings.values():
            for param in params:
                _all_intervals.append(param['interval'])

        # Generate all possible params combinations
        raw_params_combs = list(itertools.product(*_all_intervals))

        # Create a named params dict for every combination
        params_combinations = []
        for raw_params_comb in raw_params_combs:
            # Set a 'value' key for every param with numerical value
            named_params_comb = copy.deepcopy(self._assist_settings)
            for i, param_value in enumerate(raw_params_comb):
                key = tuple(named_params_comb[i].keys())[0]
                named_params_comb[i][key]['value'] = param_value

            # Reshape from `[R1, D1_p1, D1_p2]` to `[R1, D1[p1, p2]]`
            reshaped_named_param_set = {}
            for named_param in copy.deepcopy(named_params_comb):
                k, v = named_param.popitem()
                if k in reshaped_named_param_set.keys():
                    reshaped_named_param_set[k].append(v)
                else:
                    reshaped_named_param_set[k] = [v]

            # Add reshaped named combination to final list
            params_combinations.append(reshaped_named_param_set)
        return params_combinations

    def _filter_settings(self, params_settings):
        # Filter parameters variation settings only for existed elements in self._base_circuit
        settings_filtered = {}
        for elem_name in list(self._base_circuit.element_names):
            if elem_name[0] not in params_settings['elements']:
                raise UnknownElementToVariate(f'Element {elem_name} is unknown for parameters variation')
            # Create a dict like 'R1' = {<Resistance variate settings>}
            settings_filtered[elem_name] = params_settings['elements'][elem_name[0]]
        return settings_filtered

    def _generate_intervals(self, settings):
        # Make from intervals descriptions intervals with points itself
        for k, params in settings.items():
            for i, param in enumerate(params):
                settings[k][i]['interval'] = self._description_to_interval_points(param['nominal'])
        return settings

    @staticmethod
    def _description_to_interval_points(nominal):
        # Make from interval description interval with points itself
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
        elif nominal['type'] == 'list':
            a = list(nominal['value'])
            return a
        else:
            raise UnknownIntervalType(f'Interval {nominal["type"]} unknown')

    def _make_assist_settings(self):
        # Fill assist variable with no intervals, just elements description
        clean_settings = []
        for k, params in self._settings.items():
            for param in params:
                clean_param = copy.deepcopy(param)
                del clean_param['interval']
                del clean_param['nominal']
                clean_settings.append({k: clean_param})
        return clean_settings

    def _circuit_plot_title(self, params_combination) -> str:
        els = []
        for k, params in params_combination.items():
            s_params = []
            for param in params:
                if param["cir_key"] is not None:
                    s_params.append(f'{param["cir_key"]}={param["value"]}')
                else:
                    if param["value"] >= 1:
                        s_params.append(f'{param["value"]:.0f}{param["cir_unit"]}')
                    else:
                        s_params.append(f'{param["value"]:.4f}{param["cir_unit"]}')
            els.append(f'{k}({", ".join(s_params)})')
        elems = "\n".join(els)
        return f"""Class: [{self.circuit_class}]\n{elems}"""
