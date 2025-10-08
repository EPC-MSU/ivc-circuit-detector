import glob
import json
import os

from generate_dataset.parameters_changer import ParametersChanger
from generate_dataset.simulator_ivc import SimulatorIVC

BASE_CLASSES_FOLDER = "circuit_classes"
PARAMETERS_SETTINGS_PATH = "generate_dataset\\parameters_variations.json"
MEASUREMENTS_SETTINGS_PATH = "generate_dataset\\measurement_settings.json"
DEFAULT_DATASET_FOLDER = "dataset"


def generate_dataset(save_png=False, dataset_dir=None, disable_filtering=False):
    """
    Generate dataset from circuit classes.

    Args:
        save_png: Whether to save PNG images for each dataset file
        dataset_dir: Output directory for dataset (default: "dataset")
        disable_filtering: If True, skip boundary condition filtering (default: False)
    """
    if dataset_dir is None:
        dataset_dir = DEFAULT_DATASET_FOLDER

    with open(PARAMETERS_SETTINGS_PATH, "r") as f:
        parameters_settings = json.load(f)

    with open(MEASUREMENTS_SETTINGS_PATH, "r") as f:
        measurements_settings = json.load(f)

    classes_folders = glob.glob(os.path.join(BASE_CLASSES_FOLDER, "*"))

    for measurement_variant in measurements_settings["variants"]:
        if measurement_variant.get("enabled", False) is False:
            continue
        for circuit_class_folder in classes_folders:
            _, cls = os.path.split(circuit_class_folder)
            cir_path = os.path.join(circuit_class_folder, cls + ".cir")
            scheme_png_path = os.path.join(circuit_class_folder, cls + ".png")
            output_path = os.path.join(dataset_dir, measurement_variant["name"])

            changer = ParametersChanger(cir_path, parameters_settings)
            changer.generate_circuits()
            changer.dump_circuits_on_disk(output_path)

            simulator = SimulatorIVC(measurement_variant)

            params_combinations = changer._get_params_combinations(changer._settings)

            # Format parameter details for the error message
            def format_params(param_combination):
                params_parts = []
                for element_name, element_params in param_combination.items():
                    for param in element_params:
                        param_name = param.get("_name", param.get("cir_key", "value"))
                        param_value = param["value"]
                        param_unit = param.get("_units", param.get("cir_unit", ""))
                        params_parts.append(f"{element_name}_{param_name}={param_value}{param_unit}")
                return ", ".join(params_parts)

            for i, (circuit, params_combination) in enumerate(zip(changer.circuits, params_combinations)):
                print(f"Generating {output_path}_params{i:03d} with {format_params(params_combination)}")

                # Get original I-V curve
                original_ivc = simulator.get_ivc(circuit)

                # Check if original circuit differs enough from all bound circuits
                should_save = True

                if not disable_filtering:
                    # Generate bound circuits for comparison
                    bound_circuits_info = changer.generate_bound_circuits_with_params(params_combination)
                    min_difference_threshold = changer.min_difference_threshold

                    for bound_circuit, bound_params_combination in bound_circuits_info:
                        bound_ivc = simulator.get_ivc(bound_circuit)
                        difference = simulator.compare_ivc(original_ivc, bound_ivc)

                        # If any bound circuit is too similar (difference <= threshold), skip this circuit
                        if difference <= min_difference_threshold:
                            should_save = False
                            # Store info for detailed error message
                            similarity_difference = difference
                            similar_bound_params = bound_params_combination
                            break

                # Only save files if circuit differs enough from all bounds (or filtering is disabled)
                if should_save:
                    analysis = original_ivc
                    if measurement_variant["noise_settings"]["without_noise"]:
                        uzf_name = os.path.join(output_path, f"{cls}_params{i:03d}_noise_no.uzf")
                        png_name = os.path.join(output_path, f"{cls}_params{i:03d}_noise_no.png")
                        simulator.save_ivc(circuit.plot_title, analysis, uzf_name)
                        simulator.save_plot(circuit.plot_title, analysis, png_name, scheme_png_path, save_png=save_png)

                    for noise_number in range(measurement_variant["noise_settings"]["with_noise_copies"]):
                        analysis = simulator.add_noise(analysis, measurement_variant["noise_settings"])

                        uzf_name = os.path.join(output_path, f"{cls}_params{i:03d}_noise{noise_number+1:03d}.uzf")
                        png_name = os.path.join(output_path, f"{cls}_params{i:03d}_noise{noise_number+1:03d}.png")
                        simulator.save_ivc(circuit.plot_title, analysis, uzf_name)
                        simulator.save_plot(circuit.plot_title, analysis, png_name, scheme_png_path, save_png=save_png)
                else:
                    if not disable_filtering:
                        print(f"Too similar (difference: {similarity_difference:.4f} <= threshold: "
                              f"{min_difference_threshold}) to boundary circuit {format_params(similar_bound_params)}")
