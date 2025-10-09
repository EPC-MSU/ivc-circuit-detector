import glob
import json
import os

from generate_dataset.parameters_changer import ParametersChanger
from generate_dataset.simulator_ivc import SimulatorIVC

BASE_CLASSES_FOLDER = "circuit_classes"
PARAMETERS_SETTINGS_PATH = "generate_dataset\\parameters_variations.json"
MEASUREMENTS_SETTINGS_PATH = "generate_dataset\\measurement_settings.json"
DEFAULT_DATASET_FOLDER = "dataset"


def _format_params(param_combination):
    """
    Format parameter combination for display in messages.

    Args:
        param_combination: Dictionary of element names to their parameters

    Returns:
        Formatted string representation of parameters
    """
    params_parts = []
    for element_name, element_params in param_combination.items():
        for param in element_params:
            param_name = param.get("_name", param.get("cir_key", "value"))
            param_value = param["value"]
            param_unit = param.get("_units", param.get("cir_unit", ""))
            params_parts.append(f"{element_name}_{param_name}={param_value}{param_unit}")
    return ", ".join(params_parts)


def _check_filtering(original_ivc, params_combination, changer, simulator):
    """
    Check if circuit should be filtered out based on boundary conditions.

    Args:
        original_ivc: Original I-V curve
        params_combination: Parameter combination for this circuit
        changer: ParametersChanger instance
        simulator: SimulatorIVC instance

    Returns:
        Tuple (should_save, similarity_difference, similar_bound_params)
        should_save: True if circuit should be saved
        similarity_difference: Difference value if filtered
        similar_bound_params: Parameters of similar bound circuit if filtered
    """
    should_save = True
    similarity_difference = None
    similar_bound_params = None

    bound_circuits_info = changer.generate_bound_circuits_with_params(params_combination)
    min_difference_threshold = changer.min_difference_threshold

    for bound_circuit, bound_params_combination in bound_circuits_info:
        bound_ivc = simulator.get_ivc(bound_circuit)
        difference = simulator.compare_ivc(original_ivc, bound_ivc)

        # If any bound circuit is too similar (difference <= threshold), skip this circuit
        if difference <= min_difference_threshold:
            should_save = False
            similarity_difference = difference
            similar_bound_params = bound_params_combination
            break

    return should_save, similarity_difference, similar_bound_params


def _save_circuit_files(circuit, original_ivc, simulator, measurement_variant, output_path,
                        cls, index, scheme_png_path, save_png):
    """
    Save circuit I-V curve files (with and without noise).

    Args:
        circuit: Circuit object
        original_ivc: Original I-V curve analysis
        simulator: SimulatorIVC instance
        measurement_variant: Measurement variant settings
        output_path: Output directory path
        cls: Circuit class name
        index: Circuit index
        scheme_png_path: Path to circuit scheme PNG
        save_png: Whether to save PNG plots
    """
    analysis = original_ivc

    # Save without noise if enabled
    if measurement_variant["noise_settings"]["without_noise"]:
        uzf_name = os.path.join(output_path, f"{cls}_params{index:03d}_noise_no.uzf")
        png_name = os.path.join(output_path, f"{cls}_params{index:03d}_noise_no.png")
        simulator.save_ivc(circuit.plot_title, analysis, uzf_name)
        simulator.save_plot(circuit.plot_title, analysis, png_name, scheme_png_path, save_png=save_png)

    # Save with noise copies
    for noise_number in range(measurement_variant["noise_settings"]["with_noise_copies"]):
        analysis = simulator.add_noise(analysis, measurement_variant["noise_settings"])

        uzf_name = os.path.join(output_path, f"{cls}_params{index:03d}_noise{noise_number+1:03d}.uzf")
        png_name = os.path.join(output_path, f"{cls}_params{index:03d}_noise{noise_number+1:03d}.png")
        simulator.save_ivc(circuit.plot_title, analysis, uzf_name)
        simulator.save_plot(circuit.plot_title, analysis, png_name, scheme_png_path, save_png=save_png)


def _process_single_circuit(circuit, params_combination, index, cls, changer, simulator,
                            measurement_variant, output_path, scheme_png_path,
                            save_png, disable_filtering):
    """
    Process a single circuit: simulate, filter (optionally), and save.

    Args:
        circuit: Circuit object
        params_combination: Parameter combination for this circuit
        index: Circuit index
        cls: Circuit class name
        changer: ParametersChanger instance
        simulator: SimulatorIVC instance
        measurement_variant: Measurement variant settings
        output_path: Output directory path
        scheme_png_path: Path to circuit scheme PNG
        save_png: Whether to save PNG plots
        disable_filtering: If True, skip boundary condition filtering
    """
    print(f"Generating {output_path}_params{index:03d} with {_format_params(params_combination)}")

    # Make actual simulation to get I-V curve
    original_ivc = simulator.get_ivc(circuit)

    # Check if original circuit differs enough from all bound circuits
    should_save = True
    similarity_difference = None
    similar_bound_params = None

    if not disable_filtering:
        should_save, similarity_difference, similar_bound_params = _check_filtering(
            original_ivc, params_combination, changer, simulator
        )

    # Only save files if circuit differs enough from all bounds (or filtering is disabled)
    if should_save:
        _save_circuit_files(circuit, original_ivc, simulator, measurement_variant,
                            output_path, cls, index, scheme_png_path, save_png)
    else:
        if not disable_filtering:
            print(f"Too similar (difference: {similarity_difference:.4f} <= threshold: "
                  f"{changer.min_difference_threshold}) to boundary circuit {_format_params(similar_bound_params)}")


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
        # For enabled variants of a measurement we take each circuit class one by one
        for circuit_class_folder in classes_folders:
            _, cls = os.path.split(circuit_class_folder)
            cir_path = os.path.join(circuit_class_folder, cls + ".cir")
            scheme_png_path = os.path.join(circuit_class_folder, cls + ".png")
            output_path = os.path.join(dataset_dir, measurement_variant["name"])

            # For each measurement setting and a circuit class
            # we need to create a specific circuit with elements parameters variations and save it to *.cir file
            changer = ParametersChanger(cir_path, parameters_settings)
            changer.generate_circuits()
            changer.dump_circuits_on_disk(output_path)

            # Create a simulator for the measurement settings
            simulator = SimulatorIVC(measurement_variant)

            # Get parameter combinations for all circuits
            params_combinations = changer._get_params_combinations(changer._settings)

            for i, (circuit, params_combination) in enumerate(zip(changer.circuits, params_combinations)):
                _process_single_circuit(circuit, params_combination, i, cls, changer, simulator,
                                        measurement_variant, output_path, scheme_png_path,
                                        save_png, disable_filtering)
