import glob
import json
import os

from generate_dataset.parameters_changer import ParametersChanger
from generate_dataset.simulator_ivc import SimulatorIVC

BASE_CLASSES_FOLDER = "circuit_classes"
PARAMETERS_SETTINGS_PATH = "generate_dataset\\parameters_variations.json"
MEASUREMENTS_SETTINGS_PATH = "generate_dataset\\measurement_settings.json"
DEFAULT_DATASET_FOLDER = "dataset"


def generate_dataset(save_png=False, dataset_dir=None):
    """
    Generate dataset from circuit classes.

    Args:
        save_png: Whether to save PNG images for each dataset file
        dataset_dir: Output directory for dataset (default: "dataset")
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

            for i, (circuit, params_combination) in enumerate(zip(changer.circuits, params_combinations)):
                print(output_path, f"params{i:03d}")

                # Get original I-V curve
                original_ivc = simulator.get_ivc(circuit)

                # Generate bound circuits for comparison
                bound_circuits = changer.generate_bound_circuits(params_combination)

                # Check if original circuit differs enough from all bound circuits
                should_save = True
                min_difference_threshold = changer.min_difference_threshold

                for bound_circuit in bound_circuits:
                    bound_ivc = simulator.get_ivc(bound_circuit)
                    difference = simulator.compare_ivc(original_ivc, bound_ivc)

                    # If any bound circuit is too similar (difference <= threshold), skip this circuit
                    if difference <= min_difference_threshold:
                        should_save = False
                        break

                # Only save files if circuit differs enough from all bounds
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
                    print(f"Skipping circuit {i:03d} - too similar to boundary circuits (threshold: {min_difference_threshold})")
