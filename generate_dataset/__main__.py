import argparse
import logging

from generate_dataset.dataset_generator import generate_dataset
from generate_dataset.validate_circuit_classes import validate_circuit_classes, validate_measurement_settings

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]", level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset from `circuit_classes` to output folder")
    parser.add_argument("-i", "--image", action="store_true", help="Add IVC-png-image to each dataset file")
    parser.add_argument("--dataset-dir", default="dataset",
                        help="Output directory for generated dataset (default: dataset)")
    parser.add_argument("--disable-filtering", action="store_true",
                        help="Disable boundary condition filtering")
    args = parser.parse_args()

    validate_circuit_classes()
    validate_measurement_settings()

    filter_status = "disabled" if args.disable_filtering else "enabled"
    logging.info(f"Generating dataset to {args.dataset_dir} with filtering {filter_status}...")
    generate_dataset(save_png=args.image, dataset_dir=args.dataset_dir, disable_filtering=args.disable_filtering)
