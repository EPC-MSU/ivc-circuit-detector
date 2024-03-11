import argparse
import logging

from generate_dataset.dataset_generator import generate_dataset
from generate_dataset.validate_circuit_classes import (
    validate_circuit_classes, validate_measurement_settings)

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]", level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset from `circuit_classes` to `dataset` folder")
    parser.add_argument('-i', "--image", action="store_true", help="Add IVC-png-image to each dataset file")
    parser.add_argument('-d', "--debug", action="store_true", help="Don't use this flag please!!!")
    args = parser.parse_args()

    validate_circuit_classes()
    validate_measurement_settings()
    if args.image:
        logging.info('Generating dataset with images...')
        generate_dataset(save_png=True, debug=args.debug)
    else:
        logging.info('Generating dataset...')
        generate_dataset(save_png=False, debug=args.debug)
