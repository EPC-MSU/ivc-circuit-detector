import logging
import argparse
from generate_dataset.validate_circuit_classes import validate_circuit_classes

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]", level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset from `circuit classes` to `dataset` folder")
    parser.add_argument('-i', "--image", action="store_true", help="Add IVC-png-image to each dataset file")
    args = parser.parse_args()

    validate_circuit_classes()
    if args.image is None:
        pass
    else:
        pass
