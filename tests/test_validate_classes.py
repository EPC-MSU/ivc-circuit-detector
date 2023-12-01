from unittest import TestCase
from generate_dataset.validate_circuit_classes import validate_circuit_classes


class ClassesValidate(TestCase):
    def test_validate_circuit_classes(self):
        self.assertIsNone(validate_circuit_classes())
