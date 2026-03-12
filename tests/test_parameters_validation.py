import json
import os
import unittest
import jsonschema


class TestParametersValidation(unittest.TestCase):
    """Test validation of parameters_variations.json against its schema."""

    def setUp(self):
        """Set up test fixtures."""
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.parameters_file = os.path.join(self.base_dir, "generate_dataset", "parameters_variations.json")
        self.schema_file = os.path.join(self.base_dir, "generate_dataset", "parameters_variations_schema.json")

    def test_parameters_validates_against_schema(self):
        """Test that parameters_variations.json validates against its schema."""
        # Load schema
        with open(self.schema_file, "r") as f:
            schema = json.load(f)

        # Load parameters
        with open(self.parameters_file, "r") as f:
            parameters = json.load(f)

        # Validate - jsonschema will handle all the validation logic
        jsonschema.validate(parameters, schema)

    def test_single_element_list_validation(self):
        """Test that single-element lists validate correctly."""
        # Create test data with single-element list
        test_single_element = {
            "title": "Test Single Element",
            "description": "Test single element list validation",
            "version": "1.0.0",
            "filter": {
                "bounds_extension_percentage": 10.0,
                "min_difference_threshold": 0.05
            },
            "elements": {
                "R": [{
                    "_name": "resistance",
                    "_units": "Ohm",
                    "cir_key": None,
                    "cir_unit": "Ohm",
                    "nominal": {
                        "type": "list",
                        "value": [42.0]  # Single element
                    }
                }]
            }
        }

        # Load schema and validate
        with open(self.schema_file, "r") as f:
            schema = json.load(f)

        # Should validate successfully
        jsonschema.validate(test_single_element, schema)


    def _base_valid_config(self):
        return {
            "title": "Test",
            "description": "Test",
            "version": "1.0.0",
            "filter": {
                "bounds_extension_percentage": 10.0,
                "min_difference_threshold": 0.05
            },
            "elements": {
                "R": [{
                    "_name": "resistance",
                    "_units": "Ohm",
                    "cir_key": None,
                    "cir_unit": "Ohm",
                    "nominal": {"type": "list", "value": [1.0]}
                }]
            }
        }

    def test_missing_filter_section_fails_validation(self):
        """Test that config without 'filter' section fails schema validation."""
        test_data = self._base_valid_config()
        del test_data["filter"]

        with open(self.schema_file, "r") as f:
            schema = json.load(f)

        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(test_data, schema)

    def test_missing_bounds_extension_percentage_fails_validation(self):
        """Test that filter without 'bounds_extension_percentage' fails schema validation."""
        test_data = self._base_valid_config()
        del test_data["filter"]["bounds_extension_percentage"]

        with open(self.schema_file, "r") as f:
            schema = json.load(f)

        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(test_data, schema)

    def test_missing_min_difference_threshold_fails_validation(self):
        """Test that filter without 'min_difference_threshold' fails schema validation."""
        test_data = self._base_valid_config()
        del test_data["filter"]["min_difference_threshold"]

        with open(self.schema_file, "r") as f:
            schema = json.load(f)

        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(test_data, schema)


if __name__ == "__main__":
    unittest.main()
