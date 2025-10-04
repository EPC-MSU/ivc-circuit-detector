import json
import os
import unittest

try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False


class TestParametersValidation(unittest.TestCase):
    """Test validation of parameters_variations.json against its schema."""

    def setUp(self):
        """Set up test fixtures."""
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.parameters_file = os.path.join(self.base_dir, "generate_dataset", "parameters_variations.json")
        self.schema_file = os.path.join(self.base_dir, "generate_dataset", "parameters_variations_schema.json")

    def test_parameters_file_exists(self):
        """Test that parameters_variations.json exists."""
        self.assertTrue(os.path.exists(self.parameters_file),
                       f"Parameters file not found: {self.parameters_file}")

    def test_schema_file_exists(self):
        """Test that parameters_variations_schema.json exists."""
        self.assertTrue(os.path.exists(self.schema_file),
                       f"Schema file not found: {self.schema_file}")

    def test_parameters_file_is_valid_json(self):
        """Test that parameters_variations.json is valid JSON."""
        with open(self.parameters_file, "r") as f:
            try:
                json.load(f)
            except json.JSONDecodeError as e:
                self.fail(f"Parameters file is not valid JSON: {e}")

    def test_schema_file_is_valid_json(self):
        """Test that parameters_variations_schema.json is valid JSON."""
        with open(self.schema_file, "r") as f:
            try:
                json.load(f)
            except json.JSONDecodeError as e:
                self.fail(f"Schema file is not valid JSON: {e}")

    @unittest.skipUnless(JSONSCHEMA_AVAILABLE, "jsonschema package not available")
    def test_parameters_validates_against_schema(self):
        """Test that parameters_variations.json validates against its schema."""
        # Load schema
        with open(self.schema_file, "r") as f:
            schema = json.load(f)

        # Load parameters
        with open(self.parameters_file, "r") as f:
            parameters = json.load(f)

        # Validate
        try:
            jsonschema.validate(parameters, schema)
        except jsonschema.ValidationError as e:
            self.fail(f"Parameters validation failed: {e.message}")
        except jsonschema.SchemaError as e:
            self.fail(f"Schema is invalid: {e.message}")

    def test_required_fields_present(self):
        """Test that all required top-level fields are present."""
        with open(self.parameters_file, "r") as f:
            parameters = json.load(f)

        required_fields = ["title", "description", "version", "elements"]
        for field in required_fields:
            self.assertIn(field, parameters, f"Required field '{field}' missing from parameters")

    def test_version_format(self):
        """Test that version follows semantic versioning format."""
        with open(self.parameters_file, "r") as f:
            parameters = json.load(f)

        version = parameters.get("version", "")
        # Check if version matches x.y.z pattern
        import re
        version_pattern = r"^\d+\.\d+\.\d+$"
        self.assertRegex(version, version_pattern,
                        f"Version '{version}' does not follow semantic versioning (x.y.z)")

    def test_element_types_structure(self):
        """Test that element types have correct structure."""
        with open(self.parameters_file, "r") as f:
            parameters = json.load(f)

        elements = parameters.get("elements", {})
        self.assertIsInstance(elements, dict, "Elements should be a dictionary")

        for element_type, element_params in elements.items():
            # Element type should be single uppercase letter
            self.assertRegex(element_type, r"^[A-Z]$",
                           f"Element type '{element_type}' should be single uppercase letter")

            # Element params should be array
            self.assertIsInstance(element_params, list,
                                f"Element '{element_type}' parameters should be a list")

            # Each parameter should have required fields
            for i, param in enumerate(element_params):
                required_param_fields = ["_name", "_units", "cir_key", "cir_unit", "nominal"]
                for field in required_param_fields:
                    self.assertIn(field, param,
                                f"Parameter {i} of element '{element_type}' missing field '{field}'")

    def test_nominal_parameter_types(self):
        """Test that nominal parameter types are valid."""
        with open(self.parameters_file, "r") as f:
            parameters = json.load(f)

        valid_types = ["uniform_interval", "exponential_interval", "list"]
        elements = parameters.get("elements", {})

        for element_type, element_params in elements.items():
            for i, param in enumerate(element_params):
                nominal = param.get("nominal", {})
                param_type = nominal.get("type")

                self.assertIn(param_type, valid_types,
                            f"Parameter {i} of element '{element_type}' has invalid type '{param_type}'")

                # Test type-specific requirements
                if param_type in ["uniform_interval", "exponential_interval"]:
                    self.assertIn("interval", nominal,
                                f"Interval parameter {i} of element '{element_type}' missing 'interval'")
                    self.assertIn("interval_points", nominal,
                                f"Interval parameter {i} of element '{element_type}' missing 'interval_points'")
                elif param_type == "list":
                    self.assertIn("value", nominal,
                                f"List parameter {i} of element '{element_type}' missing 'value'")
                    # Test that list values are not empty
                    values = nominal.get("value", [])
                    self.assertIsInstance(values, list,
                                        f"List parameter {i} of element '{element_type}' 'value' should be a list")
                    self.assertGreater(len(values), 0,
                                     f"List parameter {i} of element '{element_type}' 'value' should not be empty")

    def test_current_parameters_use_list_type(self):
        """Test that current parameters configuration uses 'list' type (as documented)."""
        with open(self.parameters_file, "r") as f:
            parameters = json.load(f)

        elements = parameters.get("elements", {})

        for element_type, element_params in elements.items():
            for i, param in enumerate(element_params):
                nominal = param.get("nominal", {})
                param_type = nominal.get("type")

                self.assertEqual(param_type, "list",
                               f"Parameter {i} of element '{element_type}' should use 'list' type, got '{param_type}'")

    def test_single_element_list_as_constant(self):
        """Test that single-element lists can act as constants."""
        with open(self.parameters_file, "r") as f:
            parameters = json.load(f)

        # Create test data with single-element list (acts as constant)
        test_single_element = {
            "title": "Test Single Element",
            "description": "Test single element list acting as constant",
            "version": "1.0.0",
            "elements": {
                "R": [{
                    "_name": "resistance",
                    "_units": "Ohm",
                    "cir_key": None,
                    "cir_unit": "Ohm",
                    "nominal": {
                        "type": "list",
                        "value": [42.0]  # Single element - acts as constant
                    }
                }]
            }
        }

        # Load schema and validate
        with open(self.schema_file, "r") as f:
            schema = json.load(f)

        try:
            if JSONSCHEMA_AVAILABLE:
                import jsonschema
                jsonschema.validate(test_single_element, schema)
            self.assertTrue(True)  # Test passes if no exception
        except Exception as e:
            self.fail(f"Single-element list validation failed: {e}")


if __name__ == "__main__":
    if not JSONSCHEMA_AVAILABLE:
        print("Warning: jsonschema package not available. Some tests will be skipped.")
        print("Install with: pip install jsonschema")

    unittest.main()