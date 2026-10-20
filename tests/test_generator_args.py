"""Generator configuration/count pairs and argument failures."""

from unittest.mock import patch

from generator.args import parse_args, validate_args
from tests.support import CommandLineTestCase


class GeneratorArgumentsTests(CommandLineTestCase):
    def parse(self, *args):
        return parse_args(["my_torch_generator", *args])

    def test_single_configuration(self):
        self.assertEqual(self.parse(self.config, "3"),
                         {"configurations": [(self.config, 3)]})

    def test_multiple_configurations_preserve_order(self):
        self.assertEqual(
            self.parse(self.config, "2", self.other_config, "5"),
            {"configurations": [(self.config, 2), (self.other_config, 5)]},
        )

    def test_default_argv(self):
        with patch("sys.argv", ["my_torch_generator", self.config, "1"]):
            self.assertEqual(parse_args(), {"configurations": [(self.config, 1)]})

    def test_help_without_configuration(self):
        for flag in ("--help", "-h"):
            with self.subTest(flag=flag):
                self.assert_cli_help(lambda: self.parse(flag), "my_torch_generator")

    def test_no_arguments(self):
        self.assert_cli_error(lambda: self.parse(), "required")

    def test_incomplete_configuration_pairs(self):
        for args in [(self.config,), (self.config, "1", self.other_config)]:
            with self.subTest(args=args):
                self.assert_cli_error(lambda: self.parse(*args),
                                      "must be followed by a network count")

    def test_counts_must_be_integers(self):
        for count in ("abc", "1.5"):
            with self.subTest(count=count):
                self.assert_cli_error(lambda: self.parse(self.config, count),
                                      "must be an integer")

    def test_counts_must_be_positive(self):
        for count in ("0", "-1"):
            with self.subTest(count=count):
                self.assert_cli_error(lambda: self.parse(self.config, count),
                                      "must be greater than zero")

    def test_missing_configuration(self):
        self.assert_cli_error(lambda: self.parse(self.missing, "1"),
                              "does not exist")

    def test_missing_later_configuration_is_rejected(self):
        self.assert_cli_error(lambda: self.parse(self.config, "1", self.missing, "2"),
                              "does not exist")

    def test_directory_is_not_a_configuration_file(self):
        self.assert_cli_error(lambda: self.parse(str(self.directory), "1"),
                              "does not exist")

    def test_unknown_and_abbreviated_options(self):
        for option in ("--unknown", "--he"):
            with self.subTest(option=option):
                self.assert_cli_error(lambda: self.parse(self.config, "1", option),
                                      "unrecognized arguments")

    def test_validation_normalizes_counts(self):
        args = {"configurations": [(self.config, "2"), (self.other_config, 3)]}
        validate_args(args)
        self.assertEqual(args["configurations"], [(self.config, 2), (self.other_config, 3)])

    def test_validation_rejects_nonconvertible_count(self):
        self.assert_cli_error(
            lambda: validate_args({"configurations": [(self.config, None)]}),
            "must be an integer",
        )
