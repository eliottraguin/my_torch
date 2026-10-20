"""Parse and validate the neural network generator command line."""

import argparse
import sys


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(84, f"{self.prog}: error: {message}\n")


def _create_parser():
    parser = _ArgumentParser(
        prog="./my_torch_generator",
        usage="%(prog)s config_file_1 nb_1 [config_file_2 nb_2 ...]",
        description="Generate neural networks from configuration files.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "configurations",
        nargs="+",
        metavar="CONFIG_FILE NB",
        help="Configuration file followed by a positive number of networks; repeat as needed.",
    )
    return parser


def parse_args(argv=None):
    """Parse sys.argv or a complete argv and return configuration/count pairs."""
    parser = _create_parser()
    values = parser.parse_args(None if argv is None else argv[1:]).configurations
    if len(values) % 2:
        parser.error("each configuration file must be followed by a network count")
    args = {"configurations": list(zip(values[::2], values[1::2]))}
    validate_args(args)
    return args


def validate_args(args):
    """Validate counts and normalize them to positive integers."""
    configurations = []
    for config_file, count in args["configurations"]:
        try:
            number = int(count)
        except (TypeError, ValueError):
            _create_parser().error(f"network count for '{config_file}' must be an integer")
        if number <= 0:
            _create_parser().error(f"network count for '{config_file}' must be greater than zero")
        configurations.append((config_file, number))
    args["configurations"] = configurations


USAGE = _create_parser().format_help().rstrip()
