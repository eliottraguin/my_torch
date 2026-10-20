"""Parse and validate the chessboard analyzer command line."""

import argparse
import sys


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(84, f"{self.prog}: error: {message}\n")


def _create_parser():
    parser = _ArgumentParser(
        prog="./my_torch_analyzer",
        usage="%(prog)s [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE",
        description="Train a neural network or predict chessboard states.",
        allow_abbrev=False,
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--predict",
        dest="mode",
        action="store_const",
        const="predict",
        help="Predict each FEN chessboard; expected outputs are optional.",
    )
    modes.add_argument(
        "--train",
        dest="mode",
        action="store_const",
        const="train",
        help="Train using FEN chessboards followed by their expected outputs.",
    )
    parser.add_argument(
        "--save",
        metavar="SAVEFILE",
        help="Save the trained network to SAVEFILE instead of LOADFILE (training only).",
    )
    parser.add_argument(
        "loadfile",
        metavar="LOADFILE",
        help="File containing the neural network to load.",
    )
    parser.add_argument(
        "chessfile",
        metavar="CHESSFILE",
        help="File containing chessboards in FEN notation.",
    )
    return parser


def parse_args(argv):
    """Accept a complete argv, including the program name, and return a dict."""
    return vars(_create_parser().parse_args(argv[1:]))


def validate_args(args):
    """Validate the dependency between training mode and the save option."""
    if args["save"] is not None and args["mode"] != "train":
        _create_parser().error("--save only works in train mode")


USAGE = _create_parser().format_help().rstrip()
