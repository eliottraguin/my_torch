"""Command-line entry point for the chessboard analyzer."""

import sys

from .application import run
from .args import parse_args


def main(argv=None):
    """Parse a complete argv and translate application errors into exit code 84."""
    try:
        args = parse_args(sys.argv if argv is None else argv)
        run(args)
    except (OSError, ValueError, NotImplementedError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(84) from error


if __name__ == "__main__":
    main()
