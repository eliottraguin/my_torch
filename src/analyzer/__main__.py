import sys
from analyzer.args import parse_args, validate_args, USAGE


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--help":
        print(USAGE)
        sys.exit(0)
    if len(sys.argv) == 1:
        print(USAGE, file=sys.stderr)
        sys.exit(84)

    args = parse_args(sys.argv)
    validate_args(args)

    print(f"Mode: {args['mode']}")
    print(f"Network file: {args['loadfile']}")
    print(f"Chess file: {args['chessfile']}")
    if args["save"]:
        print(f"Save to: {args['save']}")


if __name__ == "__main__":
    main()
