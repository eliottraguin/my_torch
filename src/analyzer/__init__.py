import sys
from analyzer.args import parse_args, validate_args, USAGE


def main():
    args = parse_args(sys.argv)

    print(f"Mode: {args['mode']}")
    print(f"Network file: {args['loadfile']}")
    print(f"Chess file: {args['chessfile']}")
    if args["save"]:
        print(f"Save to: {args['save']}")


if __name__ == "__main__":
    main()
