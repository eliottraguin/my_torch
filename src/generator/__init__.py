import sys
from generator.args import parse_args, validate_args


def main():
    args = parse_args(sys.argv)

    for config_file, count in args["configurations"]:
        print(f"Configuration file: {config_file}")
        print(f"Number of networks: {count}")


if __name__ == "__main__":
    main()
