import sys
from generator.args import parse_args
from generator.configfile import ConfigFile
from generator.neuralfile import NeuralFile


def main():
    args = parse_args(sys.argv)

    for config_file, count in args["configurations"]:
        config = ConfigFile(config_file)
        neural_file = NeuralFile(config)

        for _ in range(count):
            neural_file.create_file()


if __name__ == "__main__":
    main()
