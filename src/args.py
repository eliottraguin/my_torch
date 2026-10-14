import sys


USAGE = """USAGE
\t./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE

DESCRIPTION
\t--train\t\tLaunch the neural network in training mode. Each chessboard in
\t\t\tCHESSFILE must contain inputs in FEN notation and the expected
\t\t\toutput separated by space. If --save is specified, the newly
\t\t\ttrained network will be saved in SAVEFILE. Otherwise, it will
\t\t\tbe saved in LOADFILE.

\t--predict\tLaunch the neural network in prediction mode. Each chessboard
\t\t\tin CHESSFILE must contain inputs in FEN notation, and optionally
\t\t\tan expected output.

\t--save\t\tSave neural network into SAVEFILE. Only works in train mode.

\tLOADFILE\tFile containing an artificial neural network
\tCHESSFILE\tFile containing chessboards"""


def parse_args(argv):
    args = {
        "mode": None,
        "save": None,
        "loadfile": None,
        "chessfile": None,
    }
    i = 1
    while i < len(argv):
        if argv[i] == "--train":
            args["mode"] = "train"
        elif argv[i] == "--predict":
            args["mode"] = "predict"
        elif argv[i] == "--save":
            i += 1
            if i >= len(argv):
                print("Error: --save requires a filename", file=sys.stderr)
                sys.exit(84)
            args["save"] = argv[i]
        else:
            if args["loadfile"] is None:
                args["loadfile"] = argv[i]
            elif args["chessfile"] is None:
                args["chessfile"] = argv[i]
            else:
                print(f"Error: unexpected argument '{argv[i]}'", file=sys.stderr)
                sys.exit(84)
        i += 1
    return args


def validate_args(args):
    if args["mode"] is None:
        print("Error: --train or --predict is required", file=sys.stderr)
        sys.exit(84)
    if args["loadfile"] is None:
        print("Error: LOADFILE is required", file=sys.stderr)
        sys.exit(84)
    if args["chessfile"] is None:
        print("Error: CHESSFILE is required", file=sys.stderr)
        sys.exit(84)
    if args["save"] and args["mode"] != "train":
        print("Error: --save only works in train mode", file=sys.stderr)
        sys.exit(84)
