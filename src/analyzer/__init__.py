import sys
from analyzer.args import parse_args
from analyzer.chess.chessfile import ChessFile

def main():
    args = parse_args(sys.argv)
    try:
        chessfile = ChessFile(args["chessfile"])
    except FileNotFoundError:
        print(f"Error: chess file '{args['chessfile']}' does not exist.", file=sys.stderr)
        sys.exit(84)
    except ValueError as e:
        print(f"{e}", file=sys.stderr)
        sys.exit(84)


if __name__ == "__main__":
    main()
