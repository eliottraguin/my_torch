import sys
from analyzer.args import parse_args
from analyzer.chess.chessfile import ChessFile
from analyzer.chess.chessencoder import ChessEncoder
from analyzer.neural_network import NeuralNetwork

def core(args):
    neural_network = NeuralNetwork.load_from_file(args["loadfile"])
    chessfile = ChessFile(args["chessfile"])
    for i, fen_game in enumerate(chessfile.fen_games):
        chessencoder = ChessEncoder(fen_game)
        encoded = chessencoder.encode()
        prediction, confidence = neural_network.predict(encoded)
        print(f"FEN game {i}: Prediction: {prediction}, Confidence: {confidence}")


def main():
    args = parse_args(sys.argv)
    try:
        core(args)
    except FileNotFoundError:
        print(f"Error: chess file '{args['chessfile']}' does not exist.", file=sys.stderr)
        sys.exit(84)
    except ValueError as e:
        print(f"{e}", file=sys.stderr)
        sys.exit(84)


if __name__ == "__main__":
    main()
