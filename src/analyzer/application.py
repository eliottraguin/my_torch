"""Load analyzer inputs and dispatch once to the selected mode."""

from .chess.chessfile import ChessFile
from .neural_network import NeuralNetwork
from .prediction import run_prediction
from .training import run_training


def run(args):
    """Load the shared inputs once, then execute prediction or training."""
    mode = args["mode"]
    if mode not in ("predict", "train"):
        raise ValueError(f"Unknown analyzer mode: {mode}")

    network = NeuralNetwork.load_from_file(args["loadfile"])
    positions = ChessFile(args["chessfile"]).fen_games

    if mode == "predict":
        run_prediction(network, positions)
    else:
        save_path = args["save"] if args["save"] is not None else args["loadfile"]
        run_training(network, positions, save_path)
