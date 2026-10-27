"""Check mode routing, prediction ordering and CLI error handling."""

from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

import numpy as np

import analyzer
from analyzer.application import run
from analyzer.chess.chessencoder import ChessEncoder
from analyzer.chess.fen import FEN
from analyzer.prediction import run_prediction
from tests.support import CommandLineTestCase


class ApplicationTests(CommandLineTestCase):
    def setUp(self):
        super().setUp()
        self.options = {"mode": "predict", "loadfile": self.network,
                        "chessfile": self.boards, "save": None}
        self.positions = [object(), object()]
        self.loaded_network = Mock()
        patches = {
            "load_network": patch("analyzer.application.NeuralNetwork.load_from_file",
                                  return_value=self.loaded_network),
            "load_positions": patch("analyzer.application.ChessFile",
                                    return_value=SimpleNamespace(fen_games=self.positions)),
            "predict": patch("analyzer.application.run_prediction"),
            "train": patch("analyzer.application.run_training"),
        }
        self.calls = {}
        for name, patcher in patches.items():
            self.calls[name] = patcher.start()
            self.addCleanup(patcher.stop)

    def test_prediction_loads_inputs_once_and_calls_only_prediction(self):
        run(self.options)
        self.calls["load_network"].assert_called_once_with(self.network)
        self.calls["load_positions"].assert_called_once_with(self.boards)
        self.calls["predict"].assert_called_once_with(self.loaded_network, self.positions)
        self.calls["train"].assert_not_called()

    def test_training_defaults_to_source_model_destination(self):
        self.options["mode"] = "train"
        run(self.options)
        self.calls["train"].assert_called_once_with(
            self.loaded_network, self.positions, self.network,
        )
        self.calls["predict"].assert_not_called()

    def test_training_passes_explicit_save_destination(self):
        self.options.update(mode="train", save=self.saved_network)
        run(self.options)
        self.calls["train"].assert_called_once_with(
            self.loaded_network, self.positions, self.saved_network,
        )
        self.calls["predict"].assert_not_called()

    def test_unknown_mode_is_rejected_before_reading_files(self):
        self.options["mode"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Unknown analyzer mode"):
            run(self.options)
        for mocked in self.calls.values():
            mocked.assert_not_called()

    def test_loading_error_prevents_both_workflows(self):
        self.calls["load_network"].side_effect = ValueError("Invalid model")
        with self.assertRaisesRegex(ValueError, "Invalid model"):
            run(self.options)
        self.calls["load_positions"].assert_not_called()
        self.calls["predict"].assert_not_called()
        self.calls["train"].assert_not_called()


class PredictionTests(unittest.TestCase):
    def test_predictions_keep_position_order_and_display_format(self):
        positions = [
            FEN("8/8/8/8/8/8/8/K6k w - - 0 1 Nothing").parse(),
            FEN("4k3/8/8/8/8/8/4R3/4K3 b - - 0 12 Check").parse(),
        ]
        network = Mock()
        network.predict.side_effect = [("Nothing", 0.75), ("Check", 0.9)]
        stdout = StringIO()
        with redirect_stdout(stdout):
            run_prediction(network, positions)
        self.assertEqual(stdout.getvalue().splitlines(), [
            "FEN game 0: Prediction: Nothing, Confidence: 0.75",
            "FEN game 1: Prediction: Check, Confidence: 0.9",
        ])
        self.assertEqual(network.predict.call_count, 2)
        for call, position in zip(network.predict.call_args_list, positions):
            np.testing.assert_array_equal(call.args[0], ChessEncoder(position).encode())

    def test_empty_input_produces_no_predictions(self):
        network = Mock()
        stdout = StringIO()
        with redirect_stdout(stdout):
            run_prediction(network, [])
        self.assertEqual(stdout.getvalue(), "")
        network.predict.assert_not_called()


class MainTests(CommandLineTestCase):
    def test_main_parses_and_delegates(self):
        argv = ["my_torch_analyzer", "--predict", self.network, self.boards]
        with patch("analyzer.run") as application:
            self.assertIsNone(analyzer.main(argv))
        application.assert_called_once_with({
            "mode": "predict", "loadfile": self.network,
            "chessfile": self.boards, "save": None,
        })

    def test_application_errors_are_reported_on_stderr_with_exit_84(self):
        argv = ["my_torch_analyzer", "--predict", self.network, self.boards]
        for error in (PermissionError("Cannot read model"),
                      ValueError("Invalid position"),
                      NotImplementedError("Training unavailable")):
            with self.subTest(error=type(error).__name__), patch("analyzer.run", side_effect=error):
                self.assert_cli_error(lambda: analyzer.main(argv), str(error))

    def test_help_does_not_start_application(self):
        with patch("analyzer.run") as application:
            self.assert_cli_help(lambda: analyzer.main(["my_torch_analyzer", "--help"]),
                                 "my_torch_analyzer")
        application.assert_not_called()
