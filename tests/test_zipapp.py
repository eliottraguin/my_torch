"""Exercise the packaged analyzer with no source-directory import path."""

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from tests.support import write_test_network


class AnalyzerZipappTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = TemporaryDirectory(prefix="mytorch zipapp ")
        cls.addClassCleanup(temporary.cleanup)
        cls.directory = Path(temporary.name)
        cls.archive = cls.directory / "my_torch_analyzer"
        source = Path(__file__).resolve().parents[1] / "src"
        subprocess.run(
            [sys.executable, "-m", "zipapp", str(source),
             "-m", "analyzer:main", "-o", str(cls.archive)],
            check=True, capture_output=True, text=True,
        )

    def run_analyzer(self, *args):
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        return subprocess.run(
            [sys.executable, str(self.archive), *args],
            cwd=self.directory, env=environment,
            capture_output=True, text=True, timeout=10,
        )

    def test_no_arguments_reports_usage_without_traceback(self):
        result = self.run_analyzer()
        self.assertEqual(result.returncode, 84, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertIn("usage:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_help_loads_packaged_modules(self):
        result = self.run_analyzer("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage:", result.stdout)
        self.assertEqual(result.stderr, "")

    def test_prediction_loads_a_chess_file(self):
        network = self.directory / "network.nn"
        boards = self.directory / "boards.txt"
        write_test_network(network)
        boards.write_text("8/8/8/8/8/8/8/K6k w - - 0 1\n")
        result = self.run_analyzer("--predict", str(network), str(boards))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")

    def test_prediction_loads_labeled_records_with_mode_last(self):
        network = self.directory / "labeled-network.nn"
        boards = self.directory / "labeled-boards.txt"
        write_test_network(network)
        boards.write_text("8/8/8/8/8/8/8/K6k w - - 0 1 Checkmate Black\n")
        result = self.run_analyzer(str(network), str(boards), "--predict")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")

    def test_incomplete_fen_reports_an_error_without_traceback(self):
        network = self.directory / "invalid-network.nn"
        boards = self.directory / "invalid-boards.txt"
        write_test_network(network)
        boards.write_text("8/8/8/8/8/8/8/K6k w -\n")
        result = self.run_analyzer(str(network), str(boards), "--predict")
        self.assertEqual(result.returncode, 84, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertIn("Invalid FEN", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_training_uses_its_own_handler_and_reports_exit_84(self):
        network = self.directory / "training-network.nn"
        boards = self.directory / "training-boards.txt"
        output = self.directory / "trained.nn"
        write_test_network(network)
        original = network.read_bytes()
        boards.write_text("8/8/8/8/8/8/8/K6k w - - 0 1 Nothing\n")
        result = self.run_analyzer("--train", "--save", str(output), str(network), str(boards))
        self.assertEqual(result.returncode, 84)
        self.assertEqual(result.stdout, "")
        self.assertIn("Training is not implemented", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(network.read_bytes(), original)
        self.assertFalse(output.exists())
