"""Shared real-file fixtures and command-line assertions."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


def write_test_network(path):
    """Write a deterministic model compatible with the 838-feature encoder."""
    Path(path).write_text(json.dumps({
        "layers": [838, 3], "learning_rate": 0.01,
        "labels": ["Checkmate", "Check", "Nothing"],
        "weights": [[[0.0] * 838 for _ in range(3)]],
        "biases": [[0.0, 1.0, -1.0]],
    }))


class CommandLineTestCase(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="mytorch tests ")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.network = str(self.directory / "network.nn")
        write_test_network(self.network)
        self.boards = str(self.create_file("boards.txt"))
        self.config = str(self.create_file("network.conf"))
        self.other_config = str(self.create_file("other.conf"))
        self.saved_network = str(self.directory / "saved.nn")
        self.missing = str(self.directory / "missing")

    def create_file(self, name):
        path = self.directory / name
        # These tests validate CLI arguments, not model or FEN file contents.
        path.touch()
        return path

    def assert_cli_error(self, callback, message):
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as error:
                callback()
        self.assertEqual(error.exception.code, 84)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn(message, stderr.getvalue())

    def assert_cli_help(self, callback, program):
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as error:
                callback()
        self.assertEqual(error.exception.code, 0)
        self.assertIn(program, stdout.getvalue())
        self.assertIn("usage:", stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")
