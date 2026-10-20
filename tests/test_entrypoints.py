"""Main functions must pass command-line validation errors to the caller."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest.mock import patch

import analyzer
import generator
from tests.support import CommandLineTestCase


class EntryPointTests(CommandLineTestCase):
    def test_analyzer_valid_invocations(self):
        cases = [
            ["--predict", self.network, self.boards],
            ["--train", self.network, self.boards],
            ["--train", "--save", self.saved_network, self.network, self.boards],
        ]
        for args in cases:
            with self.subTest(args=args):
                stderr = StringIO()
                with patch("sys.argv", ["my_torch_analyzer", *args]), \
                        redirect_stdout(StringIO()), redirect_stderr(stderr):
                    self.assertIsNone(analyzer.main())
                self.assertEqual(stderr.getvalue(), "")

    def test_generator_valid_invocation(self):
        args = ["my_torch_generator", self.config, "2", self.other_config, "1"]
        stderr = StringIO()
        with patch("sys.argv", args), redirect_stdout(StringIO()), redirect_stderr(stderr):
            self.assertIsNone(generator.main())
        self.assertEqual(stderr.getvalue(), "")

    def test_entrypoints_help(self):
        for module, program in [(analyzer, "my_torch_analyzer"),
                                (generator, "my_torch_generator")]:
            with self.subTest(program=program), patch("sys.argv", [program, "--help"]):
                self.assert_cli_help(module.main, program)

    def test_entrypoints_reject_empty_argv(self):
        for module, program in [(analyzer, "my_torch_analyzer"),
                                (generator, "my_torch_generator")]:
            with self.subTest(program=program), patch("sys.argv", [program]):
                self.assert_cli_error(module.main, "required")

    def test_entrypoints_reject_missing_files(self):
        cases = [(analyzer, ["my_torch_analyzer", "--predict", self.missing, self.boards]),
                 (generator, ["my_torch_generator", self.missing, "1"])]
        for module, argv in cases:
            with self.subTest(argv=argv), patch("sys.argv", argv):
                self.assert_cli_error(module.main, "does not exist")
