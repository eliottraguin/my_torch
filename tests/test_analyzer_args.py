"""Analyzer argument syntax, option dependencies and input paths."""

from analyzer.args import parse_args
from tests.support import CommandLineTestCase


class AnalyzerArgumentsTests(CommandLineTestCase):
    def parse(self, *args):
        return parse_args(["my_torch_analyzer", *args])

    def test_predict(self):
        self.assertEqual(
            self.parse("--predict", self.network, self.boards),
            {"mode": "predict", "save": None,
             "loadfile": self.network, "chessfile": self.boards},
        )

    def test_train_without_save(self):
        args = self.parse("--train", self.network, self.boards)
        self.assertEqual(args["mode"], "train")
        self.assertIsNone(args["save"])

    def test_train_with_new_save_path(self):
        args = self.parse("--train", "--save", self.saved_network,
                          self.network, self.boards)
        self.assertEqual(args["save"], self.saved_network)

    def test_save_after_positionals(self):
        args = self.parse("--train", self.network, self.boards,
                          "--save", self.saved_network)
        self.assertEqual(args["save"], self.saved_network)

    def test_help_without_input_files(self):
        for flag in ("--help", "-h"):
            with self.subTest(flag=flag):
                self.assert_cli_help(lambda: self.parse(flag), "my_torch_analyzer")

    def test_missing_arguments(self):
        cases = [(), ("--predict",), ("--train", self.network),
                 (self.network, self.boards)]
        for args in cases:
            with self.subTest(args=args):
                self.assert_cli_error(lambda: self.parse(*args), "required")

    def test_conflicting_modes_in_both_orders(self):
        for modes in [("--predict", "--train"), ("--train", "--predict")]:
            with self.subTest(modes=modes):
                self.assert_cli_error(
                    lambda: self.parse(*modes, self.network, self.boards),
                    "not allowed",
                )

    def test_unknown_and_abbreviated_options(self):
        for option in ("--unknown", "--pred", "--he"):
            with self.subTest(option=option):
                self.assert_cli_error(
                    lambda: self.parse("--predict", self.network, self.boards, option),
                    "unrecognized arguments",
                )

    def test_extra_positional(self):
        self.assert_cli_error(
            lambda: self.parse("--predict", self.network, self.boards, "extra"),
            "unrecognized arguments",
        )

    def test_save_requires_a_filename(self):
        self.assert_cli_error(
            lambda: self.parse("--train", self.network, self.boards, "--save"),
            "expected one argument",
        )

    def test_save_is_rejected_in_prediction(self):
        self.assert_cli_error(
            lambda: self.parse("--predict", "--save", self.saved_network,
                               self.network, self.boards),
            "--save only works in train mode",
        )

    def test_missing_network(self):
        self.assert_cli_error(
            lambda: self.parse("--predict", self.missing, self.boards),
            "LOADFILE does not exist",
        )

    def test_missing_chess_file(self):
        self.assert_cli_error(
            lambda: self.parse("--predict", self.network, self.missing),
            "CHESSFILE does not exist",
        )

    def test_directories_are_not_input_files(self):
        for network, boards, message in [
            (str(self.directory), self.boards, "LOADFILE"),
            (self.network, str(self.directory), "CHESSFILE"),
        ]:
            with self.subTest(message=message):
                self.assert_cli_error(
                    lambda: self.parse("--predict", network, boards), message,
                )
