"""Chess files must load FEN records without depending on the working directory."""

from pathlib import Path

from analyzer.chess.chessfile import ChessFile
from analyzer.chess.fen import FEN
from tests.support import CommandLineTestCase


POSITION = "8/8/8/8/8/8/8/K6k w - - 0 1"


class ChessFileTests(CommandLineTestCase):
    def test_empty_file(self):
        self.assertEqual(ChessFile(self.boards).fen_games, [])

    def test_records_preserve_optional_and_multiword_labels(self):
        Path(self.boards).write_text(
            f"{POSITION}\n{POSITION} Check\n{POSITION} Checkmate Black\n"
        )
        chessfile = ChessFile(self.boards)
        self.assertEqual(len(chessfile.fen_games), 3)
        self.assertEqual(
            [fen.label for fen in chessfile.fen_games],
            [None, ["Check"], ["Checkmate", "Black"]],
        )
        for fen in chessfile.fen_games:
            self.assertIsInstance(fen, FEN)
            self.assertEqual(fen.active_color, "w")
            self.assertEqual(fen.fullmove_number, "1")

    def test_repeated_parsing_does_not_duplicate_records(self):
        Path(self.boards).write_text(f"{POSITION} Check\n")
        chessfile = ChessFile(self.boards)
        self.assertEqual(len(chessfile.get_fen_games()), 1)
        self.assertEqual(len(chessfile.fen_games), 1)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            ChessFile(self.missing)

    def test_incomplete_record(self):
        Path(self.boards).write_text("8/8/8/8/8/8/8/K6k w -\n")
        with self.assertRaisesRegex(ValueError, "Invalid FEN string in line"):
            ChessFile(self.boards)
