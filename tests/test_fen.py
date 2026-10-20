"""FEN parsing must return usable objects, with or without optional labels."""

from contextlib import redirect_stdout
from io import StringIO
import unittest

from analyzer.chess.fen import FEN


POSITION = "8/8/8/8/8/8/8/K6k w - - 0 1"


class FENTests(unittest.TestCase):
    def test_parse_returns_the_same_object(self):
        fen = FEN(f"{POSITION} Checkmate Black")
        self.assertIs(fen.parse(), fen)

    def test_six_fields_without_label(self):
        fen = FEN(POSITION).parse()
        self.assertEqual(fen.piece_placement, "8/8/8/8/8/8/8/K6k")
        self.assertEqual(fen.active_color, "w")
        self.assertEqual(fen.castling_availability, "-")
        self.assertEqual(fen.en_passant_target_square, "-")
        self.assertEqual(fen.halfmove_clock, "0")
        self.assertEqual(fen.fullmove_number, "1")
        self.assertIsNone(fen.label)

    def test_optional_labels_preserve_all_words(self):
        for label in ("Check", "Checkmate Black"):
            with self.subTest(label=label):
                fen = FEN(f"{POSITION} {label}").parse()
                self.assertEqual(fen.label, label.split())

    def test_incomplete_fen_is_rejected(self):
        for fen_string in ("", "8/8/8/8/8/8/8/K6k w - - 0"):
            with self.subTest(fen=fen_string):
                with self.assertRaisesRegex(ValueError, "Invalid FEN format"):
                    FEN(fen_string).parse()

    def test_display_after_parsing(self):
        for label in (None, "Checkmate Black"):
            with self.subTest(label=label):
                fen = FEN(POSITION if label is None else f"{POSITION} {label}").parse()
                output = StringIO()
                with redirect_stdout(output):
                    fen.display()
                self.assertIn("Active Color: w", output.getvalue())
                self.assertIn("Fullmove Number: 1", output.getvalue())
                if label:
                    self.assertIn(f"Label: {label}", output.getvalue())
                else:
                    self.assertNotIn("Label:", output.getvalue())
