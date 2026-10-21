"""Verify the encoder's feature layout and malformed-input handling."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from analyzer import main
from analyzer.chess.chessencoder import ChessEncoder
from analyzer.chess.fen import FEN


BOARD = "4k3/8/8/8/8/8/4R3/4K3"
EMPTY_BOARD = "8/8/8/8/8/8/8/8"


class ChessEncoderTests(unittest.TestCase):
    def make_encoder(self, board=BOARD, turn="w", castling="-", ep="-",
                     halfmove="0", fullmove="12", label=""):
        fen = FEN(
            f"{board} {turn} {castling} {ep} {halfmove} {fullmove} {label}"
        ).parse()
        return ChessEncoder(fen)

    def test_empty_board_is_768_zeros(self):
        result = self.make_encoder(board=EMPTY_BOARD).encode_board()
        np.testing.assert_array_equal(result, np.zeros(768, dtype=np.float32))
        self.assertEqual(result.dtype, np.float32)

    def test_all_twelve_piece_channels(self):
        # These indices are the public feature contract, independent of PIECES.
        channels = [("P", 0), ("N", 1), ("B", 2), ("R", 3),
                    ("Q", 4), ("K", 5), ("p", 6), ("n", 7),
                    ("b", 8), ("r", 9), ("q", 10), ("k", 11)]
        for piece, channel in channels:
            with self.subTest(piece=piece):
                board = f"{piece}7/8/8/8/8/8/8/8"
                expected = np.zeros(768, dtype=np.float32)
                expected[channel] = 1
                np.testing.assert_array_equal(
                    self.make_encoder(board=board).encode_board(), expected,
                )

    def test_board_orientation_from_a8_to_h1(self):
        encoder = self.make_encoder(board="R6r/8/8/8/8/8/8/K6k")
        expected = np.zeros(768, dtype=np.float32)
        expected[0 * 12 + 3] = 1    # White rook a8.
        expected[7 * 12 + 9] = 1    # Black rook h8.
        expected[56 * 12 + 5] = 1   # White king a1.
        expected[63 * 12 + 11] = 1  # Black king h1.
        np.testing.assert_array_equal(encoder.encode_board(), expected)

    def test_mixed_pieces_and_empty_square_counts(self):
        encoder = self.make_encoder(board="8/8/8/1p1p2p1/8/8/8/8")
        expected = np.zeros((8, 8, 12), dtype=np.float32)
        expected[3, 1, 6] = 1
        expected[3, 3, 6] = 1
        expected[3, 6, 6] = 1
        np.testing.assert_array_equal(encoder.encode_board(), expected.reshape(-1))

    def test_initial_board_has_32_pieces(self):
        encoder = self.make_encoder(
            board="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
        )
        planes = encoder.encode_board().reshape(8, 8, 12)
        self.assertEqual(planes.sum(), 32)
        self.assertEqual(planes[0, 4, 11], 1)  # Black king e8.
        self.assertEqual(planes[7, 4, 5], 1)   # White king e1.
        self.assertTrue(np.all(planes.sum(axis=2) <= 1))

    def test_rejects_wrong_number_of_ranks(self):
        for count in (1, 7, 9):
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError, "8 ranks"):
                    self.make_encoder(board="/".join(["8"] * count)).encode_board()

    def test_rejects_wrong_square_counts(self):
        for rank in ("", "7", "88", "8P", "PPPPPPPPP"):
            with self.subTest(rank=rank):
                with self.assertRaisesRegex(ValueError, "square count"):
                    self.make_encoder(board=f"{rank}/8/8/8/8/8/8/8").encode_board()

    def test_rejects_unknown_piece_symbols(self):
        for symbol in ("X", "0", "9", "!"):
            with self.subTest(symbol=symbol):
                with self.assertRaises(ValueError):
                    self.make_encoder(board=f"{symbol}7/8/8/8/8/8/8/8").encode_board()

    def test_both_players_to_move(self):
        for turn, expected in (("w", 0), ("b", 1)):
            with self.subTest(turn=turn):
                result = self.make_encoder(turn=turn).encode_turn()
                np.testing.assert_array_equal(result, [expected])
                self.assertEqual(result.dtype, np.float32)

    def test_rejects_unknown_player(self):
        for turn in ("x", "W", "black"):
            with self.subTest(turn=turn):
                with self.assertRaisesRegex(ValueError, "active color"):
                    self.make_encoder(turn=turn).encode_turn()

    def test_no_en_passant_is_64_zeros(self):
        result = self.make_encoder().encode_en_passant()
        np.testing.assert_array_equal(result, np.zeros(64, dtype=np.float32))

    def test_en_passant_on_every_file_for_both_players(self):
        for turn, rank, row in (("w", "6", 2), ("b", "3", 5)):
            for column, file in enumerate("abcdefgh"):
                with self.subTest(turn=turn, square=file + rank):
                    expected = np.zeros(64, dtype=np.float32)
                    expected[row * 8 + column] = 1
                    result = self.make_encoder(turn=turn, ep=file + rank).encode_en_passant()
                    np.testing.assert_array_equal(result, expected)
                    self.assertEqual(result.dtype, np.float32)

    def test_rejects_wrong_en_passant_length(self):
        for square in ("a", "d6junk", "--"):
            with self.subTest(square=square):
                with self.assertRaisesRegex(ValueError, "en passant"):
                    self.make_encoder(ep=square).encode_en_passant()

    def test_rejects_unknown_en_passant_file(self):
        for square in ("A6", "i6", "z6", "16"):
            with self.subTest(square=square):
                with self.assertRaisesRegex(ValueError, "en passant"):
                    self.make_encoder(ep=square).encode_en_passant()

    def test_rejects_invalid_en_passant_rank(self):
        for square in ("a0", "a9", "a4", "aa", "a²"):
            with self.subTest(square=square):
                with self.assertRaisesRegex(ValueError, "en passant"):
                    self.make_encoder(ep=square).encode_en_passant()

    def test_en_passant_rank_must_match_player(self):
        for turn, square in (("w", "d3"), ("b", "d6")):
            with self.subTest(turn=turn, square=square):
                with self.assertRaisesRegex(ValueError, "active color"):
                    self.make_encoder(turn=turn, ep=square).encode_en_passant()

    def test_all_16_castling_combinations(self):
        for flags in product((0, 1), repeat=4):
            rights = "".join(right for right, flag in zip("KQkq", flags) if flag) or "-"
            with self.subTest(rights=rights):
                result = self.make_encoder(castling=rights).encode_castling()
                np.testing.assert_array_equal(result, flags)
                self.assertEqual(result.dtype, np.float32)

    def test_rejects_unknown_or_mixed_castling_rights(self):
        for rights in ("Kz", "x", "-K", "K-", "--"):
            with self.subTest(rights=rights):
                with self.assertRaises(ValueError):
                    self.make_encoder(castling=rights).encode_castling()

    def test_rejects_duplicate_castling_rights(self):
        for rights in ("KK", "qq", "KQK", "KQkqk"):
            with self.subTest(rights=rights):
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    self.make_encoder(castling=rights).encode_castling()

    def test_nonnegative_halfmove_counters(self):
        for count in (0, 1, 37, 100, 150):
            with self.subTest(count=count):
                result = self.make_encoder(halfmove=str(count)).encode_halfmove_clock()
                np.testing.assert_array_equal(result, [count])
                self.assertEqual(result.dtype, np.float32)
                self.assertTrue(np.isfinite(result).all())

    def test_rejects_invalid_halfmove_counters(self):
        for count in ("-1", "1.5", "nan", "inf", "abc", "9" * 40):
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError, "halfmove clock"):
                    self.make_encoder(halfmove=count).encode_halfmove_clock()

    def test_complete_feature_layout_shape_and_dtype(self):
        encoder = self.make_encoder(
            board="r3k2r/8/8/3pP3/8/8/8/R3K2R", castling="Kq", ep="d6",
        )
        expected = np.zeros(838, dtype=np.float32)
        # Explicit square indices in a8..h1 order and piece channel indices.
        for square, channel in [(0, 9), (4, 11), (7, 9), (27, 6),
                                (28, 0), (56, 3), (60, 5), (63, 3)]:
            expected[square * 12 + channel] = 1
        expected[768] = 0       # White to move.
        expected[769 + 19] = 1  # En passant target d6.
        expected[833] = 1       # White kingside castling.
        expected[836] = 1       # Black queenside castling.
        expected[837] = 0       # Halfmove clock.
        result = encoder.encode()
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.shape, (838,))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.isfinite(result).all())

    def test_halfmove_counter_is_the_last_feature(self):
        result = self.make_encoder(halfmove="37").encode()
        self.assertEqual(result[837], 37)

    def test_encoding_is_deterministic_and_returns_independent_arrays(self):
        encoder = self.make_encoder()
        first = encoder.encode()
        expected = first.copy()
        np.testing.assert_array_equal(encoder.encode(), expected)
        first[:] = -1
        np.testing.assert_array_equal(encoder.encode(), expected)

    def test_encoding_does_not_modify_fen(self):
        encoder = self.make_encoder(label="Check")
        original = vars(encoder.fen_game).copy()
        encoder.encode()
        self.assertEqual(vars(encoder.fen_game), original)

    def test_optional_labels_never_change_features(self):
        expected = self.make_encoder().encode()
        for label in ("Nothing", "Check", "Checkmate Black"):
            with self.subTest(label=label):
                np.testing.assert_array_equal(self.make_encoder(label=label).encode(), expected)

    def test_fullmove_number_is_not_an_input_feature(self):
        np.testing.assert_array_equal(
            self.make_encoder(fullmove="1").encode(),
            self.make_encoder(fullmove="500").encode(),
        )

    def test_reported_checkmate_dataset_encodes_without_errors(self):
        fixture = Path(__file__).parent / "dataset_example/checkmate/10_pieces.txt"
        records = [line for line in fixture.read_text().splitlines() if line.strip()]
        self.assertGreater(len(records), 0)
        for number, line in enumerate(records, start=1):
            with self.subTest(line=number):
                result = ChessEncoder(FEN(line).parse()).encode()
                self.assertEqual(result.shape, (838,))
                self.assertEqual(result.dtype, np.float32)
                self.assertTrue(np.isfinite(result).all())

    def test_invalid_encoder_inputs_exit_84_at_cli_boundary(self):
        cases = [
            f"{BOARD} w - a0 0 12",
            f"{BOARD} w KK - 0 12",
            f"{BOARD} w - - nan 12",
            "8P/8/8/8/8/8/8/8 w - - 0 12",
        ]
        with TemporaryDirectory(prefix="mytorch encoder tests ") as directory:
            network = Path(directory) / "network.nn"
            chessfile = Path(directory) / "positions.txt"
            network.touch()
            for line in cases:
                with self.subTest(fen=line):
                    chessfile.write_text(line + "\n")
                    stdout, stderr = StringIO(), StringIO()
                    argv = ["my_torch_analyzer", "--predict", str(network), str(chessfile)]
                    with patch("sys.argv", argv), redirect_stdout(stdout), redirect_stderr(stderr):
                        with self.assertRaises(SystemExit) as error:
                            main()
                    self.assertEqual(error.exception.code, 84)
                    self.assertEqual(stdout.getvalue(), "")
                    self.assertTrue(stderr.getvalue())
