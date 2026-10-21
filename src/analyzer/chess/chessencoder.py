""" Chess board encoder. """

import numpy as np



class ChessEncoder:
    """ Chess board encoder. """

    PIECES = "PNBRQKpnbrqk"

    def __init__(self, fen_game):
        self.fen_game = fen_game

    def encode_board(self):
        board = np.zeros((8, 8, 12), dtype=np.float32)
        ranks = self.fen_game.piece_placement.split("/")
        if len(ranks) != 8:
            raise ValueError("A FEN board must contain exactly 8 ranks")

        for row, rank in enumerate(ranks):
            col = 0
            square_count = sum(
                int(symbol) if symbol in "12345678" else 1
                for symbol in rank
            )
            if square_count != 8:
                raise ValueError(f"Invalid square count in FEN rank: {rank}")

            for symbol in rank:
                if symbol in "12345678":
                    col += int(symbol)
                else:
                    piece_index = self.PIECES.index(symbol)
                    board[row, col, piece_index] = 1.0
                    col += 1
        return board.reshape(-1)

    def encode_turn(self):
        """ Encode the turn into a single value. """
        if self.fen_game.active_color not in ["w", "b"]:
            raise ValueError(f"Invalid active color in FEN: {self.fen_game.active_color}")
        return np.array([0.0 if self.fen_game.active_color == "w" else 1.0], dtype=np.float32)

    def encode_en_passant(self):
        """ Encode the en passant target square into a one-hot encoded array. """
        en_passant = np.zeros(64, dtype=np.float32)
        square = self.fen_game.en_passant_target_square
        if square != "-":
            if len(square) != 2:
                raise ValueError(f"Invalid en passant target square: {square}")
            if not square[0].islower() or not square[1].isdigit():
                raise ValueError(f"Invalid en passant target square: {square}")
            if square[0] not in "abcdefgh":
                raise ValueError(f"Invalid en passant target square: {square}")
            expected_rank = "6" if self.fen_game.active_color == "w" else "3"
            if square[1] != expected_rank:
                raise ValueError(f"Invalid en passant target square: {square} for active color {self.fen_game.active_color}")
            col = ord(square[0]) - ord("a")
            row = 8 - int(square[1])
            en_passant[row * 8 + col] = 1.0
        return en_passant

    def encode_castling(self):
        """ Encode the castling availability into a one-hot encoded array. """
        castling = np.zeros(4, dtype=np.float32)
        if not all(c in "KQkq" for c in self.fen_game.castling_availability) and self.fen_game.castling_availability != "-":
            raise ValueError(f"Invalid castling availability in FEN: {self.fen_game.castling_availability}")
        if len(set(self.fen_game.castling_availability)) != len(self.fen_game.castling_availability):
            raise ValueError("Duplicate castling rights")
        if "K" in self.fen_game.castling_availability:
            castling[0] = 1.0
        if "Q" in self.fen_game.castling_availability:
            castling[1] = 1.0
        if "k" in self.fen_game.castling_availability:
            castling[2] = 1.0
        if "q" in self.fen_game.castling_availability:
            castling[3] = 1.0
        return castling

    def encode_halfmove_clock(self):
        """ Encode the halfmove clock into a single value. """
        if not self.fen_game.halfmove_clock.isdigit() or int(self.fen_game.halfmove_clock) < 0 or int(self.fen_game.halfmove_clock) > 250:
            raise ValueError(f"Invalid halfmove clock in FEN: {self.fen_game.halfmove_clock}")
        return np.array([float(self.fen_game.halfmove_clock)], dtype=np.float32)

    def encode(self):
        """ Encode the chess board into a string representation. """
        pieces = self.encode_board()
        turn = self.encode_turn()
        en_passant = self.encode_en_passant()
        castling = self.encode_castling()
        halfmove_clock = self.encode_halfmove_clock()

        return np.concatenate((pieces, turn, en_passant, castling, halfmove_clock))