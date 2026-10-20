class FEN:
    """Class to handle FEN (Forsyth-Edwards Notation) operations."""

    def __init__(self, fen_string):
        """
        Initialize the FEN with the given FEN string.

        Args:
            fen_string (str): The FEN string to be parsed.
        """
        self.fen_string = fen_string
        self.piece_placement = None
        self.active_color = None
        self.castling_availability = None
        self.en_passant_target_square = None
        self.halfmove_clock = None
        self.fullmove_number = None
        self.label = None

    def parse(self):
        """Parse the FEN components and return this object."""
        fields = self.fen_string.split()
        if len(fields) < 6:
            raise ValueError(f"Invalid FEN format: {self.fen_string}")

        self.piece_placement = fields[0]
        self.active_color = fields[1]
        self.castling_availability = fields[2]
        self.en_passant_target_square = fields[3]
        self.halfmove_clock = fields[4]
        self.fullmove_number = fields[5]
        self.label = fields[6:] if len(fields) > 6 else None
        return self
