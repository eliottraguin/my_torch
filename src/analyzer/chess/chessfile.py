from .fen import FEN

class ChessFile:
    """Class to handle chess file operations."""

    def __init__(self, filename):
        """
        Initialize the ChessFile with the given filename.
        Parse the six FEN fields first, then the optional remaining label, including multiword labels.

        Args:
            filename (str): The path to the chess file.
        """
        self.filename = filename
        self.content = self.read()
        self.fen_games : list[FEN] = self.get_fen_games()

    def read(self):
        """Read the chess file and return its content."""
        with open(self.filename, 'r') as file:
            return file.read()

    def get_fen_games(self):
        """Parse the chess file content into FEN fields and optional label."""
        fen_games = []
        lines = self.content.strip().splitlines()
        for line in lines:
            try:
                fen_string = line.strip()
                fen = FEN(fen_string).parse()
            except ValueError as e:
                raise ValueError(f"Invalid FEN string in line: {line}") from e
            fen_games.append(fen)
        return fen_games
