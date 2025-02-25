'''
    This class is responsible for storing all the information about the current state of the chess game.
    It will also determine any valid moves given any position, while keeping a move log.
'''

class GameState():
    def __init__(self): 
        # for more efficiency in the future, using a numpy array may be better. Also good for training an AI model
        # A list of lists, representing each row in the chess board.
        # Board is an 8x8 2-D where pieces are represented by first their color (character 1) and their piece type (character 2)
        # Empty space represented by "--"
        self.board = [
            ["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"],
            ["bp", "bp", "bp", "bp", "bp", "bp", "bp", "bp"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["wp", "wp", "wp", "wp", "wp", "wp", "wp", "wp"],
            ["wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"],
        ]

        self.whiteToMove = True
        # In the current state, white goes first
        self.moveLog = []