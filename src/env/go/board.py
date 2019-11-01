import numpy

class Board:

    def __init__(self, size = 19):
        self.size          = size
        self.reset()

    def reset(self):
        self.active_player = 1

        self.board = numpy.zeros((self.size, self.size))
        self.board_1 = self.board.copy()
        self.board_2 = self.board.copy()
        self.board_3 = self.board.copy()

        self.compute_legal_moves()

    def _print(self):

        for y in range(self.size):
            for x in range(self.size):
                v = self.board[y][x]
                if v > 0:
                    print("B", end = "")
                if v < 0:
                    print("W", end = "")
                if v == 0:
                    print(".", end = "")
                print(" ", end = "")
            print("\n", end = "")
        print("\n", end = "")

    def compute_legal_moves(self):
        self.legal_moves = []

        empty_fields = numpy.where(self.board == 0)
        empty_fields = numpy.column_stack((empty_fields[0], empty_fields[1]))

        for empty_field in empty_fields:
            self.legal_moves.append((empty_field[0], empty_field[1]))

    def compute_freedoms(self):
        for y in range(self.size):
            for x in range(self.size):

    def play_move(self, move):
        if move in self.legal_moves:
            self.board[move[0]][move[1]] = self.active_player

            if self.active_player == 1:
                self.active_player = -1
            else:
                self.active_player = 1

            self.compute_legal_moves()
            return 0
        else:
            return -1

    def get_legal_moves(self):
        return self.legal_moves


board = Board(9)

 
while len(board.get_legal_moves()) != 0:   
    legal_moves = board.get_legal_moves()
    move_id = numpy.random.randint(len(legal_moves))
    res = board.play_move(legal_moves[move_id])

    board._print()

    print("move result = ", res, len(board.get_legal_moves()))