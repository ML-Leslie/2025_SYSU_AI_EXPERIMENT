init_board = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 0),
)


def print_board(board):
    for row in board:
        print(" ".join(f"{num}" for num in row))


print_board(init_board)
