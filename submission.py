import logic
import random
from AbstractPlayers import *
from math import log
import time

# commands to use for move players. dictionary : Move(enum) -> function(board),
# all the functions {up,down,left,right) receive board as parameter and return tuple of (new_board, done, score).
# new_board is according to the step taken, done is true if the step is legal, score is the sum of all numbers that
# combined in this step.
# (you can see GreedyMovePlayer implementation for example)
from constants import *

commands = {Move.UP: logic.up, Move.DOWN: logic.down,
            Move.LEFT: logic.left, Move.RIGHT: logic.right}


# generate value between {2,4} with probability p for 4
def gen_value(p=PROBABILITY):
    return logic.gen_two_or_four(p)


class GreedyMovePlayer(AbstractMovePlayer):
    """Greedy move player provided to you (no need to change),
    the player receives time limit for a single step and the board as parameter and return the next move that gives
    the best score by looking one step ahead.
    """

    def get_move(self, board, time_limit) -> Move:
        optional_moves_score = {}
        for move in Move:
            new_board, done, score = commands[move](board)
            if done:
                optional_moves_score[move] = score

        return max(optional_moves_score, key=optional_moves_score.get)


class HeuristicFunction:
    def __init__(self):
        self.w_empty = 2.7
        self.w_max_val = 1
        self.w_monotonic = 1
        self.w_same_tiles = 1

    def calc_heuristic(self, board):
        empty = self.w_empty * self._empty_cells(board)
        max_val = self.w_max_val * self._max_value(board)
        mono_val = self.w_monotonic * self._monotonic_board(board)
        same_tiles_val = self.w_same_tiles * self._same_tiles(board)

        return empty + mono_val + max_val + same_tiles_val

    def _empty_cells(self, board) -> float:
        empty_cells = 0
        for i in range(GRID_LEN):
            for j in range(GRID_LEN):
                if board[i][j] == 0:
                    empty_cells += 1

        if empty_cells == 0:
            return 1
        res = log(empty_cells)
        return res

    def _max_value(self, board) -> float:
        max_val = 0
        for i in range(GRID_LEN):
            for j in range(GRID_LEN):
                if board[i][j] > max_val:
                    max_val = board[i][j]
        return log(max_val)

    def _monotonic_board(self, board):
        left_to_right = 0
        right_to_left = 0
        up_to_down = 0
        down_to_up = 0
        val_curr = 0
        val_next = 0
        for x in range(GRID_LEN):
            for y in range(GRID_LEN - 1):
                # horizontal
                val_curr = board[x][y]
                val_next = board[x][y + 1]
                if val_curr != 0 or val_next != 0:
                    if val_curr >= val_next:
                        left_to_right += 1
                    if val_next >= val_curr:
                        right_to_left += 1

                    # vertical
                val_curr = board[y][x]
                val_next = board[y + 1][x]
                if val_curr != 0 or val_next != 0:
                    if val_curr >= val_next:
                        up_to_down += 1
                    if val_next >= val_curr:
                        down_to_up += 1
        res = max(right_to_left, left_to_right) + max(down_to_up, up_to_down)
        if res == 0:
            return 0
        return log(res)

    def _same_tiles(self, board):
        counter = 0
        for row in range(GRID_LEN):
            for col in range(GRID_LEN - 1):
                if board[row][col] == board[row][col + 1] and board[row][col] != 0:
                    counter += 1
                    col += 1
        for col in range(GRID_LEN):
            for row in range(GRID_LEN - 1):
                if board[row][col] == board[row + 1][col] and board[row][col] != 0:
                    counter += 1
                    row += 1
        if counter == 0:
            return 0
        return log(counter)


class RandomIndexPlayer(AbstractIndexPlayer):
    """Random index player provided to you (no need to change),
    the player receives time limit for a single step and the board as parameter and return the next indices to
    put 2 randomly.
    """

    def get_indices(self, board, value, time_limit) -> (int, int):
        a = random.randint(0, len(board) - 1)
        b = random.randint(0, len(board) - 1)
        while board[a][b] != 0:
            a = random.randint(0, len(board) - 1)
            b = random.randint(0, len(board) - 1)
        return a, b


# part A
class ImprovedGreedyMovePlayer(AbstractMovePlayer):
    """Improved greedy Move Player,
    implement get_move function with greedy move that looks only one step ahead with heuristic.
    (you can add helper functions as you want).
    """

    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # TODO: add here if needed

    def get_move(self, board, time_limit) -> Move:
        # TODO: erase the following line and implement this function.
        optional_moves_score = {}
        heuristic = HeuristicFunction()
        for move in Move:
            new_board, done, score = commands[move](board)
            if done:
                optional_moves_score[move] = heuristic.calc_heuristic(new_board)

        return max(optional_moves_score, key=optional_moves_score.get)


# TODO: add here helper functions in class, if needed


# part B
class MiniMaxMovePlayer(AbstractMovePlayer):
    """MiniMax Move Player,
    implement get_move function according to MiniMax algorithm
    (you can add helper functions as you want).
    """

    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # TODO: add here if needed

    def get_move(self, board, time_limit) -> Move:
        # TODO: erase the following line and implement this function.

        raise NotImplementedError

    # TODO: add here helper functions in class, if needed


class MiniMaxIndexPlayer(AbstractIndexPlayer):
    """MiniMax Index Player,
    this player is the opponent of the move player and need to return the indices on the board where to put 2.
    the goal of the player is to reduce move player score.
    implement get_indices function according to MiniMax algorithm, the value in minimax player value is only 2.
    (you can add helper functions as you want).
    """

    def __init__(self):
        AbstractIndexPlayer.__init__(self)
        # TODO: add here if needed

    def get_indices(self, board, value, time_limit) -> (int, int):
        # TODO: erase the following line and implement this function.
        raise NotImplementedError

    # TODO: add here helper functions in class, if needed


# part C
class ABMovePlayer(AbstractMovePlayer):
    """Alpha Beta Move Player,
    implement get_move function according to Alpha Beta MiniMax algorithm
    (you can add helper functions as you want)
    """

    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # TODO: add here if needed

    def get_move(self, board, time_limit) -> Move:
        # TODO: erase the following line and implement this function.
        raise NotImplementedError

    # TODO: add here helper functions in class, if needed


# part D
class ExpectimaxMovePlayer(AbstractMovePlayer):
    """Expectimax Move Player,
    implement get_move function according to Expectimax algorithm.
    (you can add helper functions as you want)
    """

    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # TODO: add here if needed

    def get_move(self, board, time_limit) -> Move:
        # TODO: erase the following line and implement this function.
        raise NotImplementedError

    # TODO: add here helper functions in class, if needed


class ExpectimaxIndexPlayer(AbstractIndexPlayer):
    """Expectimax Index Player
    implement get_indices function according to Expectimax algorithm, the value is number between {2,4}.
    (you can add helper functions as you want)
    """

    def __init__(self):
        AbstractIndexPlayer.__init__(self)
        # TODO: add here if needed

    def get_indices(self, board, value, time_limit) -> (int, int):
        # TODO: erase the following line and implement this function.
        raise NotImplementedError

    # TODO: add here helper functions in class, if needed


# Tournament
class ContestMovePlayer(AbstractMovePlayer):
    """Contest Move Player,
    implement get_move function as you want to compete in the Tournament
    (you can add helper functions as you want)
    """

    def __init__(self):
        AbstractMovePlayer.__init__(self)
        # TODO: add here if needed

    def get_move(self, board, time_limit) -> Move:
        # TODO: erase the following line and implement this function.
        raise NotImplementedError

    # TODO: add here helper functions in class, if needed