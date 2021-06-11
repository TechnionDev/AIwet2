import logic
import random
from AbstractPlayers import *
from math import log
import copy
import signal
import time

# commands to use for move players. dictionary : Move(enum) -> function(board),
# all the functions {up,down,left,right) receive board as parameter and return tuple of (new_board, done, score).
# new_board is according to the step taken, done is true if the step is legal, score is the sum of all numbers that
# combined in this step.
# (you can see GreedyMovePlayer implementation for example)
from constants import *

inf = 1 << 32
PROBABILITY_2 = 0.9
PROBABILITY_4 = 0.1

commands = {Move.UP: logic.up, Move.DOWN: logic.down,
            Move.LEFT: logic.left, Move.RIGHT: logic.right}


def signal_handler(signum, frame):
    raise Exception("Timeout")


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


def calc_heuristic(board):
    empty = 3 * _empty_cells(board)
    max_val = 3 * _max_value(board)
    mono_val = 1 * _monotonic_board_2(board)
    same_tiles_val = 2 * _same_tiles(board)
    return empty + mono_val + max_val + same_tiles_val


def _empty_cells(board) -> float:
    empty_cells = 0
    for i in range(GRID_LEN):
        for j in range(GRID_LEN):
            if board[i][j] == 0:
                empty_cells += 1
    return empty_cells


def _max_value(board) -> float:
    max_val = 0
    for i in range(GRID_LEN):
        for j in range(GRID_LEN):
            if board[i][j] > max_val:
                max_val = board[i][j]
    return log(max_val)


def _monotonic_board_2(board):
    left_to_right = 0
    right_to_left = 0
    up_to_down = 0
    down_to_up = 0
    for x in range(GRID_LEN):
        for y in range(GRID_LEN - 1):
            # horizontal
            val_curr = board[x][y]
            val_next = board[x][y + 1]
            if val_curr != 0 or val_next != 0:
                if val_curr < val_next:
                    left_to_right += val_next - val_curr
                if val_curr > val_next:
                    right_to_left += val_curr - val_next
            # vertical
            val_curr = board[y][x]
            val_next = board[y + 1][x]
            if val_curr != 0 or val_next != 0:
                if val_curr < val_next:
                    up_to_down += val_next - val_curr
                if val_curr > val_next:
                    down_to_up += val_curr - val_next

    res = min(left_to_right, right_to_left) + min(up_to_down, down_to_up)
    if res == 0:
        return 0
    return -log(res)


def _same_tiles(board):
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
    return counter


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

    def get_move(self, board, time_limit) -> Move:
        optional_moves_score = {}
        for move in Move:
            new_board, done, score = commands[move](board)
            if done:
                optional_moves_score[move] = calc_heuristic(new_board)

        return max(optional_moves_score, key=optional_moves_score.get)


# part B
class MiniMaxMovePlayer(AbstractMovePlayer):
    """MiniMax Move Player,
    implement get_move function according to MiniMax algorithm
    (you can add helper functions as you want).
    """

    def __init__(self):
        AbstractMovePlayer.__init__(self)

    def get_move(self, board, time_limit) -> Move:
        move = Move.LEFT
        iter = 0
        signal.signal(signal.SIGALRM, signal_handler)
        signal.setitimer(signal.ITIMER_REAL, 0.8 * time_limit)
        try:
            while True:
                move = MiniMaxMovePlayer.min_max_move(board, iter)[0]
                iter += 1
        except Exception as msg:
            pass
        return move

    @staticmethod
    def min_max_move(board, iteration) -> (Move, float):
        optional_moves_score = {}
        if iteration == 0:
            for move in Move:
                new_board, done, score = commands[move](board)
                if done:
                    optional_moves_score[move] = calc_heuristic(new_board)
            if not optional_moves_score:
                return Move.LEFT, 0  # need to put something in the second value
            else:
                res_move = max(optional_moves_score, key=optional_moves_score.get)
                return res_move, optional_moves_score[res_move]
        else:
            for move in Move:
                new_board, done, score = commands[move](board)
                if done:
                    optional_moves_score[move] = MiniMaxIndexPlayer.min_max_index(new_board, iteration - 1)[1]
            if not optional_moves_score:
                return Move.LEFT, 0
            else:
                res_move = max(optional_moves_score, key=optional_moves_score.get)
                return res_move, optional_moves_score[res_move]


class MiniMaxIndexPlayer(AbstractIndexPlayer):
    """MiniMax Index Player,
    this player is the opponent of the move player and need to return the indices on the board where to put 2.
    the goal of the player is to reduce move player score.
    implement get_indices function according to MiniMax algorithm, the value in minimax player value is only 2.
    (you can add helper functions as you want).
    """

    def __init__(self):
        AbstractIndexPlayer.__init__(self)

    def get_indices(self, board, value, time_limit) -> (int, int):
        row = 0
        col = 0
        iteration = 0

        signal.signal(signal.SIGALRM, signal_handler)
        signal.setitimer(signal.ITIMER_REAL, 0.80 * time_limit)
        try:
            while True:
                row, col = MiniMaxIndexPlayer.min_max_index(board, iteration)[0]
                iteration += 1
        except Exception as msg:
            pass
        return row, col

    @staticmethod
    def min_max_index(board, iteration) -> ((int, int), float):
        optional_index_score = {}
        if iteration == 0:
            for row in range(GRID_LEN):
                for col in range(GRID_LEN):
                    if board[row][col] == 0:
                        new_board = copy.deepcopy(board)
                        new_board[row][col] = 2
                        optional_index_score[(row, col)] = calc_heuristic(new_board)
            res_index = min(optional_index_score, key=optional_index_score.get)
            return res_index, optional_index_score[res_index]
        else:
            for row in range(GRID_LEN):
                for col in range(GRID_LEN):
                    if board[row][col] == 0:
                        new_board = copy.deepcopy(board)
                        new_board[row][col] = 2
                        optional_index_score[(row, col)] = (MiniMaxMovePlayer.min_max_move(new_board, iteration - 1))[1]
            return min(optional_index_score, key=optional_index_score.get), min(optional_index_score.values())


# part C
class ABMovePlayer(AbstractMovePlayer):
    """Alpha Beta Move Player,
    implement get_move function according to Alpha Beta MiniMax algorithm
    (you can add helper functions as you want)
    """

    def __init__(self):
        AbstractMovePlayer.__init__(self)

    def get_move(self, board, time_limit) -> Move:
        move = Move.LEFT
        iter = 0
        start = time.time()
        signal.signal(signal.SIGALRM, signal_handler)
        signal.setitimer(signal.ITIMER_REAL, 0.80 * time_limit)
        try:
            while True:
                move = ABMovePlayer.min_max_move(board, iter)[0]
                iter += 1
        except Exception as msg:
            pass
        return move

    @staticmethod
    def min_max_move(board, iteration, alpha=-inf, beta=inf) -> (Move, float):
        optional_moves_score = {}
        if iteration == 0:
            for move in Move:
                new_board, done, score = commands[move](board)
                if done:
                    optional_moves_score[move] = calc_heuristic(new_board)
            if not optional_moves_score:
                return Move.LEFT, 0  # need to put something in the second value
            else:
                res_move = max(optional_moves_score, key=optional_moves_score.get)
                return res_move, optional_moves_score[res_move]
        else:
            for move in Move:
                new_board, done, score = commands[move](board)
                if done:
                    optional_moves_score[move] = ABMovePlayer.min_max_index(new_board, iteration - 1, alpha, beta)[1]
                    alpha = max(alpha, optional_moves_score[move])
                    if alpha >= beta:
                        break
            if not optional_moves_score:
                return Move.LEFT, 0
            else:
                res_move = max(optional_moves_score, key=optional_moves_score.get)
                return res_move, optional_moves_score[res_move]

    @staticmethod
    def min_max_index(board, iteration, alpha=-inf, beta=inf) -> ((int, int), float):
        optional_index_score = {}
        if iteration == 0:
            for row in range(GRID_LEN):
                for col in range(GRID_LEN):
                    if board[row][col] == 0:
                        new_board = copy.deepcopy(board)
                        new_board[row][col] = 2
                        optional_index_score[(row, col)] = calc_heuristic(new_board)
            res_index = min(optional_index_score, key=optional_index_score.get)
            return res_index, optional_index_score[res_index]
        else:
            for row in range(GRID_LEN):
                for col in range(GRID_LEN):
                    if board[row][col] == 0:
                        new_board = copy.deepcopy(board)
                        new_board[row][col] = 2
                        optional_index_score[(row, col)] = \
                            (ABMovePlayer.min_max_move(new_board, iteration - 1, alpha, beta))[1]
                        beta = min(beta, optional_index_score[(row, col)])
                        if beta <= alpha:
                            break
            return min(optional_index_score, key=optional_index_score.get), min(optional_index_score.values())


# part D
class ExpectimaxMovePlayer(AbstractMovePlayer):
    """Expectimax Move Player,
    implement get_move function according to Expectimax algorithm.
    (you can add helper functions as you want)
    """

    def __init__(self):
        AbstractMovePlayer.__init__(self)

    def get_move(self, board, time_limit) -> Move:
        move = Move.LEFT
        iter = 0
        signal.signal(signal.SIGALRM, signal_handler)
        signal.setitimer(signal.ITIMER_REAL, 0.80 * time_limit)
        try:
            while True:
                move = ExpectimaxMovePlayer.min_max_move(board, iter)[0]
                iter += 1
        except Exception as msg:
            pass

        return move

    @staticmethod
    def min_max_move(board, iteration) -> (Move, float):
        optional_moves_score = {}
        if iteration == 0:
            for move in Move:
                new_board, done, score = commands[move](board)
                if done:
                    optional_moves_score[move] = calc_heuristic(new_board)
            if not optional_moves_score:
                return Move.LEFT, 0  # need to put something in the second value
            else:
                res_move = max(optional_moves_score, key=optional_moves_score.get)
                return res_move, optional_moves_score[res_move]
        else:
            for move in Move:
                new_board, done, score = commands[move](board)
                if done:
                    optional_moves_score[move] = ExpectimaxIndexPlayer.min_max_index(new_board, iteration - 1, 2)[
                                                     1] * PROBABILITY_2
                    optional_moves_score[move] += ExpectimaxIndexPlayer.min_max_index(new_board, iteration - 1, 4)[
                                                      1] * PROBABILITY_4

            if not optional_moves_score:
                return Move.LEFT, 0
            else:
                res_move = max(optional_moves_score, key=optional_moves_score.get)
                return res_move, optional_moves_score[res_move]


class ExpectimaxIndexPlayer(AbstractIndexPlayer):
    """Expectimax Index Player
    implement get_indices function according to Expectimax algorithm, the value is number between {2,4}.
    (you can add helper functions as you want)
    """

    def __init__(self):
        AbstractIndexPlayer.__init__(self)

    def get_indices(self, board, value, time_limit) -> (int, int):
        row = 0
        col = 0
        iteration = 0

        signal.signal(signal.SIGALRM, signal_handler)
        signal.setitimer(signal.ITIMER_REAL, 0.80 * time_limit)
        try:
            while True:
                row, col = self.min_max_index(board, iteration, value)[0]
                iteration += 1
        except Exception as msg:
            pass
        return row, col

    @staticmethod
    def min_max_index(board, iteration, value) -> ((int, int), float):
        optional_index_score = {}
        if iteration == 0:
            for row in range(GRID_LEN):
                for col in range(GRID_LEN):
                    if board[row][col] == 0:
                        new_board = copy.deepcopy(board)
                        new_board[row][col] = value
                        optional_index_score[(row, col)] = calc_heuristic(new_board)
            res_index = min(optional_index_score, key=optional_index_score.get)
            return res_index, optional_index_score[res_index]
        else:
            for row in range(GRID_LEN):
                for col in range(GRID_LEN):
                    if board[row][col] == 0:
                        new_board = copy.deepcopy(board)
                        new_board[row][col] = value
                        optional_index_score[(row, col)] = \
                            (ExpectimaxMovePlayer.min_max_move(new_board, iteration - 1))[1]
            return min(optional_index_score, key=optional_index_score.get), min(optional_index_score.values())


# Tournament
class ContestMovePlayer(AbstractMovePlayer):
    """Contest Move Player,
    implement get_move function as you want to compete in the Tournament
    (you can add helper functions as you want)
    """

    def __init__(self):
        AbstractMovePlayer.__init__(self)

    def get_move(self, board, time_limit) -> Move:
        move = Move.LEFT
        iter = 0
        signal.signal(signal.SIGALRM, signal_handler)
        signal.setitimer(signal.ITIMER_REAL, 0.80 * time_limit)
        try:
            while True:
                move = ABMovePlayer.min_max_move(board, iter)[0]
                iter += 1
        except Exception:
            pass
        return move