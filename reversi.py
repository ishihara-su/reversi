# reversi.py

from enum import Enum, auto
import math
import random
import sys


INVALID = -1
EMPTY = 0
BLACK = 1
WHITE = 2

WIN = 1
LOSE = -1
DRAW = 0


def error_exit(message):
    print(message, file=sys.stderr)
    sys.exit(1)


def another_color(color: int):
    if color == BLACK:
        return WHITE
    elif color == WHITE:
        return BLACK
    else:
        raise ValueError('Undefined player color')


class GameStatus(Enum):
    NORMAL = auto()
    PASSED = auto()
    END = auto()


class Board:
    def __init__(self, size=8):
        if size % 2 != 0 or size < 4:
            raise ValueError('Board size must be a even number larger than 2.')
        self.size = size
        self.states = [[0 for x in range(size)] for y in range(size)]
        self.gains = [[0 for x in range(size)] for y in range(size)]
        self.clear_all()

    def change_turn(self):
        self.turn = another_color(self.turn)
        self.clear_gains()
        self.eval_gain_all(self.turn)

    def clear_all(self):
        for y in range(self.size):
            for x in range(self.size):
                self.states[y][x] = 0
                self.gains[y][x] = 0
        self.game_status = GameStatus.NORMAL
        self.turn = BLACK
        self.scores = {BLACK: 2, WHITE: 2}
        self.initial_placement()
        self.eval_gain_all(self.turn)

    def clear_gains(self):
        for y in range(self.size):
            for x in range(self.size):
                self.gains[y][x] = 0

    def initial_placement(self):
        for x, y, color in [(self.size//2, self.size//2-1, BLACK),
                            (self.size//2-1, self.size//2, BLACK),
                            (self.size//2-1, self.size//2-1, WHITE),
                            (self.size//2, self.size//2, WHITE)]:
            self.states[y][x] = color

    def place(self, x: int, y: int, color: int) -> None:
        if color < 1 or 2 < color:
            raise ValueError('Undefined player number')
        if x < 0 or y < 0:
            if self.game_status == GameStatus.PASSED:
                self.game_status = GameStatus.END
                return
            elif self.game_status == GameStatus.NORMAL:
                self.game_status = GameStatus.PASSED
                return
            else:
                raise RuntimeError('Game is running after it ended.')
        gain = self.gains[y][x]
        if gain <= 0 and self.game_status == GameStatus.NORMAL:
            raise RuntimeError('Placed on an invalid square.')
        self.states[y][x] = color
        self.turn_over(x, y, color)
        self.add_score(gain, color)
        if sum(self.scores.values()) == self.size * self.size:
            self.game_status = GameStatus.END
        else:
            self.game_status = GameStatus.NORMAL

    def add_score(self, gain, color):
        self.scores[color] += gain + 1
        self.scores[another_color(color)] -= gain

    def eval_gain_all(self, color: int) -> bool:
        sum_gain = 0
        for y in range(self.size):
            for x in range(self.size):
                g = self.eval_gain(x, y, color)
                sum_gain += g
                self.gains[y][x] = g
        if sum_gain == 0:
            return False
        return True

    def state(self, x, y):
        if 0 <= x <= self.size - 1 and 0 <= y <= self.size - 1:
            return self.states[y][x]
        else:
            return INVALID

    def _turn_over_line(self, x: int, y: int, color: int,
                        xd: int, yd: int) -> None:
        if xd < -1 or xd > 1 or yd < -1 or 1 < yd:
            raise ValueError('xd, yd must be -1, 0, or 1.')
        c = 0
        another = another_color(color)
        while self.state(x + (c + 1) * xd, y + (c + 1) * yd) == another:
            c += 1
        if self.state(x + (c + 1) * xd, y + (c + 1) * yd) != color or c <= 0:
            return
        c = 0
        while self.state(x + (c + 1) * xd, y + (c + 1) * yd) == another:
            self.states[y + (c + 1) * yd][x + (c+1) * xd] = color
            c += 1

    def turn_over(self, x, y, color) -> int:
        for xd in [-1, 0, 1]:
            for yd in [-1, 0, 1]:
                if xd == 0 and yd == 0:
                    continue
                self._turn_over_line(x, y, color, xd, yd)

    def _eval_gain_line(self, x: int, y: int, color: int,
                        xd: int, yd: int) -> int:
        if xd < -1 or xd > 1 or yd < -1 or 1 < yd:
            raise ValueError('xd, yd must be -1, 0, or 1.')
        c = 0
        another = another_color(color)
        while self.state(x + (c + 1) * xd, y + (c + 1) * yd) == another:
            c += 1
        if self.state(x + (c + 1) * xd, y + (c + 1) * yd) != color or c <= 0:
            return 0
        return c

    def eval_gain(self, x, y, color) -> int:
        if self.states[y][x] != 0:
            return 0
        gain = 0
        for xd in [-1, 0, 1]:
            for yd in [-1, 0, 1]:
                if xd == 0 and yd == 0:
                    continue
                gain += self._eval_gain_line(x, y, color, xd, yd)
        return gain

    def show(self, show_guide: bool = False):
        print('+', end='')
        for x in range(self.size):
            print('----+', end='')
        print('')
        c_code = ord('a')
        for y in range(self.size):
            print('|', end='')
            for x in range(self.size):
                if self.states[y][x] == BLACK:
                    print(' ●  |', end='')
                elif self.states[y][x] == WHITE:
                    print(' ○  |', end='')
                elif show_guide and self.gains[y][x] > 0:
                    print(f'  {chr(c_code)} |', end='')
                    c_code += 1
                else:
                    print('    |', end='')
            print()
            print('+', end='')
            for x in range(self.size):
                print('----+', end='')
            print('')
        print(
            f"Score: Black: {self.scores[BLACK]}, White: {self.scores[WHITE]}")
        print(f"Turn: {'Black' if self.turn == BLACK else 'White'}")

    def get_code(self):
        code = (1 if self.turn == BLACK else 2, )
        for y in range(self.size):
            rowcode = 0
            for x in range(self.size):
                rowcode <<= 2
                rowcode += self.states[y][x]
            code = (*code, rowcode)
        return code


class Trainer():
    def __init__(self):
        pass

    def run(self):
        pass


class Agent:
    def __init__(self, color: int = BLACK) -> None:
        self.reset(color)
        self.n_games = 0
        self.n_wins = 0
        self.n_draw = 0

    def reset(self, color: int, learning: bool = True):
        self.color = color

    def place(board: Board) -> None:
        raise NotImplementedError('place() is not implemented.')

    def set_result(self, result: int) -> None:
        self.n_games += 1
        if result == WIN:
            self.n_wins += 1
        elif result == DRAW:
            self.n_draw += 1


class QLAgent(Agent):
    def __init__(self, color: int = BLACK,
                 learning_rate: float = 0.3,
                 discount_ratio: float = 0.7) -> None:
        super().__init__(color)
        self.learning_rate = learning_rate
        self.discount_ratio = discount_ratio
        self._qtable = dict()
        self.initial_q = 0.1

    def reset(self, color: int, learning: bool = True):
        super().reset(color)
        self.history = []
        self.learning = learning

    def place(self, board: Board) -> None:
        code = board.get_code()
        if code not in self._qtable:
            self.setup_qtable(board, code)
        if self.learning:
            x, y = self.select_softmax(board)
            self.history.append((board.get_code(), (x, y)))
        else:
            x, y = self.select_best(board)
        board.place(x, y, self.color)

    def setup_qtable(self, board: Board, code: tuple[int, ...]):
        self._qtable[code] = dict()
        for y in range(board.size):
            for x in range(board.size):
                if board.gains[y][x] > 0:
                    self._qtable[code][(x, y)] = self.initial_q

    def select_softmax(self, board: Board) -> tuple[int, int]:
        code = board.get_code()
        if len(self._qtable[code]) < 1:
            return (-1, -1)
        temperature = 0.5
        xmax = max([q/temperature for q in self._qtable[code].values()])
        weights = dict()
        for pos in self._qtable[code].keys():
            weights[pos] = math.exp(self._qtable[code][pos]/temperature - xmax)
        sum_weight = sum(weights.values())
        r = random.random() * sum_weight
        for pos in self._qtable[code].keys():
            if r < weights[pos]:
                return pos
            r -= weights[pos]
        assert False

    def select_best(self, board: Board) -> tuple[int, int]:
        candidates = []
        for y in range(board.size):
            for x in range(board.size):
                if board.gains[y][x] > 0:
                    candidates.append((x, y))
        maxq = -float('inf')
        best_pos = (-1, -1)
        code = board.get_code()
        for pos in candidates:
            try:
                q = self._qtable[code][pos]
            except KeyError:
                continue
            if q > maxq:
                maxq = q
                best_pos = pos
        if best_pos != (-1, -1):
            return best_pos
        return random.choice(candidates)

    def set_result(self, result: int) -> None:
        super().set_result(result)
        if not self.learning:
            return
        s = 0.0
        maxq = 0.0
        if result == WIN:
            s = 1.0
        elif result == DRAW:
            s = 0.0
        else:
            s = -1.0
        for (code, pos) in reversed(self.history):
            if pos == (-1, -1):
                continue
            oldq = self._qtable[code][pos]
            newq = oldq + self.learning_rate * \
                (s + self.discount_ratio * maxq - oldq)
            maxq = max(newq, maxq)
            self._qtable[code][pos] = newq


class RandomAgent(Agent):
    def place(self, board: Board) -> None:
        if self.color != board.turn:
            raise RuntimeError('Unmatched player color')
        candidates = []
        for y in range(board.size):
            for x in range(board.size):
                if board.gains[y][x] > 0:
                    candidates.append((x, y))
        if len(candidates) > 0:
            x, y = random.choice(candidates)
        else:
            x, y = -1, -1
        board.place(x, y, self.color)


class HumanAgent(Agent):
    def place(self, board: Board) -> None:
        if self.color != board.turn:
            raise RuntimeError('Unmatched player color')
        candidates = dict()
        c_code = ord('a')
        for y in range(board.size):
            for x in range(board.size):
                if board.gains[y][x] > 0:
                    candidates[chr(c_code)] = (x, y)
                    c_code += 1
        board.show(show_guide=True)
        while True:
            c = input('> ')
            if c == 'q':
                print('Bye')
                sys.exit(0)
            if c == 'p':
                if len(candidates) > 0:
                    print('You cannot pass now.')
                    continue
                x, y = -1, -1
                break
            try:
                x, y = candidates[c]
                break
            except KeyError:
                print('Undefined key.')
                continue
        board.place(x, y, self.color)


def run_game(agent_black: Agent, agent_white: Agent,
             board: Board, view: bool = False, learning: bool = True) -> None:
    agent_black.reset(BLACK, learning)
    agent_white.reset(WHITE, learning)
    agents = [agent_black, agent_white]
    count = 0
    board.clear_all()
    while True:
        if view:
            board.show()
        a = agents[count % 2]
        a.place(board)
        board.change_turn()
        count += 1
        if board.game_status == GameStatus.END:
            break
    result_str = ''
    if board.scores[BLACK] == board.scores[WHITE]:
        agent_black.set_result(DRAW)
        agent_white.set_result(DRAW)
        result_str = 'Draw.'
    elif board.scores[BLACK] > board.scores[WHITE]:
        agent_black.set_result(WIN)
        agent_white.set_result(LOSE)
        result_str = 'Black won.'
    else:
        agent_black.set_result(LOSE)
        agent_white.set_result(WIN)
        result_str = 'White won.'
    if view:
        board.show()
        print(result_str)


def human_human_game(size: int = 8):
    board = Board(size)
    run_game(HumanAgent(), HumanAgent(), board)


def human_random_game(size: int = 8):
    board = Board(size)
    run_game(HumanAgent(), RandomAgent(), board, True)


def train(a1: Agent, a2: Agent, board: Board,
          max_epoch: int = 10000, view: bool = False) -> None:
    for i in range(max_epoch):
        run_game(a1, a2, board, view, learning=True)
        run_game(a2, a1, board, view, learning=True)


def human_qa_game(size: int = 8, num_epoch: int = 10000):
    board = Board(size)
    q1 = QLAgent()
    q2 = QLAgent()
    train(q1, q2, board, num_epoch, view=True)
    ha = HumanAgent()
    run_game(ha, q1, board, view=True, learning=False)

if __name__ == '__main__':
    # human_human_game(6)
    # human_random_game(4)
    # run_game(RandomAgent(), RandomAgent(), Board(4), True)
    human_qa_game(4, 40)
