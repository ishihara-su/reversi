# reversi.py

import sys

from pip import main

def error_exit(message):
    print(message, file=sys.stderr)
    sys.exit(1)

class Board:
    def __init__(self, size=8):
        self.size = size
        self.state = [[0] * size]*size
        self.initial_placement()

    def inital_placement(self):
        # TODO: コマを初期配置する
        pass

    def place(self, x, y, player):
        if player < 1 or 2 < player:
            raise ValueError('Undefined player number')
        if self.state[y][x] != 0:
            raise RuntimeError('Placement on an occupied cell.')
        self.state[y][x] = player
        self.update()

    def update(self):
        # オセロのルールにしたがって、コマをひっくり返す。
        pass

    def show(self):
        print('+', end='')
        for x in range(self.size):
            print('----+', end='')
        print('')
        for y in range(self.size):
            print('|', end='')
            for x in range(self.size):
                if self.state[y][x] == '1':
                    print(f' ⚫ |', end='')
                elif self.state[y][x] = '2':
                    print(f' ⚪ |', end='')
                else:
                    print(f'    |', end='')
            print('+', end='')
            for x in range(self.size):
                print('----+', end='')
            print('')

class Trainer():
    def __init__(self):
        pass

    def run(self):
        pass

def main():
    t = Trainer()
    t.run()

if __name__ == '__main__':
    main()
