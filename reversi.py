# reversi.py

import sys

def error_exit(message):
    print(message, file=sys.stderr)
    sys.exit(1)

class Board:
    def __init__(self, size=8):
        if size % 2 != 0 or size < 4:
            raise ValueError('Board size must be a even number larger than 2.')
        self.size = size
        self.state = [[0] * size]*size
        self.initial_placement()

    def inital_placement(self):
        # TODO: コマを初期配置する
        # TODO: 引数にプレイヤーのインスタンスを入れる。
        black_cells = [(self.size//2, self.size//2), (self.size//2 + 1, self.size//2+1)]
        white_cells = [(self.size//2+1, self.size//2), (self.size//2, self.size//2+1)]
        pass

    def place(self, x, y, player):
        if player < 1 or 2 < player:
            raise ValueError('Undefined player number')
        if self.state[y][x] != 0:
            raise RuntimeError('Placement on an occupied cell.')
        self.state[y][x] = player
        self.update()

        # TODO: オセロのルールにしたがって、コマをひっくり返す。

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
                elif self.state[y][x] == '2':
                    print(f' ⚪ |', end='')
                else:
                    print(f'    |', end='')
            print('+', end='')
            for x in range(self.size):
                print('----+', end='')
            print('')
            fruits = ['apple', 'banana', 'cherry']
            for x in fruits:
              print(x)


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
