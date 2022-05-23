# reversi.py

import sys

def error_exit(message):
    print(message, file=sys.stderr)
    sys.exit(1)

def another_player(player):
    if player == 1:
        return 0
    elif player == 2:
        return 1
    else:
        raise ValueError('Undefined player id')

class Board:
    def __init__(self, size=8):
        if size % 2 != 0 or size < 4:
            raise ValueError('Board size must be a even number larger than 2.')
        self.size = size
        self.state = [[0] * size]*size
        self.initial_placement()

    # NOTE そもそもこのクラスに設けるべきメソッドではないのではないか
    def inital_placement(self):
        # TODO: コマを初期配置する
        # TODO: 引数にプレイヤーのインスタンスを入れる。
        black_cells = [(self.size//2, self.size//2), (self.size//2 + 1, self.size//2+1)]
        white_cells = [(self.size//2+1, self.size//2), (self.size//2, self.size//2+1)]
        pass

    def place(self, x, y, player):
        if player < 1 or 2 < player:
            raise ValueError('Undefined player number')
       # 置けるかどうか確認
        self.state[y][x] = player
        self.update()

    # TODO: オセロのルールにしたがって、コマをひっくり返す。
    def turnover(self, x, y, player):
        pass


    def check_gain(self, x, y, player):
        if x < 0 or self.size <= x or y < 0 or self.size <= y:
            raise ValueError('Invalid cell position')
        if self.state[y][x] != 0:
            # ここの部分はエラーにしないで、単に0を返すだけでも良いかも知れない
            raise RuntimeError('Placement on an occupied cell.')
        another = another_player(player)
        gain = 0
        if x > 1:
            c = 0
            while x - 1 - c >= 0 and self.state[y][x - 1 - c] == another:
                c += 1
            if self.state[y][x - 1 - c] == player:
                gain += c
        if x < self.size - 1:
            c = 0
            while x + 1 + c <= self.size - 1 and self.state[y][x + 1 + c] == another:
                c += 1
            if self.state[y][x + 1 + c] == player:
                gain += c
        if y > 1:
            c = 0
            while y - 1 - c >= 0 and self.state[y - 1 - c][x] == another:
                c += 1
            if self.state[y - 1 - c][x] == player:
                gain += c
        if y < self.size - 1:
            c = 0
            while y + 1 + c <= self.size - 1 and self.state[y + 1 + c][x] == another:
                c += 1
            if self.state[y + 1 + c][x] == player:
                gain += c
        # ななめ左下に進む
        if x < self.size - 1 and y > 1:
            c = 0
            while x - 1 - c >= 0 and y + 1 + c <= self.size - 1 and self.state[y+1+c][x-1-c] == another:
                c += 1
            if self.state[y+1+c][x-1-c] == player:
                gain += c
        # TODO ななめ右上
        # TODO ななめ左上
        # TODO ななめ右下
        return gain

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
