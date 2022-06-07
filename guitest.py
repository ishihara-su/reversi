import PySimpleGUI as sg

"""
    Test application to show and control the game board.
"""

BOARD_SIZE = 8

layout = [
    [sg.Text('Reversi')],
    [sg.Graph((800, 800), (0, 450), (450, 0), key='-BOARD-',
              change_submits=True, drag_submits=False)],
    [sg.Button('Next'), sg.Button('Exit')]
]

window = sg.Window('Reversi', layout, finalize=True)
g = window['-BOARD-']

for row in range(BOARD_SIZE):
    for col in range(BOARD_SIZE):
