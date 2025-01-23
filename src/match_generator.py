import os
from typing import Any, List, Optional
from copy import deepcopy
from enums import Command, Option, OptionType, Strategy, GameType, PlayerColor, GameState, BugType, Direction
from board import Board
from game import Move, Bug
from ai import Brain, Random, AlphaBetaPruner
from engine import Engine
import csv
from random import randint
from time import time

def generate_headers(engine: Engine, sample_board: dict) -> List[str]:
    headers = ['number_of_turn', 'last_move_played_by', 'current_player_turn'] + [f'{key}_moves' for key in list(sample_board.keys())] 
    for bug in engine.board.get_board_bugs():
        for direction in Direction:
            headers.append(f'{bug}_{direction.name}_neighbor')
    headers.append('result')
    return headers

def play_match(engine: Engine, arguments: List[str]) -> List[List[Any]]:
    engine.newgame(arguments)
    match_id = int(time())
    sample_board = Board(" ".join(arguments)).get_stats()
    headers = generate_headers(engine, sample_board)
    rows = [headers]
    turn = 1

    while not engine.board.gameover:
        valid_moves = engine.board.valid_moves.split(";")
        move = valid_moves[randint(0, len(valid_moves) - 1)]
        last_move_played_by = engine.board.current_player_color
        engine.play(move)
        board_stats = deepcopy(engine.board.get_stats())
        neighbor_stats = deepcopy(engine.board.get_neighbor_stats())

        row = [turn, last_move_played_by, engine.board.current_player_color] + list(board_stats.values())
        for bug in engine.board.get_board_bugs():
            for direction in Direction:
                row.append(neighbor_stats[bug][str(direction)])
        rows.append(row)
        turn += 1

    return rows, match_id

def determine_result(engine: Engine) -> Optional[str]:
    if engine.board.state == GameState.WHITE_WINS:
        return 'White'
    elif engine.board.state == GameState.BLACK_WINS:
        return 'Black'
    elif engine.board.state == GameState.DRAW:
        return 'Draw'
    return None

def save_match_results(rows: List[List[Any]], match_id: int, result: str, file_path: str) -> None:
    for row in rows:
        if row[0] == match_id:
            row.append(result)
    with open(file_path, mode='w', newline='') as write_file:
        writer = csv.writer(write_file)
        writer.writerows(rows)

def match_generator(engine: Engine, arguments: List[str], num_matches: int) -> None:
    for _ in range(num_matches):
        rows, match_id = play_match(engine, arguments)
        result = determine_result(engine)
        if result:
            file_path = os.path.join(os.path.dirname(__file__), '..', 'data', f'match_{match_id}_results.csv')
            save_match_results(rows, match_id, result, file_path)
            print(f"Game over, match duration: {round(time() - match_id, 3)} seconds")

if __name__ == "__main__":
    match_generator(Engine(), ["Base"], 2)
