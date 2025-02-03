import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Any, List, Optional
from copy import deepcopy
from core.enums import Command, Option, OptionType, Strategy, GameType, PlayerColor, GameState, BugType, Direction
from core.board import Board
from core.game import Move, Bug
from core.ai import Brain, Random, AlphaBetaPruner
from core.engine import Engine
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

def play_match(engine: Engine, arguments: List[str]) -> Optional[List[List[Any]]]:
    
    engine.newgame(arguments)
    match_id = int(time())
    sample_board = Board(" ".join(arguments)).get_stats()
    headers = generate_headers(engine, sample_board)
    rows = [headers]
    turn = 1

    while not engine.board.gameover:
        if turn > 100:
            print(f"Match discarded due to excessive turns: {turn}")
            return None
        
        # # Select the move
        # move = engine.brains[engine.board.current_player_color].calculate_best_move(engine.board, time_limit=5)

        # Random move
        # Generating the list of all valid moves
        valid_moves = engine.board.valid_moves.split(";")
        last_move_played_by = engine.board.current_player_color

        # Loop to avoid playing a move that ends the game againts the player that played it
        while True:
            # Selecting a random move
            index = randint(0, len(valid_moves) - 1)
            move = valid_moves[index]

            # Play the move
            engine.play_match_generator(move)

            # Check if the game is over and the player who lost is the same that played the move
            if engine.board.state == GameState.WHITE_WINS and last_move_played_by == PlayerColor.BLACK:
                engine.undo(['1'])
            elif engine.board.state == GameState.BLACK_WINS and last_move_played_by == PlayerColor.WHITE:
                engine.undo(['1'])
            else:
                break
            

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
    for i, row in enumerate(rows):
        if i != 0:
            row.append(result)
    with open(file_path, mode='w', newline='') as write_file:
        writer = csv.writer(write_file)
        writer.writerows(rows)

def match_generator(engine: Engine, arguments: List[str], num_matches: int) -> None:
    white_wins = 0
    black_wins = 0

    # Create a folder to store the results (this folder is in the 'data' folder)
    folder = 'newtest'
    folder_path = os.path.join(os.path.dirname(__file__), '../..', f'data/{folder}')
    os.makedirs(folder_path, exist_ok=True)

    k = 0
    while k < num_matches:
        result_data = play_match(engine, arguments)
        if result_data is None:
            continue
        rows, match_id = result_data
        result = determine_result(engine)
        if result:
            file_path = os.path.join(folder_path, f'match_{match_id}_results.csv')
            save_match_results(rows, match_id, result, file_path)
            print(f"Game over, match duration: {round(time() - match_id, 3)} seconds")
            if result == 'White':
                white_wins += 1
            elif result == 'Black':
                black_wins += 1
        k += 1

    print(f"White wins: {white_wins}")
    print(f"Black wins: {black_wins}")

if __name__ == "__main__":
    match_generator(Engine(), ["Base"], 10)
