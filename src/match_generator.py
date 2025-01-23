import os
from typing import TypeGuard, Final, Optional, Any
from copy import deepcopy
from enums import Command, Option, OptionType, Strategy, GameType, PlayerColor, GameState, BugType, Direction
from board import Board
from game import Move, Bug
from ai import Brain, Random, AlphaBetaPruner
from engine import Engine
import re
import csv
from random import randint, choice
from time import time

def match_generator(engine, arguments: list[str], num_matches: int) -> None:
    """
    Engine loop to play automatically against itself for a specified number of matches.
    Generates for each match a file csv containing all the information about it.
    """
    for _ in range(num_matches):
        engine.newgame(arguments)
        # Starting time of the game
        time_start = time()
        match_id = int(time_start)

        file_path = os.path.join(os.path.dirname(__file__), '..', 'data', f'match_{match_id}_results.csv')
        sample_board = Board(" ".join(arguments)).get_stats()
        headers = ['ID_match', 'number_of_turn', 'last_move_played_by', 'current_player_turn'] + [f'{key}_moves' for key in list(sample_board.keys())] + ['result']
        
        # Add headers for neighbors information
        for color in PlayerColor:
            for bug_type in BugType:
                for direction in Direction:
                    headers.append(f'{color.name}_{bug_type.name}_{direction.name}_neighbor')
        
        rows = [headers]
        turn = 1

        while not engine.board.gameover:
            # Generating the list of all valid moves
            valid_moves = engine.board.valid_moves.split(";")
            # Selecting a random move
            index = randint(0, len(valid_moves) - 1)
            move = valid_moves[index]
            last_move_played_by = engine.board.current_player_color
            # Play the move
            engine.play(move)
            # Save the current board state after the move
            board_stats = deepcopy(engine.board.get_stats())

            # Get neighbors information
            neighbor_stats = deepcopy(engine.board.get_neighbor_stats())

            row = [match_id, turn, last_move_played_by, engine.board.current_player_color] + list(board_stats.values())
            
            # Add neighbors information to the row
            for bug in engine.board.get_board_bugs():
                    for direction in Direction:
                        row.append(neighbor_stats[bug][str(direction)])

            rows.append(row)
            turn += 1

        # Ending time of the game
        time_end = time()
        print("Game over, match duration:", round(time_end - time_start, 3), "seconds")

        # Determine the result of the match
        if engine.board.state == GameState.WHITE_WINS:
            result = 'White'
        elif engine.board.state == GameState.BLACK_WINS:
            result = 'Black'
        elif engine.board.state == GameState.DRAW:
            result = 'Draw'
        else:
            continue  # Skip if the game state is invalid

        # Update the result for all rows of this match
        for row in rows:
            if row[0] == match_id:
                row.append(result)

        with open(file_path, mode='w', newline='') as write_file:
            writer = csv.writer(write_file)
            writer.writerows(rows)

if __name__ == "__main__":

  # Generates for a specified number of matches a file csv containing all the information about the match
  match_generator(Engine(), ["Base"], 2)