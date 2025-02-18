# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from board import Board


from typing import Optional, Iterator
from random import choice
from time import sleep

from abc import ABC, abstractmethod

class Brain(ABC):
  """
  Base abstract class for AI agents.
  """

  def __init__(self) -> None:
    self._cache: Optional[str] = None

  @abstractmethod
  def calculate_best_move(self, board: Board, max_depth: int = 0, time_limit: int = 0) -> str:
    """
    Calculates the best move for the given board state, following the agent's policy.

    :param board: Current playing board.
    :type board: Board
    :param max_depth: Maximum lookahead depth, defaults to 0.
    :type max_depth: int, optional
    :param time_limit: Maximum time (in seconds) to calculate the best move, defaults to 0.
    :type time_limit: int, optional
    :return: Stringified best move.
    :rtype: str
    """
    pass

  def empty_cache(self) -> None:
    """
    Empties the current cache for the best move.  
    To be called OUTSIDE this class when needed.
    """
    self._cache = None

class Random(Brain):
  """
  Random acting AI agent.
  """

  def calculate_best_move(self, board: Board, max_depth: int = 0, time_limit: int = 0) -> str:
    if not self._cache:
      self._cache = choice(board.valid_moves.split(";"))
    sleep(0.5)
   
    return self._cache



