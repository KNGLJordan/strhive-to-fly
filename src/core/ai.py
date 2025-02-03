from typing import Optional, Set, Iterator
from random import choice
from time import sleep, time
from copy import deepcopy
from abc import ABC, abstractmethod
from board import Board
from enums import Direction
import pickle
import pandas as pd
from ml.ml_utils import df_preprocessing
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import numpy as np
    

type ABNode = tuple[Board, float, float, float, bool, Iterator[tuple[Board, str]], float, str]
"""
Alpha-beta pruning node.
"""

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

class AlphaBetaPruner(Brain):
  """
  AI agent following an alpha-beta pruning policy.
  """

  def calculate_best_move(self, board: Board, max_depth: int = 0, time_limit: int = 0) -> str:
    if not self._cache:
      self._cache = self.abpruning(board, max_depth, time_limit)
    return self._cache

  def abpruning(self, root: Board, max_depth: int = 0, time_limit: int = 0) -> str:
    """
    Minmax with alpha-beta pruning to search for the best move, given the depth and time constraints.  
    This is an iterative version rather than recursive.

    :param root: Starting node.
    :type root: Board
    :param max_depth: Maximum allowed depth to search at, defaults to 0.
    :type max_depth: int, optional
    :param time_limit: Maximum amount of seconds allowed to spend, defaults to 0.
    :type time_limit: int, optional
    :return: Best move.
    :rtype: str
    """
    start_time = time()

    # Stack to simulate recursion
    stack: list[ABNode] = []
    # Each stack frame contains: (node, depth, alpha, beta, maximizing, child_iter, best_value, best_move)
    stack.append((root, max_depth or float('inf'), float('-inf'), float('inf'), True, iter(self.gen_children(root)), float('-inf'), root.valid_moves.split(";")[0]))

    result: str = root.valid_moves.split(";")[0]

    while stack:
      if time_limit and time() - start_time > time_limit:
        break

      node, depth, alpha, beta, maximizing, child_iter, best_value, best_move = stack.pop()

      if depth == 0 or node.gameover:
        # Evaluate leaf or terminal node
        evaluation = self.evaluate(node)
        if stack:
          # Pass the evaluation result back up the stack
          stack[-1] = self._update_parent(stack[-1], evaluation, best_move)
        else:
          # If this is the root node, store the result
          result = best_move
        continue

      try:
        # Process the next child
        child, move = next(child_iter)
        # Push this node back onto the stack to continue processing later
        stack.append((node, depth, alpha, beta, maximizing, child_iter, best_value, best_move))
        # Push the child onto the stack for processing
        stack.append((child, depth - 1, alpha, beta, not maximizing, iter(self.gen_children(child)), float('-inf') if not maximizing else float('inf'), move))
      except StopIteration:
        # All children processed; finalize the value for this node
        if stack:
          # Pass the result back up the stack
          stack[-1] = self._update_parent(stack[-1], best_value, best_move)
        else:
          # If this is the root node, store the result
          result = best_move

    return result

  def _update_parent(self, parent: ABNode, child_value: float, child_move: str) -> ABNode:
    """
    Updates the given parent with the information from the child.

    :param parent: Parent node to update.
    :type parent: ABNode
    :param child_value: Child node value.
    :type child_value: float
    :param child_move: Move that led to the child node.
    :type child_move: str
    :return: Updated parent node.
    :rtype: ABNode
    """
    # Update the stack frame based on the returned child value
    node, depth, alpha, beta, maximizing, child_iter, best_value, best_move = parent

    if maximizing:
      if child_value > best_value:
        best_value = child_value
        best_move = child_move
      alpha = max(alpha, best_value)
    else:
      best_value = min(best_value, child_value)
      if child_value < best_value:
        best_value = child_value
        best_move = child_move
      beta = min(beta, best_value)

    if beta <= alpha:
      # Clear the iterator to skip remaining children (prune)
      child_iter: Iterator[tuple[Board, str]] = iter([])

    return node, depth, alpha, beta, maximizing, child_iter, best_value, best_move

  def evaluate(self, node: Board) -> float:
    """
    Evaluates the given node.  
    Currently, it's a very naive implementation that weights the winning state (how many pieces surround the enemy queen minus how many pieces surround yours) and the mobility state (amount of your available moves minus the enemy's).

    :param node: Playing board.
    :type node: Board
    :return: Node value.
    :rtype: float
    """
    minimizing_color = node.current_player_color
    maximizing_color = minimizing_color.opposite
    return (node.count_queen_neighbors(minimizing_color) - node.count_queen_neighbors(maximizing_color)) * 20 + (len(node.calculate_valid_moves_for_player(maximizing_color, True)) - len(node.calculate_valid_moves_for_player(minimizing_color))) // 2

  def gen_children(self, parent: Board) -> Set[tuple[Board, str]]:
    """
    Generates valid children from the given parent.

    :param parent: Parent node.
    :type parent: Board
    :return: Set of children.
    :rtype: Set[tuple[Board, str]]
    """
    return {(deepcopy(parent).play(move), move) for move in parent.valid_moves.split(";") if not parent.gameover}

class AlphaBetaPrunerMLClassifier(Brain):
  """
  AI agent following an alpha-beta pruning policy.
  """
  def __init__(self) -> None:
    super().__init__()
    self.model = pickle.load(open('model/model0.pkl', 'rb'))

  def calculate_best_move(self, board: Board, max_depth: int = 0, time_limit: int = 0) -> str:
    if not self._cache:
      self._cache = self.abpruning(board, max_depth, time_limit)
    return self._cache

  def abpruning(self, root: Board, max_depth: int = 0, time_limit: int = 0) -> str:
    """
    Minmax with alpha-beta pruning to search for the best move, given the depth and time constraints.  
    This is an iterative version rather than recursive.

    :param root: Starting node.
    :type root: Board
    :param max_depth: Maximum allowed depth to search at, defaults to 0.
    :type max_depth: int, optional
    :param time_limit: Maximum amount of seconds allowed to spend, defaults to 0.
    :type time_limit: int, optional
    :return: Best move.
    :rtype: str
    """
    start_time = time()

    # Stack to simulate recursion
    stack: list[ABNode] = []
    # Each stack frame contains: (node, depth, alpha, beta, maximizing, child_iter, best_value, best_move)
    stack.append((root, max_depth or float('inf'), float('-inf'), float('inf'), True, iter(self.gen_children(root)), float('-inf'), root.valid_moves.split(";")[0]))

    result: str = root.valid_moves.split(";")[0]

    while stack:
      if time_limit and time() - start_time > time_limit:
        break

      node, depth, alpha, beta, maximizing, child_iter, best_value, best_move = stack.pop()

      if depth == 0 or node.gameover:
        # Evaluate leaf or terminal node
        evaluation = self.evaluate(node)
        if stack:
          # Pass the evaluation result back up the stack
          stack[-1] = self._update_parent(stack[-1], evaluation, best_move)
        else:
          # If this is the root node, store the result
          result = best_move
        continue

      try:
        # Process the next child
        child, move = next(child_iter)
        # Push this node back onto the stack to continue processing later
        stack.append((node, depth, alpha, beta, maximizing, child_iter, best_value, best_move))
        # Push the child onto the stack for processing
        stack.append((child, depth - 1, alpha, beta, not maximizing, iter(self.gen_children(child)), float('-inf') if not maximizing else float('inf'), move))
      except StopIteration:
        # All children processed; finalize the value for this node
        if stack:
          # Pass the result back up the stack
          stack[-1] = self._update_parent(stack[-1], best_value, best_move)
        else:
          # If this is the root node, store the result
          result = best_move

    return result

  def _update_parent(self, parent: ABNode, child_value: float, child_move: str) -> ABNode:
    """
    Updates the given parent with the information from the child.

    :param parent: Parent node to update.
    :type parent: ABNode
    :param child_value: Child node value.
    :type child_value: float
    :param child_move: Move that led to the child node.
    :type child_move: str
    :return: Updated parent node.
    :rtype: ABNode
    """
    # Update the stack frame based on the returned child value
    node, depth, alpha, beta, maximizing, child_iter, best_value, best_move = parent

    if maximizing:
      if child_value > best_value:
        best_value = child_value
        best_move = child_move
      alpha = max(alpha, best_value)
    else:
      best_value = min(best_value, child_value)
      if child_value < best_value:
        best_value = child_value
        best_move = child_move
      beta = min(beta, best_value)

    if beta <= alpha:
      # Clear the iterator to skip remaining children (prune)
      child_iter: Iterator[tuple[Board, str]] = iter([])

    return node, depth, alpha, beta, maximizing, child_iter, best_value, best_move

  def evaluate(self, node: Board) -> float:
    """
    Evaluates the given node.  
    It uses a pre-trained ML model to predict the best move.

    :param node: Playing board.
    :type node: Board
    :return: Node value.
    :rtype: float
    """
    

    # Structure of the data
    # 'number_of_turn','last_move_played_by','current_player_turn',
    # features relative to possible moves for every bug
    # features relative to neighbors for every bug

    # Extracting the features
    number_of_turn = 0.5 #standard assignment because number of turn is not available from the board
    last_move_played_by = node.current_player_color.opposite
    current_player_turn = node.current_player_color
    board_stats = deepcopy(node.get_stats())
    neighbor_stats = deepcopy(node.get_neighbor_stats())
    
    x = [number_of_turn, last_move_played_by, current_player_turn] + list(board_stats.values())
    for bug in node.get_board_bugs():
      for direction in Direction:
        x.append(neighbor_stats[bug][str(direction)])

    #Creating a DataFrame to store the extracted features
    headers = ['number_of_turn', 'last_move_played_by', 'current_player_turn'] + [f'{key}_moves' for key in list(board_stats.keys())] 
    for bug in node.get_board_bugs():
        for direction in Direction:
            headers.append(f'{bug}_{direction.name}_neighbor')

    X = pd.DataFrame(columns=headers)

    # Adding the extracted features to the dataframe
    X.loc[0] = x

    X = df_preprocessing(X)

    # Predicting the best move
    probabilities = self.model.predict_proba(X)[0]
    prob_black_win = probabilities[0]
    prob_white_win = probabilities[1]

    if node.current_player_color == 'Black':
      return prob_black_win
    else:
      return prob_white_win

  def gen_children(self, parent: Board) -> Set[tuple[Board, str]]:
    """
    Generates valid children from the given parent.

    :param parent: Parent node.
    :type parent: Board
    :return: Set of children.
    :rtype: Set[tuple[Board, str]]
    """
    return {(deepcopy(parent).play(move), move) for move in parent.valid_moves.split(";") if not parent.gameover}

class AlphaBetaPrunerNeuralNetwork(Brain):
  """
  AI agent following an alpha-beta pruning policy.
  """
  def __init__(self) -> None:
    super().__init__()
    self.model = load_model('model/nn_model0.keras')

  def calculate_best_move(self, board: Board, max_depth: int = 0, time_limit: int = 0) -> str:
    if not self._cache:
      self._cache = self.abpruning(board, max_depth, time_limit)
    return self._cache

  def abpruning(self, root: Board, max_depth: int = 0, time_limit: int = 0) -> str:
    """
    Minmax with alpha-beta pruning to search for the best move, given the depth and time constraints.  
    This is an iterative version rather than recursive.

    :param root: Starting node.
    :type root: Board
    :param max_depth: Maximum allowed depth to search at, defaults to 0.
    :type max_depth: int, optional
    :param time_limit: Maximum amount of seconds allowed to spend, defaults to 0.
    :type time_limit: int, optional
    :return: Best move.
    :rtype: str
    """
    start_time = time()

    # Stack to simulate recursion
    stack: list[ABNode] = []
    # Each stack frame contains: (node, depth, alpha, beta, maximizing, child_iter, best_value, best_move)
    stack.append((root, max_depth or float('inf'), float('-inf'), float('inf'), True, iter(self.gen_children(root)), float('-inf'), root.valid_moves.split(";")[0]))

    result: str = root.valid_moves.split(";")[0]

    while stack:
      if time_limit and time() - start_time > time_limit:
        break

      node, depth, alpha, beta, maximizing, child_iter, best_value, best_move = stack.pop()

      if depth == 0 or node.gameover:
        # Evaluate leaf or terminal node
        evaluation = self.evaluate(node)
        if stack:
          # Pass the evaluation result back up the stack
          stack[-1] = self._update_parent(stack[-1], evaluation, best_move)
        else:
          # If this is the root node, store the result
          result = best_move
        continue

      try:
        # Process the next child
        child, move = next(child_iter)
        # Push this node back onto the stack to continue processing later
        stack.append((node, depth, alpha, beta, maximizing, child_iter, best_value, best_move))
        # Push the child onto the stack for processing
        stack.append((child, depth - 1, alpha, beta, not maximizing, iter(self.gen_children(child)), float('-inf') if not maximizing else float('inf'), move))
      except StopIteration:
        # All children processed; finalize the value for this node
        if stack:
          # Pass the result back up the stack
          stack[-1] = self._update_parent(stack[-1], best_value, best_move)
        else:
          # If this is the root node, store the result
          result = best_move

    return result

  def _update_parent(self, parent: ABNode, child_value: float, child_move: str) -> ABNode:
    """
    Updates the given parent with the information from the child.

    :param parent: Parent node to update.
    :type parent: ABNode
    :param child_value: Child node value.
    :type child_value: float
    :param child_move: Move that led to the child node.
    :type child_move: str
    :return: Updated parent node.
    :rtype: ABNode
    """
    # Update the stack frame based on the returned child value
    node, depth, alpha, beta, maximizing, child_iter, best_value, best_move = parent

    if maximizing:
      if child_value > best_value:
        best_value = child_value
        best_move = child_move
      alpha = max(alpha, best_value)
    else:
      best_value = min(best_value, child_value)
      if child_value < best_value:
        best_value = child_value
        best_move = child_move
      beta = min(beta, best_value)

    if beta <= alpha:
      # Clear the iterator to skip remaining children (prune)
      child_iter: Iterator[tuple[Board, str]] = iter([])

    return node, depth, alpha, beta, maximizing, child_iter, best_value, best_move

  def evaluate(self, node: Board) -> float:
    """
    Evaluates the given node.  
    It uses a pre-trained ML model to predict the best move.

    :param node: Playing board.
    :type node: Board
    :return: Node value.
    :rtype: float
    """
    

    # Structure of the data
    # 'number_of_turn','last_move_played_by','current_player_turn',
    # features relative to possible moves for every bug
    # features relative to neighbors for every bug

    # Extracting the features
    
    #---------------------------------------------------OLD CODE BEFORE Board.to_np_array()-------------------------------------------------------------
    # last_move_played_by = node.current_player_color.opposite
    # current_player_turn = node.current_player_color
    # board_stats = deepcopy(node.get_stats())
    # neighbor_stats = deepcopy(node.get_neighbor_stats())
    
    # x = [last_move_played_by, current_player_turn] + list(board_stats.values())
    # for bug in node.get_board_bugs():
    #   for direction in Direction:
    #     x.append(neighbor_stats[bug][str(direction)])

    # #Creating a DataFrame to store the extracted features
    # headers = ['number_of_turn', 'last_move_played_by', 'current_player_turn'] + [f'{key}_moves' for key in list(board_stats.keys())] 
    # for bug in node.get_board_bugs():
    #     for direction in Direction:
    #         headers.append(f'{bug}_{direction.name}_neighbor')

    # X = pd.DataFrame(columns=headers)

    # # Adding the extracted features to the dataframe
    # X.loc[0] = x

    # X = df_preprocessing(X)

    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    #---------------------------------------------------------------------------------------------------------------------------------------
    X = node.to_np_array()

    # Predicting the best move
    prob_white_win = self.model.predict(X)[0]
    prob_black_win = 1 - prob_white_win

    if node.current_player_color == 'Black':
      return prob_black_win
    else:
      return prob_white_win

  def gen_children(self, parent: Board) -> Set[tuple[Board, str]]:
    """
    Generates valid children from the given parent.

    :param parent: Parent node.
    :type parent: Board
    :return: Set of children.
    :rtype: Set[tuple[Board, str]]
    """
    return {(deepcopy(parent).play(move), move) for move in parent.valid_moves.split(";") if not parent.gameover}

class MCTSBrain(Brain):
    """
    MCTS-based AI agent that uses a neural network to evaluate the states.
    """

    def __init__(self, network) -> None:
      super().__init__()
      
      self.network = network  # Neural network to evaluate states

    def calculate_best_move(self, board: 'Board', max_depth: int = 0, time_limit: int = 0) -> str:
      """
      Uses MCTS to calculate the best move, evaluating each state with the neural network.
      """
      move_scores = self._mcts(board)
      
      # Select the move with the highest score
      best_move = max(move_scores, key=move_scores.get)
      
      # Return the best move
      return best_move

    def _mcts(self, board: 'Board') -> dict:
      """
      Perform MCTS for the current board state.
      Uses the neural network to evaluate the value of each potential move.
      """
      move_scores = {}
      return move_scores

    def predict_value(self, current_state: np.ndarray, future_state: np.ndarray) -> float:
      """
      This method predicts the value of a future state given the current state.
      It uses the neural network to predict the value.
      """
      return 0.0

