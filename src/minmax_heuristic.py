from board import Board
from enums import PlayerColor, GameState
from ai import Brain

from time import time
from copy import deepcopy

class MinMax(Brain):
  """
  AI agent following an alpha-beta pruning policy.
  """
  def __init__(self, weights: list[float] = [20, 0, 0, 0, 0, 0, 0, 0, 0, 0.5]) -> None:
    super().__init__()

    # Weights for the evaluation function
    if len(weights) != 10:
      raise ValueError("weights list must contain exactly 10 float values")

    # Queen neighbors
    self.k_qn = weights[0]

    # Mobility
    self.k_mq = weights[1]
    self.k_ms = weights[2]
    self.k_mb = weights[3]
    self.k_ma = weights[4]
    self.k_mg = weights[5]
    self.k_mm = weights[6]
    self.k_ml = weights[7]
    self.k_mp = weights[8]

    # Valid moves for player - Valid moves for opponent
    self.k_nm = weights[9]

  def calculate_best_move(self, board: Board, max_depth: int = 3, time_limit: int = 0) -> str:
    best_score, best_move = self.negamax(board, float('-inf'), float('inf'), max_depth)
    return best_move 

  def negamax(self, board: Board, alpha: float, beta: float, max_depth: int = 0, time_limit: int = 0) -> tuple[float, str]:
    """
    Negamax algorithm with alpha-beta pruning to search for the best move, given the depth and time constraints.
    TIME LIMIT NOT IMPLEMENTED
    SORTING NOT IMPLEMENTED
    TRANSPOSITION TABLE NOT IMPLEMENTED
    """
    start_time = time()

    if board.gameover:
        if board.state == GameState.WHITE_WINS:
            if board.current_player_color == PlayerColor.BLACK:
                return float('-inf'), ""
            else:
                return float('inf'), ""
        elif board.state == GameState.BLACK_WINS:
            if board.current_player_color == PlayerColor.WHITE:
                return float('-inf'), ""
            else:
                return float('inf'), ""
        else:
            return 0, ""
      
    if max_depth == 0:
        return self.evaluate(board), ""

    value = float('-inf')
    best_move = ""

    # Creazione della lista di mosse e delle board aggiornate
    moves = board.valid_moves.split(";")
    boards = [deepcopy(board).play(move) for move in moves]

    for move, b in zip(moves, boards):
        score, _ = self.negamax(b, -beta, -alpha, max_depth - 1, time_limit)
        score = -score  # Inversione del valore per negamax

        if score > value:
            value = score
            best_move = move  # Salviamo la mossa migliore trovata finora
        
        alpha = max(alpha, value)
        if alpha >= beta:
            break  # Taglio alpha-beta

    return value, best_move

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

    dict_moves_per_bug_type = node.get_number_moves_per_bugtype()
    present_bugtypes = dict_moves_per_bug_type.keys()

    evaluation = 0

    # Queen neighbors
    evaluation += (node.count_queen_neighbors(maximizing_color) - node.count_queen_neighbors(minimizing_color)) * self.k_qn

    # Mobility
    if maximizing_color == PlayerColor.WHITE:
      evaluation += (dict_moves_per_bug_type['wQ'] - dict_moves_per_bug_type['bQ']) * self.k_mq
      evaluation += (dict_moves_per_bug_type['wS'] - dict_moves_per_bug_type['bS']) * self.k_ms
      evaluation += (dict_moves_per_bug_type['wB'] - dict_moves_per_bug_type['bB']) * self.k_mb
      evaluation += (dict_moves_per_bug_type['wA'] - dict_moves_per_bug_type['bA']) * self.k_ma
      evaluation += (dict_moves_per_bug_type['wG'] - dict_moves_per_bug_type['bG']) * self.k_mg
      if 'wM' in present_bugtypes:
        evaluation += (dict_moves_per_bug_type['wM'] - dict_moves_per_bug_type['bM']) * self.k_mm
      if 'wL' in present_bugtypes:
        evaluation += (dict_moves_per_bug_type['wL'] - dict_moves_per_bug_type['bL']) * self.k_ml
      if 'wP' in present_bugtypes:
        evaluation += (dict_moves_per_bug_type['wP'] - dict_moves_per_bug_type['bP']) * self.k_mp
    else:
      evaluation += (dict_moves_per_bug_type['bQ'] - dict_moves_per_bug_type['wQ']) * self.k_mq
      evaluation += (dict_moves_per_bug_type['bS'] - dict_moves_per_bug_type['wS']) * self.k_ms
      evaluation += (dict_moves_per_bug_type['bB'] - dict_moves_per_bug_type['wB']) * self.k_mb
      evaluation += (dict_moves_per_bug_type['bA'] - dict_moves_per_bug_type['wA']) * self.k_ma
      evaluation += (dict_moves_per_bug_type['bG'] - dict_moves_per_bug_type['wG']) * self.k_mg
      if 'bM' in present_bugtypes:
        evaluation += (dict_moves_per_bug_type['bM'] - dict_moves_per_bug_type['wM']) * self.k_mm
      if 'bL' in present_bugtypes:
        evaluation += (dict_moves_per_bug_type['bL'] - dict_moves_per_bug_type['wL']) * self.k_ml
      if 'bP' in present_bugtypes:
        evaluation += (dict_moves_per_bug_type['bP'] - dict_moves_per_bug_type['wP']) * self.k_mp
    
    # Valid moves for player - Valid moves for opponent
    evaluation += (len(node.calculate_valid_moves_for_player(maximizing_color, True)) - len(node.calculate_valid_moves_for_player(minimizing_color))) * self.k_nm

    return evaluation