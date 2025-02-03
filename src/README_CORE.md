# strhive-to-fly/src CORE

# Enums Module

The `enums.py` module defines a set of enumerated types (enums) and flags used throughout the project. These enums standardize and document the various commands, options, strategies, player attributes, game states, game types, bug types, and directions used in the game engine. They also include helper methods and properties to facilitate parsing, validation, and transformation of values.

## Enumerations and Flags

### Command
Defines the available commands for the engine. Each command is documented with a short description and a reference to its corresponding section in the Universal Hive Protocol.

- **INFO**:  
  Displays the identifier string of the engine and its capabilities.
- **HELP**:  
  Displays the list of available commands or detailed help for a specific command.
- **OPTIONS**:  
  Displays or sets available engine options.
- **NEWGAME**:  
  Starts a new game (with options for different game types or loading an existing game state).
- **VALIDMOVES**:  
  Lists all valid moves in the current game.
- **BESTMOVE**:  
  Searches for and outputs the best move given time or depth restrictions.
- **PLAY**:  
  Plays a specified move in the current game.
- **PASS**:  
  Plays a pass move.
- **UNDO**:  
  Undoes a specified number of moves.
- **EXIT**:  
  Exits the engine.

### Option
Represents configurable engine options. For example:
- **STRATEGY_WHITE**: AI strategy for player White.
- **STRATEGY_BLACK**: AI strategy for player Black.
- **NUM_THREADS**: Number of threads to parallelize AI computations.

### OptionType
Describes the expected type for each engine option:
- **BOOL**: Boolean.
- **INT**: Integer.
- **FLOAT**: Floating-point (double).
- **ENUM**: Enumerated type.

### Strategy
Lists the possible AI strategies used by the engine:
- **RANDOM**: Random move selection.
- **MINMAX**: Minimax with alpha-beta pruning.
- **MINMAX_ML**: Minimax with alpha-beta pruning and a machine learning evaluation.
- **MINMAX_NN**: Minimax with alpha-beta pruning using a neural network for evaluation.

### PlayerColor
Defines the two possible player colors:
- **WHITE**
- **BLACK**

Includes useful properties:
- **code**: A short, lowercase string representation (e.g., `"w"` for White).
- **opposite**: Returns the opposing color.

### GameState
Represents the state of the game:
- **NOT_STARTED**: The game has not started.
- **IN_PROGRESS**: The game is ongoing.
- **DRAW**: The game ended in a draw.
- **WHITE_WINS**: White has won the game.
- **BLACK_WINS**: Black has won the game.

Also includes a class method `parse()` that converts a string into a corresponding `GameState`.

### GameType
A flag-based enum that indicates the type of game and available expansions. It always includes the base game (Base) and can optionally include expansions:
- **Base**: Base game without expansions.
- **M**: Mosquito expansion.
- **L**: Ladybug expansion.
- **P**: Pillbug expansion.

Provides a `parse()` method to convert a game type string (e.g., `"Base+L"`) into a `GameType` flag. The `__str__` method concatenates the names of the flags.

### BugType
Enumerates the types of bug pieces used in the game. Each type includes a short description of its movement rules and any restrictions:
- **QUEEN_BEE (Q)**
- **SPIDER (S)**
- **BEETLE (B)**
- **GRASSHOPPER (G)**
- **SOLDIER_ANT (A)**
- **MOSQUITO (M)**
- **LADYBUG (L)**
- **PILLBUG (P)**

### Direction
Represents directions on a hexagonal, multi-layered grid. The grid is assumed to be point-up, and directions include:
- **RIGHT ("|-")**
- **UP_RIGHT ("|/")**
- **UP_LEFT ("\\|")**
- **LEFT ("-|")**
- **DOWN_LEFT ("/|")**
- **DOWN_RIGHT ("|\\")**
- **BELOW** and **ABOVE**: Used for vertical layering.

Provides several helper methods and properties:
- **flat()**: Returns a list of all flat (horizontal) directions.
- **flat_left() / flat_right()**: Returns lists of directions leaning to the left or right.
- **opposite**: Computes the opposite direction.
- **left_of** / **right_of**: Computes the direction to the left or right.
- **delta_index**: Returns the index of the direction (based on the order defined in the enum).
- **is_right / is_left**: Boolean properties to check if a direction is considered right or left.

# Board Module

The `board.py` module contains the implementation of the `Board` class, which represents the game board for a strategy board game. This class is responsible for initializing and maintaining the board state, managing piece placements and movements, enforcing game rules, and providing utility methods for move generation and validation. It is a central component that integrates with AI modules, game logic, and machine learning utilities.

## Overview

The `Board` class provides functionalities to:

- **Initialize the board:**  
  Parse a game string to set the game type, state, turn, and initial moves. It also sets up internal mappings for piece positions, neighbors, and move statistics.
  
- **Manage game state:**  
  Track the positions of bug pieces, the move history, and the current game state (e.g., in progress, draw, or win conditions).

- **Generate valid moves:**  
  Compute all legal moves for the current player based on the game rules. This includes handling placements from hand as well as moves for pieces already on the board (with special handling for different bug types).

- **Update and evaluate board state:**  
  Update move statistics and neighbor information for each piece after a move is played. Check for conditions such as a queen being surrounded (a win/loss condition).

- **Support move manipulation:**  
  Execute moves (with the `play` method), undo moves (`undo`), and convert moves to their string representations for logging or further processing.

- **Data conversion:**  
  Convert the current board state into an array or NumPy array, making it suitable for preprocessing before feeding it to machine learning models.

## Key Attributes

- **Constants:**
  - `ORIGIN`: The starting position for the first piece played.
  - `NEIGHBOR_DELTAS`: Offsets defining the neighboring positions in all flat directions.

- **Internal Mappings:**
  - `_bug_to_pos`: Maps each bug piece to its current board position.
  - `_pos_to_bug`: Maps board positions to the bug pieces placed there (supports stacking).
  - `stats`: Tracks the number of moves available for each piece.
  - `neighbors`: Stores the neighboring piece for each bug in each direction.
  - `_valid_moves_cache`: Caches valid moves for each player to improve performance.

- **Move History:**
  - `move_strings`: A list of moves played in string format.
  - `moves`: A list of move objects representing the history of moves.

## Key Methods

- **Initialization and Parsing:**
  - `__init__(gamestring: str = "")`:  
    Initializes the board by parsing a game string, setting up the game type, state, turn number, and initial moves.
    
  - `_parse_gamestring(gamestring: str)`:  
    Breaks down the game string into its components (game type, state, turn, and moves).
    
  - `_play_initial_moves(moves: list[str])`:  
    Plays the initial moves as specified in the game string to bring the board to its current state.

- **State Representation:**
  - `__str__()`:  
    Returns a string representation of the board state, including the game type, state, current player turn, and move history.

- **Properties:**
  - `current_player_color`: Returns the color of the current player.
  - `current_player_turn`: Returns the turn number for the current player.
  - `current_player_queen_in_play`: Indicates whether the current player's queen is already on the board.
  - `gameover`: Indicates whether the game has ended (win, draw, etc.).
  - `valid_moves`: Returns a string of all valid moves available to the current player.

- **Move Generation and Execution:**
  - `calculate_valid_moves_for_player(color: PlayerColor, force: bool = False)`:  
    Computes the set of legal moves for a given player based on the current board state.
    
  - `play(move_string: str)`:  
    Executes a move (given as a move string), updating the board state and internal mappings accordingly. It also checks and updates the game state (e.g., for win or draw conditions).
    
  - `undo(amount: int = 1)`:  
    Reverts a specified number of moves, restoring the board to a previous state.
    
  - `stringify_move(move: Optional[Move])`:  
    Converts a move object into its string representation.

- **Utility Methods:**
  - `update_utility_dictionaries()`:  
    Updates the `stats` and `neighbors` dictionaries based on the current positions of the bug pieces.
    
  - `count_queen_neighbors(color: PlayerColor)`:  
    Counts how many neighboring pieces are adjacent to a player's queen (used to determine if the queen is surrounded).
    
  - `get_stats()`, `get_neighbor_stats()`, `get_board_bugs()`:  
    Provide statistical data and information on the board state, such as move counts and neighbor relationships.

- **Movement and Position Helpers:**
  - Methods like `_get_valid_placements_for_color()`, `_get_sliding_moves()`, `_get_beetle_moves()`, `_get_grasshopper_moves()`, `_get_mosquito_moves()`, `_get_ladybug_moves()`, and `_get_pillbug_special_moves()`  
    Compute valid moves for different bug types based on their unique movement rules.
    
  - `_get_neighbor(position: Position, direction: Direction)`:  
    Calculates the neighboring position from a given position in a specified direction.
    
  - `_bugs_from_pos(position: Position)`, `_pos_from_bug(bug: Bug)`:  
    Retrieve the list of bugs at a given position or the position of a specific bug.

- **Data Conversion:**
  - `to_array()`:  
    Converts the board state to a simple list representation (includes move counts and neighbor data).
    
  - `to_np_array()`:  
    Converts the board state to a NumPy array (after preprocessing), suitable for input into AI or machine learning models.

## Dependencies

This module depends on several other parts of the project and external libraries:
- **Project Modules:**
  - `enums.py`: Defines `GameType`, `GameState`, `PlayerColor`, `BugType`, and `Direction`.
  - `game.py`: Contains definitions for `Position`, `Bug`, and `Move`.
  - `ml_utils.py`: Provides functions (e.g., `np_preprocessing`) for data processing.
  
- **External Libraries:**
  - `numpy` for numerical operations.
  - Standard Python libraries such as `re`, `copy`, and `typing`.

# Engine Module

The `engine.py` module implements the main game engine for the project. It manages the overall game flow, interprets user commands, and integrates various components such as the game board, AI agents, and game state management. This engine is designed to handle a Hive-like board game and follows the Universal Hive Protocol for commands and interactions.

## Overview

The central class in this module is the `Engine` class. It is responsible for:

- **Initializing the Game:**  
  Setting up a new game using the `Board` class with a specified game string. It also loads default options, strategies, and AI agents (brains).

- **Command Handling:**  
  Processing user input commands (such as `info`, `help`, `newgame`, `play`, `undo`, etc.) in a loop. The engine uses pattern matching to parse and execute commands according to the Universal Hive Protocol.

- **Options Management:**  
  Maintaining a set of configurable options (like the strategies for White and Black and the number of threads). It provides helper methods to get and set options with proper type and value checks.

- **Integration with AI Agents:**  
  Mapping game strategies to corresponding AI agents (brains) such as `Random`, `AlphaBetaPruner`, `AlphaBetaPrunerMLClassifier`, and `AlphaBetaPrunerNeuralNetwork`. The engine uses these agents to calculate the best move based on user commands.

- **Game Flow Control:**  
  Managing the game state, including starting a new game, listing valid moves, playing moves (or passes), and undoing moves. It also supports a specialized method (`play_match_generator`) for batch match generation where board output is suppressed for performance.

- **Error Handling:**  
  Providing user-friendly error messages when invalid commands or arguments are received.

## Key Attributes and Methods

### Attributes

- **Versioning and Options:**  
  - `VERSION`: The version of the engine (e.g., "1.1.0").
  - `OPTION_TYPES`: A dictionary mapping options to their expected types (e.g., strategy options and number of threads).
  - `BRAINS`: A mapping of available strategies to their corresponding AI agent instances.
  - Default option values such as `DEFAULT_STRATEGY_WHITE`, `DEFAULT_STRATEGY_BLACK`, and thread count options are defined as class-level constants.

- **Instance Attributes:**  
  - `strategywhite` and `strategyblack`: Current strategies for the White and Black players.
  - `numthreads`: The number of threads to use (based on CPU count).
  - `brains`: A dictionary mapping `PlayerColor` to their selected AI agent.
  - `board`: The current active game board (an instance of `Board`).

### Methods

- **Command Loop:**  
  - `start()`:  
    The main loop that prints engine information and continuously waits for user input. Commands are parsed using pattern matching (with Python’s `match` statement) and dispatched to appropriate handler methods.

- **Command Handlers:**  
  - `info()`: Displays engine version and capabilities.
  - `help(arguments: list[str])`: Provides help information for a specific command or a list of available commands.
  - `options(arguments: list[str])`: Gets or sets engine options.
  - `newgame(arguments: list[str])`: Creates a new game by instantiating the `Board` with a provided game string.
  - `validmoves()`: Prints the valid moves available in the current game.
  - `bestmove(restriction: str, value: str)`: Uses the current player’s AI agent to search for and output the best move under either a time or depth restriction.
  - `play(move: str)` and `play_match_generator(move: str)`: Executes a move in the current game; the latter is optimized for batch processing (e.g., for match generation).
  - `undo(arguments: list[str])`: Reverts the last move (or a specified number of moves).

- **Utility Methods:**  
  - `is_active(board: Optional[Board])`: Checks whether a game board is active before processing certain commands.
  - `error(error: str | Exception)`: Outputs error messages in a consistent format.
  - Overloaded `__getitem__` and `__setitem__` allow access to engine attributes using dictionary-like syntax.

# Game Module

The `game.py` module provides core classes that represent the fundamental game elements: tile positions, bug pieces, and moves. These classes are used throughout the project to parse and represent game state and user input.

## Classes

### Position
Represents a tile position on the game board.

- **Constructor:**  
  `Position(q: int, r: int)`  
  Creates a new position with axial coordinates `q` and `r`.

- **Methods and Operators:**
  - `__str__`: Returns a string representation of the position in the format `"(q, r)"`.
  - `__hash__`: Supports using `Position` objects as dictionary keys by hashing the `(q, r)` tuple.
  - `__eq__`: Compares two positions for equality based on their coordinates.
  - `__add__` and `__sub__`: Allow addition and subtraction of two positions, resulting in a new `Position`.

### Bug
Represents a bug piece used in the game.

- **Class Attributes:**
  - `COLORS`: A mapping from short color codes (derived from `PlayerColor`) to `PlayerColor` objects.
  - `REGEX`: A regular expression pattern used to validate and parse BugStrings. The BugString format includes a color code, bug type (from `BugType`), and an optional identifier (e.g., "wQ1" for a White Queen Bee with ID 1).

- **Static Methods:**
  - `parse(bug: str) -> Bug`:  
    Parses a BugString using the defined regex. If the string is valid, returns a `Bug` instance; otherwise, it raises a `ValueError`.

- **Constructor:**  
  `Bug(color: PlayerColor, bug_type: BugType, bug_id: int = 0)`  
  Creates a new bug with the specified color, type, and optional ID.

- **Methods and Operators:**
  - `__str__`: Returns a string representation combining the bug’s color code, type, and ID (if provided).
  - `__hash__`: Allows bugs to be used as dictionary keys by hashing their string representation.
  - `__eq__`: Compares two bugs for equality based on their color, type, and identifier.

### Move
Represents a move in the game.

- **Class Attributes:**
  - `PASS`: A constant string (`"pass"`) used to indicate a passing move.
  - `REGEX`: A regular expression pattern for validating MoveStrings. The MoveString format encodes the bug being moved and, optionally, a relative bug and direction information.

- **Static Methods:**
  - `stringify(moved: Bug, relative: Optional[Bug] = None, direction: Optional[Direction] = None) -> str`:  
    Converts move data into a MoveString. If a relative bug and a direction are provided, they are included in the resulting string; otherwise, only the moved bug is represented.

- **Constructor:**  
  `Move(bug: Bug, origin: Optional[Position], destination: Position)`  
  Initializes a move with the bug to be moved, its origin position (which can be `None` if the piece is being placed from hand), and its destination position.

- **Methods and Operators:**
  - `__str__`: Returns a string representation of the move, including the bug, origin, and destination.
  - `__hash__`: Allows moves to be used as keys in dictionaries by hashing a tuple of `(bug, origin, destination)`.
  - `__eq__`: Compares two moves for equality by checking that their bug, origin, and destination are the same.

# AI Module

The `ai.py` module implements various AI agents for determining the best move in a board game simulation. Each agent inherits from the abstract base class `Brain` and provides its own strategy for move selection. These agents range from simple random move selectors to more sophisticated algorithms using alpha-beta pruning, machine learning classifiers, and neural networks.

## Contents

- **`Brain` (Abstract Base Class):**  
  Defines the interface for all AI agents. It contains the abstract method `calculate_best_move()` and a cache-clearing utility `empty_cache()`.

- **`Random`:**  
  A simple AI agent that selects a move at random from the available valid moves. It simulates a small computation delay using `sleep`.

- **`AlphaBetaPruner`:**  
  Implements the alpha-beta pruning algorithm using an iterative, stack-based minimax search. Key components include:
  - **`abpruning()`:** Iteratively searches for the best move while pruning suboptimal branches.
  - **`_update_parent()`:** Updates parent nodes with the evaluation results of child nodes.
  - **`evaluate()`:** Provides a heuristic evaluation of a board state based on piece positions and mobility.
  - **`gen_children()`:** Generates all valid child board states from a given parent state.

- **`AlphaBetaPrunerMLClassifier`:**  
  Similar to `AlphaBetaPruner`, this agent uses a pre-trained machine learning model (loaded from a pickle file) to evaluate board states. The evaluation function:
  - Extracts relevant features from the board (including board and neighbor statistics).
  - Uses the model’s `predict_proba` method to estimate winning probabilities.
  - Adjusts the evaluation based on the current player’s color.

- **`AlphaBetaPrunerNeuralNetwork`:**  
  An extension of the alpha-beta pruning approach that leverages a neural network (loaded via Keras) for evaluation. This agent:
  - Converts the board state into a NumPy array using `Board.to_np_array()`.
  - Uses the neural network to predict winning probabilities.
  - Integrates the neural network evaluation into the alpha-beta pruning framework.

- **`MCTSBrain`:**  
  A placeholder for an MCTS-based AI agent that integrates a neural network for state evaluation. It outlines methods for:
  - Running Monte Carlo Tree Search (`_mcts()`) to evaluate potential moves.
  - Predicting the value of future states (`predict_value()`).
  Currently, the implementation for MCTS is minimal and intended for future expansion.

## Dependencies

- **Data Processing and Machine Learning:**
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `tensorflow.keras` (for neural network loading)
  - `torch` (PyTorch for potential deep learning implementations)
- **Standard Libraries:**
  - `random`, `time`, `copy`, `abc`, `pickle`
- **Project Modules:**
  - `board` (defines the game board and its operations)
  - `enums` (contains various enumerated types used across the project)
  - `ml_utils` (for data preprocessing functions like `df_preprocessing`)

Make sure that the pre-trained models are available in the expected directories:
- ML classifier: `model/model0.pkl`
- Neural network model: `model/nn_model0.keras`
