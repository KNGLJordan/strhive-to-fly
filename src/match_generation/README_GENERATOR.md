## File: `match_generator.py`

### Key Functions

- **`generate_headers(engine: Engine, sample_board: dict) -> List[str]`**  
  Generates column headers for the CSV file that will record the match details. It includes turn information, move statistics from the board, and neighbor data for each bug (game piece) in different directions.

- **`play_match(engine: Engine, arguments: List[str]) -> Optional[List[List[Any]]]`**  
  Plays a single match until the game is over or a maximum turn threshold is reached.  
  - Initializes a new game using the engine.
  - Records board and neighbor statistics at each turn.
  - Randomly selects valid moves and applies an undo mechanism to avoid immediate losses.
  - Returns the match data and a unique match ID.

- **`determine_result(engine: Engine) -> Optional[str]`**  
  Determines the match outcome based on the current game state. Returns `'White'`, `'Black'`, or `'Draw'` accordingly.

- **`save_match_results(rows: List[List[Any]], match_id: int, result: str, file_path: str) -> None`**  
  Saves the collected match statistics and results into a CSV file in the specified directory.

- **`match_generator(engine: Engine, arguments: List[str], num_matches: int) -> None`**  
  The main function that:
  - Runs a specified number of matches.
  - Saves the match results into CSV files in a folder (e.g., `RandomVsRandom` inside the `data` directory).
  - Tracks and prints the win count for both players.

### Example Execution

The module is executable as a script. When run directly, it creates an instance of the game engine, starts a new game with the `"Base"` argument, and runs 5000 matches:

```bash
python match_generator.py