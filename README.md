# StrHiveToFly

StrHiveToFly is a machine learning-based project for playing the board game Hive using AI techniques, including Monte Carlo Tree Search (MCTS) and Deep Learning.

## Project Structure

### **data/**
Contains datasets and match results.
- `RandomVsRandom/`: Games played between random agents.
- `testRandomVsRandom/`: Test matches between random agents.

### **model/**
Stores trained machine learning models used for AI decision-making.

### **src/**
Main source code directory.

#### **core/**
Contains the core logic of the Hive game.
- `ai.py`: AI-related functions.
- `board.py`: Game board representation and mechanics.
- `engine.py`: Game engine and execution logic.
- `enums.py`: Enumerations for game elements.
- `game.py`: Game management and state handling.
- `README_CORE.md`: Documentation for core module.

#### **dl/**
Deep learning-related components.
- `neural_net.py`: Neural network implementation for AI decision-making.

#### **match_generation/**
Scripts for generating game matches.
- `match_generator.py`: Script for creating game match data.
- `README_GENERATOR.md`: Documentation for match generation module.

#### **mcts/**
Implementation of Monte Carlo Tree Search for move decision-making.

#### **ml/**
Machine learning utilities and scripts.
- `ml_classification.ipynb`: Notebook for classifying game states.
- `ml_preprocessing.ipynb`: Notebook for preprocessing data.
- `ml_utils.py`: Utility functions for ML tasks.

### **Configuration & Build Files**
- `.gitignore`: Specifies files to ignore in version control.
- `LICENSE`: Project license.
- `README.md`: Main documentation file (this file).
- `striveMlRandom.spec`, `striveMMRandom.spec`, `striveNNRandom.spec`, `strivetofly.spec`: Build specifications for different AI agents.

## Getting Started
To set up the project:
1. Clone the repository:
   ```bash
   git clone https://github.com/KNGLJordan/strhive-to-fly.git
   cd strhive-to-fly
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the game engine:
   ```bash
   python src/core/engine.py
   ```

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.

