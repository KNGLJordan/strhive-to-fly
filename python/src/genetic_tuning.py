# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import random
import multiprocessing
from copy import deepcopy

from engine import Engine
from enums import PlayerColor, GameState
from minmax_heuristic import MinMax
import time


class GameAgent:
    def __init__(self, weights: list[float]):
        self.weights = weights


def play_match(agent1, agent2) -> str:
    """Simula una partita tra due agenti e restituisce il vincitore ('White', 'Black' o 'Draw')."""
    engine = Engine()
    engine.brains = {
        PlayerColor.WHITE: MinMax(agent1.weights),
        PlayerColor.BLACK: MinMax(agent2.weights),
    }
    engine.newgame(["Base"])
    turn = 1

    while not engine.board.gameover:
        start_time = time.time()
        move = engine.brains[engine.board.current_player_color].calculate_best_move(engine.board, max_depth=1)
        engine.play_match_generator(move)
        end_time = time.time()
        #print(f'Turn: {turn}, Time: {end_time - start_time:.2f} seconds')
        turn += 1
        if turn > 300:
            print("Draw")
            return "Draw"

    if engine.board.state == GameState.WHITE_WINS:
        print("White")
        return "White"
    elif engine.board.state == GameState.BLACK_WINS:
        print("Black")
        return "Black"
    else:
        return "Draw"


def evaluate_fitness(candidate_weights, opponents_weights):
    """
    Funzione helper per il multiprocessing: calcola la fitness di un candidato.
    """
    wins = 0
    total_matches = 2 * len(opponents_weights)  # Due partite per ogni avversario

    candidate = GameAgent(weights=candidate_weights)

    for opponent_weights in opponents_weights:
        opponent = GameAgent(weights=opponent_weights)

        result1 = play_match(candidate, opponent)
        if result1 == "White":
            wins += 1

        result2 = play_match(opponent, candidate)
        if result2 == "Black":
            wins += 1

    return wins / total_matches


class GeneticTuner:
    """
    Classe per il tuning dei pesi dell'euristica con algoritmo genetico,
    parallelizzando le partite per accelerare la valutazione della fitness.
    """

    def __init__(self, population_size: int, num_generations: int, mutation_rate: float,
                 crossover_rate: float, movement_weight_bounds: tuple[float, float], 
                 queen_neigh_weight_bounds: tuple[float, float], num_weights: int,
                 num_opponents: int = 3):
        """
        Inizializza il GeneticTuner.

        :param population_size: Dimensione della popolazione.
        :param num_generations: Numero di generazioni da eseguire.
        :param mutation_rate: Probabilità di mutazione per ciascun gene.
        :param crossover_rate: Probabilità di crossover fra due individui.
        :param movement_weight_bounds: Tuple (min, max) per i valori dei pesi di movimento.
        :param queen_neigh_weight_bounds: Tuple (min, max) per i valori dei pesi di vicinanza della regina.
        :param num_weights: Numero di pesi (dimensione del cromosoma).
        :param num_opponents: Numero di avversari scelti casualmente per valutare ogni candidato.
        """
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.movement_weight_bounds = movement_weight_bounds
        self.queen_neigh_weight_bounds = queen_neigh_weight_bounds
        self.num_weights = num_weights
        self.num_opponents = num_opponents

        # Inizializza la popolazione con cromosomi casuali
        self.population = [self.random_chromosome() for _ in range(population_size)]

    def random_chromosome(self) -> list[float]:
        """Genera un cromosoma casuale con pesi nei limiti specificati."""
        return [random.uniform(*self.queen_neigh_weight_bounds)] + \
               [random.uniform(*self.movement_weight_bounds) for _ in range(self.num_weights - 1)]

    def compute_fitness_parallel(self):
        """
        Calcola la fitness di tutta la popolazione in parallelo.
        """
        with multiprocessing.Pool() as pool:
            opponents = [random.sample(self.population, self.num_opponents) for _ in self.population]
            fitness_scores = pool.starmap(evaluate_fitness, zip(self.population, opponents))

        return fitness_scores

    def selection(self, fitness_scores) -> list[list[float]]:
        """
        Seleziona due cromosomi dalla popolazione in base alla fitness tramite torneo.
        """
        selected = []
        for _ in range(2):
            tournament_indices = random.sample(range(self.population_size), 3)
            best_index = max(tournament_indices, key=lambda i: fitness_scores[i])
            selected.append(deepcopy(self.population[best_index]))
        return selected

    def crossover(self, parent1: list[float], parent2: list[float]) -> tuple[list[float], list[float]]:
        """
        Effettua il crossover a un punto casuale con probabilità crossover_rate.
        """
        if random.random() < self.crossover_rate:
            point = random.randint(1, self.num_weights - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        else:
            return deepcopy(parent1), deepcopy(parent2)

    def mutate(self, chromosome: list[float]) -> list[float]:
        """
        Applica una mutazione con probabilità mutation_rate per ogni peso.
        """
        return [random.uniform(*self.queen_neigh_weight_bounds) if i == 0 and random.random() < self.mutation_rate else
                random.uniform(*self.movement_weight_bounds) if random.random() < self.mutation_rate else w
                for i, w in enumerate(chromosome)]

    def evolve(self):
        """
        Esegue l'algoritmo genetico per num_generations generazioni.
        """
        for generation in range(self.num_generations):
            # Calcola la fitness in parallelo
            fitness_scores = self.compute_fitness_parallel()

            new_population = []
            while len(new_population) < self.population_size:
                parent1, parent2 = self.selection(fitness_scores)
                child1, child2 = self.crossover(parent1, parent2)
                new_population.extend([self.mutate(child1), self.mutate(child2)])

            # Mantieni la dimensione della popolazione
            self.population = new_population[:self.population_size]

            # Trova il miglior candidato della generazione
            best_index = max(range(self.population_size), key=lambda i: fitness_scores[i])
            best_fitness = fitness_scores[best_index]
            print(f"Generazione {generation + 1}: Miglior fitness = {best_fitness}")

        # Restituisce il miglior cromosoma della popolazione
        final_fitness_scores = self.compute_fitness_parallel()
        best_index = max(range(self.population_size), key=lambda i: final_fitness_scores[i])
        return self.population[best_index]


if __name__ == '__main__':
    POPULATION_SIZE = 10
    NUM_GENERATIONS = 10
    MUTATION_RATE = 0.1
    CROSSOVER_RATE = 0.7
    MOVEMENT_WEIGHT_BOUNDS = (0.0, 2.0)
    QUEEN_NEIGH_WEIGHT_BOUNDS = (1500.0, 500.0)
    NUM_WEIGHTS = 10
    NUM_OPPONENTS = 3

    tuner = GeneticTuner(population_size=POPULATION_SIZE,
                         num_generations=NUM_GENERATIONS,
                         mutation_rate=MUTATION_RATE,
                         crossover_rate=CROSSOVER_RATE,
                         movement_weight_bounds=MOVEMENT_WEIGHT_BOUNDS,
                         queen_neigh_weight_bounds=QUEEN_NEIGH_WEIGHT_BOUNDS,
                         num_weights=NUM_WEIGHTS,
                         num_opponents=NUM_OPPONENTS)

    best_weights = tuner.evolve()
    print("Migliori pesi trovati:", best_weights)
    #saving the best weights found in a file in folder 'data/genetic_tuning/'
    with open(os.path.join(os.path.dirname(__file__), '../..', 'data/genetic_tuning/best_weights.txt'), 'w') as f:
        f.write(" ".join(map(str, best_weights)))
