// genetic_tuner.cpp
// Script per ottimizzare i pesi della funzione di valutazione di MinMax con un Genetic Algorithm

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <limits>
#include <chrono>

#include "Board.h"       // Classe che gestisce lo stato del gioco, mosse valide, applica mosse, verifica fine partita
#include "MinMax.h"     // Classe MinMax con metodi calculateBestMove
#include "Enums.h"      // Enum per GameType, Color, BoardState

using namespace std;
using namespace MzingaCpp;

// Struttura per rappresentare un individuo (insieme di pesi) e il suo fitness
struct Individual {
    vector<float> weights;    // I pesi da ottimizzare: dimensione fissa (10)
    float fitness;
};

// Parametri del Genetic Algorithm
const int POP_SIZE = 3;
const int NUM_GENERATIONS = 2;
const float MUTATION_RATE = 0.1f;
const float MUTATION_STDDEV = 0.2f;
const int TOURNAMENT_SIZE = 1;
const int NUM_GAMES_PER_INDIVIDUAL = 2; // Es. 2 come bianco, 2 come nero contro baseline

// Pesi di default (baseline) per valutazione
const vector<float> BASELINE_WEIGHTS = {1000, 1, 1, 1, 1, 1, 1, 1, 1, 1};

// Generatore random globale
std::mt19937 rng( std::chrono::high_resolution_clock::now().time_since_epoch().count() );

// Funzione per inizializzare la popolazione con pesi random
vector<Individual> initializePopulation() {
    vector<Individual> pop;
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    for(int i = 0; i < POP_SIZE; ++i) {
        Individual ind;
        ind.weights.resize(BASELINE_WEIGHTS.size());
        for(size_t j = 0; j < ind.weights.size(); ++j) {
            // Centriamo intorno al peso di base con variazione
            ind.weights[j] = BASELINE_WEIGHTS[j] + dist(rng);
        }
        ind.fitness = 0.0f;
        pop.push_back(ind);
    }
    return pop;
}

// Funzione per far giocare due agenti MinMax e restituire il punteggio (1 vittoria bianco, -1 vittoria nero, 0 pareggio)
int playGame(const vector<float>& weightsA, const vector<float>& weightsB, bool A_is_white) {
    Board board(GameType::Base);
    Color currentPlayer = Color::White;

    MinMax playerA(weightsA);
    MinMax playerB(weightsB);

    int turn = 0;

    while (!GameIsOver(board.GetBoardState())) {

        if(turn>=100){
            return 0;
        }

        string moveStr;
        if (currentPlayer == Color::White) {
            moveStr = A_is_white ? playerA.calculateBestMove(board, 0, 5)
                                 : playerB.calculateBestMove(board, 0, 5);
        } else {
            moveStr = A_is_white ? playerB.calculateBestMove(board, 0, 5)
                                 : playerA.calculateBestMove(board, 0, 5);
        }

        if (moveStr.empty()) {
            moveStr = "pass";  // fallback, oppure gestiscilo a monte
        }

        Move parsedMove;
        string parsedStr;
        if (board.TryParseMove(moveStr, parsedMove, parsedStr)) {
            board.TryPlayMove(parsedMove, moveStr);
        } else {
            cerr << "Errore parsing mossa: " << moveStr << endl;
            break;
        }

        currentPlayer = (currentPlayer == Color::White) ? Color::Black : Color::White;

        turn++;
    }

    BoardState finalState = board.GetBoardState();
    if (finalState == BoardState::WhiteWins) return 1;
    if (finalState == BoardState::BlackWins) return -1;
    return 0;
}


// Funzione di valutazione di un individuo: media del risultato su NUM_GAMES_PER_INDIVIDUAL partite contro baseline
float evaluateIndividual(const Individual& ind) {
    float total = 0.0f;
    // Alternare colori: metà delle partite come bianco, metà come nero
    for(int i = 0; i < NUM_GAMES_PER_INDIVIDUAL; ++i) {
        cout << "Playing game #" << (i + 1) << " ..." << endl;
        bool indWhite = (i % 2 == 0);
        int result = playGame(ind.weights, BASELINE_WEIGHTS, indWhite);
        // Se l'individuo era nero (indWhite==false), invertire il punteggio
        if(!indWhite) result = -result;
        total += result;
    }
    cout << "Ratio: " << total / NUM_GAMES_PER_INDIVIDUAL << endl;
    return total / NUM_GAMES_PER_INDIVIDUAL;
}

// Selezione tramite torneo
Individual tournamentSelection(const vector<Individual>& pop) {
    std::uniform_int_distribution<int> dist(0, POP_SIZE - 1);
    Individual best;
    best.fitness = -std::numeric_limits<float>::infinity();
    for(int i = 0; i < TOURNAMENT_SIZE; ++i) {
        int idx = dist(rng);
        if(pop[idx].fitness > best.fitness) {
            best = pop[idx];
        }
    }
    return best;
}

// Crossover a singolo punto
pair<Individual, Individual> crossover(const Individual& parent1, const Individual& parent2) {
    std::uniform_int_distribution<int> dist(1, parent1.weights.size() - 2);
    int cp = dist(rng);
    Individual child1, child2;
    child1.weights.resize(parent1.weights.size());
    child2.weights.resize(parent1.weights.size());
    for(size_t i = 0; i < parent1.weights.size(); ++i) {
        if((int)i < cp) {
            child1.weights[i] = parent1.weights[i];
            child2.weights[i] = parent2.weights[i];
        } else {
            child1.weights[i] = parent2.weights[i];
            child2.weights[i] = parent1.weights[i];
        }
    }
    child1.fitness = 0.0f;
    child2.fitness = 0.0f;
    return {child1, child2};
}

// Mutazione gaussiana
void mutate(Individual& ind) {
    std::normal_distribution<float> dist(0.0f, MUTATION_STDDEV);
    for(auto &w : ind.weights) {
        if(std::uniform_real_distribution<float>(0.0f, 1.0f)(rng) < MUTATION_RATE) {
            w += dist(rng);
        }
    }
}

int main() {
    // Inizializza popolazione
    cout << "Inizializzazione popolazione...\n";
    auto population = initializePopulation();

    cout << "Popolazione inizializzata con " << POP_SIZE << " individui.\n";
    for(int gen = 0; gen < NUM_GENERATIONS; ++gen) {
        // Valuta ogni individuo
        cout << "Valutazione della generazione " << gen << "...\n";
        static int evalCounter = 0;
        for(auto &ind : population) {
            cout << "Individuo #" << (++evalCounter) << endl;
            ind.fitness = evaluateIndividual(ind);
            cout << " valutato: fitness = " << ind.fitness << endl;
        }

        // Ordina popolazione per fitness decrescente (facoltativo per analisi)
        sort(population.begin(), population.end(), [](const Individual& a, const Individual& b) {
            return a.fitness > b.fitness;
        });

        // Stampa miglior individuo della generazione
        cout << "Generazione " << gen << " - Miglior fitness: " << population[0].fitness << "\n";

        // Nuova popolazione
        vector<Individual> newPop;
        // Elitismo: preserva top 2
        newPop.push_back(population[0]);
        newPop.push_back(population[1]);

        // Genera figli finché non si raggiunge la dimensione di popolazione
        while((int)newPop.size() < POP_SIZE) {
            Individual parent1 = tournamentSelection(population);
            Individual parent2 = tournamentSelection(population);
            auto [child1, child2] = crossover(parent1, parent2);
            mutate(child1);
            mutate(child2);
            newPop.push_back(child1);
            if((int)newPop.size() < POP_SIZE) newPop.push_back(child2);
        }

        population = move(newPop);
    }

    // Dopo l'ultima generazione, ridimensiona e valuta ancora
    for(auto &ind : population) {
        ind.fitness = evaluateIndividual(ind);
    }
    sort(population.begin(), population.end(), [](const Individual& a, const Individual& b) {
        return a.fitness > b.fitness;
    });

    // Stampa i migliori pesi trovati
    cout << "\\nMigliori pesi trovati:\n";
    for(size_t i = 0; i < population[0].weights.size(); ++i) {
        cout << population[0].weights[i] << (i + 1 < population[0].weights.size() ? ", " : "\n");
    }

    return 0;
}
