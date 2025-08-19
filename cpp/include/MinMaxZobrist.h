#ifndef MINMAXZOBRIST_H
#define MINMAXZOBRIST_H

#include <vector>
#include <string>
#include <limits>
#include <algorithm>
#include <fstream>
#include "Board.h"  
#include "Enums.h"  
#include "ZobristHasher.h"  // Includi l'header di ZobristHasher

namespace MzingaCpp 
{
struct EvaluationWeights {
    // Queen safety
    float k_qn = 1000.0f;  // Queen neighbors (existing)
    
    // Mobility weights (existing)
    float k_mq = 1.0f;  // Queen mobility
    float k_ms = 1.0f;  // Spider mobility
    float k_mb = 1.0f;  // Beetle mobility
    float k_ma = 1.0f;  // Ant mobility
    float k_mg = 1.0f;  // Grasshopper mobility
    float k_mm = 1.0f;  // Mosquito mobility
    float k_ml = 1.0f;  // Ladybug mobility
    float k_mp = 1.0f;  // Pillbug mobility
    float k_nm = 1.0f;  // Total moves
    
    // New BoardMetrics-based weights
    float k_pieces_in_play = 0.5f;     // Reward having pieces on board
    float k_pieces_in_hand = -0.3f;    // Penalty for pieces still in hand
    float k_pinned_pieces = -2.0f;     // Heavy penalty for pinned pieces
    float k_covered_pieces = -1.0f;    // Penalty for covered pieces
    float k_noisy_moves = 1.2f;        // Reward threatening moves
    float k_quiet_moves = 0.8f;        // Reward non-threatening mobility
    float k_friendly_neighbors = 0.4f; // Reward piece coordination
    float k_enemy_neighbors = 0.6f;    // Reward attacking enemy pieces
    
    // Piece-specific strategic weights
    float k_queen_safety = -3.0f;      // Extra penalty for queen being threatened
    float k_ant_mobility = 1.5f;       // Ants are very valuable when mobile
    float k_beetle_coverage = 0.8f;    // Beetles covering pieces is good
    float k_spider_positioning = 0.6f; // Spiders in good positions
};

enum class Flag { EXACT, LOWER_BOUND, UPPER_BOUND };

struct TTEntry {
    float value;          
    std::string bestMove; 
    int depth;            
    Flag flag;            
};   

class MinMaxZobrist 
{
    public:
        MinMaxZobrist(bool useEnhancedEval = false);
        MinMaxZobrist(const EvaluationWeights& weights, bool useEnhancedEval = true);
        MinMaxZobrist(std::vector<float> weights, bool useEnhancedEval = false);
        float evaluateFast(Board& board, int playerColor) ;
        virtual std::string calculateBestMove(Board& board, int maxDepth = 3, int timeLimit = 0);

        
        bool useEnhanced = false;
        ZobristHasher zobristHasher;  
        std::unordered_map<uint64_t, TTEntry> transpositionTable;
        EvaluationWeights weights;
        float evaluate(Board& board, int playerColor);
        float evaluate2(Board& board, int playerColor);

    private:   

        // float k_qn, k_mq, k_ms, k_mb, k_ma, k_mg, k_mm, k_ml, k_mp, k_nm;
        // float k_pieces_in_play, k_pieces_in_hand, k_pinned_pieces, k_covered_pieces;
        // float k_noisy_moves, k_quiet_moves, k_friendly_neighbors, k_enemy_neighbors;
        // float k_queen_safety, k_ant_mobility, k_beetle_coverage, k_spider_positioning;

        
        std::pair<float, std::string> negamax(Board board, int playerColor, float alpha, float beta, int maxDepth, std::atomic<bool>& timeUp);
        std::pair<float, std::string> negamaxStats(Board board, int playerColor, float alpha, float beta, int maxDepth, std::atomic<bool>& timeUp, std::atomic<int>& nodeCounter);

};
} // namespace MzingaCpp

#endif // MINMAXZOBRIST_H
