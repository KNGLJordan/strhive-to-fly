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
class MinMaxZobrist 
{
    public:
        MinMaxZobrist(std::vector<float> weights = {1000, 1, 1, 1, 1, 1, 1, 1, 1, 1});
        std::string calculateBestMove(Board& board, int maxDepth = 3, int timeLimit = 0);

    private:
        enum class Flag { EXACT, LOWER_BOUND, UPPER_BOUND };
        struct TTEntry {
            float value;          
            std::string bestMove; 
            int depth;            
            Flag flag;            
        };        
        float k_qn, k_mq, k_ms, k_mb, k_ma, k_mg, k_mm, k_ml, k_mp, k_nm;
        ZobristHasher zobristHasher;  
        std::unordered_map<uint64_t, TTEntry> transpositionTable;
        std::pair<float, std::string> negamax(Board board, int playerColor, float alpha, float beta, int maxDepth, std::atomic<bool>& timeUp);
        std::pair<float, std::string> negamaxStats(Board board, int playerColor, float alpha, float beta, int maxDepth, std::atomic<bool>& timeUp, std::atomic<int>& nodeCounter);
        float evaluate(Board& board, int playerColor);
};
} // namespace MzingaCpp

#endif // MINMAXZOBRIST_H
