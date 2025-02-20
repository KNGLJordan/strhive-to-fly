// Copyright (c) Jon Thysell <http://jonthysell.com>
// Licensed under the MIT License.

#ifndef MINMAX_H
#define MINMAX_H

#include <vector>
#include <string>
#include <limits>
#include <algorithm>
#include <fstream>
#include "Board.h"  // Assumendo che questa sia la classe Board
#include "Enums.h"  // Per PlayerColor e GameState

namespace MzingaCpp 
{
class MinMax 
{
    public:

        MinMax(std::vector<float> weights = {1000, 1, 1, 1, 1, 1, 1, 1, 1, 1});
        std::string calculateBestMove(Board& board, int maxDepth = 3, int timeLimit = 0);

    private:

        float k_qn, k_mq, k_ms, k_mb, k_ma, k_mg, k_mm, k_ml, k_mp, k_nm;
        std::pair<float, std::string> negamax(Board board, int playerColor, float alpha, float beta, int maxDepth, std::atomic<bool>& timeUp);
        float evaluate(Board& board, int playerColor);

};
} // namespace MzingaCpp

#endif // MINMAX_H
