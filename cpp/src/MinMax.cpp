#include <vector>
#include <string>
#include <limits>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <atomic>
#include <thread>

#include "Board.h"  // Assumendo che questa sia la classe Board
#include "Enums.h"  // Per PlayerColor e GameState
#include "MinMax.h"

using namespace MzingaCpp;


MinMax::MinMax(std::vector<float> weights) 
{
    if (weights.size() != 10) 
    {
        throw std::invalid_argument("weights list must contain exactly 10 float values");
    }
    
    k_qn = weights[0];
    k_mq = weights[1]; k_ms = weights[2]; k_mb = weights[3]; k_ma = weights[4];
    k_mg = weights[5]; k_mm = weights[6]; k_ml = weights[7]; k_mp = weights[8];
    k_nm = weights[9];
}

std::string MinMax::calculateBestMove(Board& board, int maxDepth, int timeLimit) 
{    
    float bestScore;
    std::string bestMove;

    int currentTurn; // 1 for white, -1 for black

    std::atomic<bool> timeUp(false);

    // if the turn is even, it's white's turn, otherwise it's black's turn
    if(board.GetCurrentTurn()%2==1) 
    {
        currentTurn = -1; //black
    } 
    else 
    { 
        currentTurn = 1; //white
    }

    if(timeLimit==0)
    {
        std::tie(bestScore, bestMove) = negamax(board, currentTurn, -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), maxDepth, timeUp);
    }
    else
    {
        //---------------------------------ITERATIVE DEEPENING APPROACH---------------------------------------------

        

        auto startTime = std::chrono::steady_clock::now();
        int depth = 1;

        std::thread timerThread([&]() {
            std::this_thread::sleep_for(std::chrono::seconds(timeLimit));
            timeUp = true;
        });

        while (!timeUp) {
            float score;
            std::string move;
            std::tie(score, move) = negamax(board, currentTurn, -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), depth, timeUp);
            
            if (!move.empty()) {
                bestScore = score;
                bestMove = move;
            }

            depth++;
        }
        
        timerThread.join();

        std::ofstream log("/home/francesco/Desktop/franz/scuolaOrtogonale/hive/strhive-to-fly/log/logcpp.txt", std::ios::app);
        log << "reached"<< " " << depth<< "\n";
        log.close();
    }
    
    return bestMove;
}

float k_qn, k_mq, k_ms, k_mb, k_ma, k_mg, k_mm, k_ml, k_mp, k_nm;

std::pair<float, std::string> MinMax::negamax(Board board, int playerColor, float alpha, float beta, int maxDepth, std::atomic<bool>& timeUp) {
    if (timeUp) return {0, ""};
    
    BoardState state = board.GetBoardState();

    // ------------------------------------ TERMINAL NODES -----------------------------------------------
    if (state == BoardState::WhiteWins) 
    {
        return (playerColor == 0) ? std::make_pair(-std::numeric_limits<float>::infinity(), "") : std::make_pair(std::numeric_limits<float>::infinity(), "");
    } 
    else if (state == BoardState::BlackWins) 
    {
        return (playerColor == 1) ? std::make_pair(-std::numeric_limits<float>::infinity(), "") : std::make_pair(std::numeric_limits<float>::infinity(), "");
    } 
    else if (state == BoardState::Draw) 
    {
        return {0, ""};
    }
    // ------------------------------------------------------------------------------------------------------
    
    // ------------------------------------ MAX DEPTH REACHED -----------------------------------------------
    if (maxDepth == 0) 
    {
        return {evaluate(board, playerColor), ""};
    }
    // ------------------------------------------------------------------------------------------------------

    float value = -std::numeric_limits<float>::infinity();
    std::string bestMove;
    
    // ------------------------------------ BOARD AND MOVE COLLECTION (SORTED BY EVALUATION) -----------------------------------------------
    auto validMoves = board.GetValidMoves();
    std::vector<std::string> moves;
    std::vector<Board> boards;
    std::vector<float> scores;
    for (const auto& move : *validMoves) {
        std::string moveStr;
        if (board.TryGetMoveString(move, moveStr)) 
        {  
            Board newBoard = board;    

            if(newBoard.TryPlayMove(move, moveStr))
            {
                newBoard.GetGameString();
                float score = evaluate(newBoard, playerColor);

                // Find the right position to insert the score
                auto it = std::lower_bound(scores.begin(), scores.end(), score, std::greater<float>());
                size_t index = std::distance(scores.begin(), it);

                // Insert the score, move, and board at the right position
                scores.insert(it, score);
                moves.insert(moves.begin() + index, moveStr);
                boards.insert(boards.begin() + index, newBoard);
            }
        }
    }
    // std::ofstream log("/home/francesco/Desktop/franz/scuolaOrtogonale/hive/strhive-to-fly/log/logcpp.txt", std::ios::app);
    // log << "Sorted moves: ";
    // for (const auto& score : scores) {
    //     log << score << ", ";
    // }
    // log << "\n";
    // log.close();
    // -------------------------------------------------------------------------------------------------------------------------------------

    // ------------------------------------ NEGAMAX ALGORITHM -----------------------------------------------
    for (size_t i = 0; i < moves.size(); ++i) {
        if (timeUp) break;

        auto [score, _] = negamax(boards[i], playerColor*-1, -beta, -alpha, maxDepth - 1, timeUp);
        score = -score;
        
        if (score > value) {
            value = score;
            bestMove = moves[i];
        }
        
        alpha = std::max(alpha, value);
        if (alpha >= beta) break;
    }
    // ------------------------------------------------------------------------------------------------------

    return {value, bestMove};
}

float MinMax::evaluate(Board& board, int playerColor) {

    int maximizingColor = playerColor;
    int minimizingColor = playerColor*-1;

    float evaluation = 0;
    
    int wQ_moves = 0;
    int bQ_moves = 0;
    int wS_moves = 0;
    int bS_moves = 0;
    int wB_moves = 0;
    int bB_moves = 0;
    int wG_moves = 0;
    int bG_moves = 0;
    int wL_moves = 0;
    int bL_moves = 0;
    int wP_moves = 0;
    int bP_moves = 0;
    int wM_moves = 0;
    int bM_moves = 0;
    int wA_moves = 0;
    int bA_moves = 0;

    // Crea un MoveSet temporaneo per raccogliere le mosse valide
    auto moveSet = board.GetValidMoves();
    for (const auto& move : *moveSet) {
        switch (move.PieceName) {
            case PieceName::wQ:
                wQ_moves++;
                break;
            case PieceName::bQ:
                bQ_moves++;
                break;
            case PieceName::wS1:
            case PieceName::wS2:
                wS_moves++;
                break;
            case PieceName::bS1:
            case PieceName::bS2:
                bS_moves++;
                break;
            case PieceName::wB1:
            case PieceName::wB2:
                wB_moves++;
                break;
            case PieceName::bB1:
            case PieceName::bB2:
                bB_moves++;
                break;
            case PieceName::wG1:
            case PieceName::wG2:
            case PieceName::wG3:
                wG_moves++;
                break;
            case PieceName::bG1:
            case PieceName::bG2:
            case PieceName::bG3:
                bG_moves++;
                break;
            case PieceName::wL:
                wL_moves++;
                break;
            case PieceName::bL:
                bL_moves++;
                break;
            case PieceName::wP:
                wP_moves++;
                break;
            case PieceName::bP:
                bP_moves++;
                break;
            case PieceName::wM:
                wM_moves++;
                break;
            case PieceName::bM:
                bM_moves++;
                break;
            case PieceName::wA1:
            case PieceName::wA2:
            case PieceName::wA3:
                wA_moves++;
                break;
            case PieceName::bA1:
            case PieceName::bA2:
            case PieceName::bA3:
                bA_moves++;
                break;
            default:
                break;
        }
    }

    int wQ_neighbors = board.CountNeighbors(PieceName::wQ);
    int bQ_neighbors = board.CountNeighbors(PieceName::bQ);

    int w_moves = wQ_moves + wS_moves + wB_moves + wA_moves + wG_moves + wM_moves + wL_moves + wP_moves;
    int b_moves = bQ_moves + bS_moves + bB_moves + bA_moves + bG_moves + bM_moves + bL_moves + bP_moves;

    if(maximizingColor==1) //maximizing White
    {
        // Queen neighbors
        evaluation += (bQ_neighbors - wQ_neighbors) * k_qn;

        // Mobility
        evaluation += (wQ_moves - bQ_moves) * k_mq;
        evaluation += (wS_moves - bS_moves) * k_ms;
        evaluation += (wB_moves - bB_moves) * k_mb;
        evaluation += (wA_moves - bA_moves) * k_ma;
        evaluation += (wG_moves - bG_moves) * k_mg;
        evaluation += (wM_moves - bM_moves) * k_mm;
        evaluation += (wL_moves - bL_moves) * k_ml;
        evaluation += (wP_moves - bP_moves) * k_mp;

        // Number of moves
        evaluation += (w_moves - b_moves) * k_nm;
    }
    else // maximizing Black
    {
        // Queen neighbors
        evaluation += (wQ_neighbors - bQ_neighbors) * k_qn;

        // Mobility
        evaluation += (bQ_moves - wQ_moves) * k_mq;
        evaluation += (bS_moves - wS_moves) * k_ms;
        evaluation += (bB_moves - wB_moves) * k_mb;
        evaluation += (bA_moves - wA_moves) * k_ma;
        evaluation += (bG_moves - wG_moves) * k_mg;
        evaluation += (bM_moves - wM_moves) * k_mm;
        evaluation += (bL_moves - wL_moves) * k_ml;
        evaluation += (bP_moves - wP_moves) * k_mp;

        // Number of moves
        evaluation += (b_moves - w_moves) * k_nm;

    }

    // std::ofstream log("/home/francesco/Desktop/franz/scuolaOrtogonale/hive/strhive-to-fly/log/logcpp.txt", std::ios::app);
    // log << "Evaluation -> " << evaluation << "\n";
    // log.close();

    return evaluation;
}
