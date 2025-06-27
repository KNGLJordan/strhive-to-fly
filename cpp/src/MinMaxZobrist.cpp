#include <vector>
#include <string>
#include <limits>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <atomic>
#include <thread>

#include "Board.h"  
#include "Enums.h"  
#include "MinMaxZobrist.h"

using namespace MzingaCpp;

MinMaxZobrist::MinMaxZobrist(std::vector<float> weights) 
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

std::string MinMaxZobrist::calculateBestMove(Board& board, int maxDepth, int timeLimit) 
{    
    float bestScore;
    std::string bestMove;

    int currentTurn; 

    std::atomic<bool> timeUp(false);

    std::atomic<int> nodeCounter(0);

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
        // std::tie(bestScore, bestMove) = negamaxStats(board, currentTurn, -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), maxDepth, timeUp, nodeCounter);
        
        // std::ofstream log("/home/francesco/Desktop/franz/scuolaOrtogonale/hive/strhive-to-fly/log/logcpp.txt", std::ios::app);
        // log << "chosen depth " << maxDepth << ", evaluated nodes " << nodeCounter.load() << "\n";
        // log.close();
    }
    else
    {
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
            //std::tie(score, move) = negamaxStats(board, currentTurn, -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), depth, timeUp, nodeCounter);
            
            if (!move.empty()) {
                bestScore = score;
                bestMove = move;
            }

            depth++;
        }
        
        timerThread.join();

        // std::ofstream log("/home/francesco/Desktop/franz/scuolaOrtogonale/hive/strhive-to-fly/log/logcpp.txt", std::ios::app);
        // log << "reached"<< " " << depth<< "\n";
        // log << "evaluated nodes " << nodeCounter.load() << "\n";
        // log.close();
    }
    
    return bestMove;
}

std::pair<float, std::string> MinMaxZobrist::negamax(
    Board board, 
    int playerColor, 
    float alpha, 
    float beta, 
    int maxDepth, 
    std::atomic<bool>& timeUp) 
{
    // 1) Se il timer è scaduto, interrompo subito
    if (timeUp) {
        return {0, ""};
    }


    // 2) Controllo se presente nella TT con profondità sufficiente
    uint64_t zobristKey = zobristHasher.computeHash(board);

    
    if (transpositionTable.find(zobristKey) != transpositionTable.end()) 
    {
        const TTEntry& entry = transpositionTable[zobristKey];

        if (entry.depth >= maxDepth) 
        {
            if (entry.flag == Flag::EXACT) 
            {
                return {entry.value, entry.bestMove};
            } 
            else if (entry.flag == Flag::LOWER_BOUND && entry.value >= beta) 
            {
                return {entry.value, entry.bestMove};
            } 
            else if (entry.flag == Flag::UPPER_BOUND && entry.value <= alpha) 
            {
                return {entry.value, entry.bestMove}; 
            }
        }
    }

    // 3) Controllo se sono in nodo terminale (vittoria/pareggio)
   BoardState state = board.GetBoardState();
    if (state == BoardState::WhiteWins) {
        float inf = std::numeric_limits<float>::infinity();
        return (playerColor == 0)
            ? std::make_pair(-inf, "")
            : std::make_pair(+inf, "");
    }
    else if (state == BoardState::BlackWins) {
        float inf = std::numeric_limits<float>::infinity();
        return (playerColor == 1)
            ? std::make_pair(-inf, "")
            : std::make_pair(+inf, "");
    }
    else if (state == BoardState::Draw) {
        return {0.0f, ""};
    }

    // 4) Se ho raggiunto la profondità massima, valuto staticamente
    if (maxDepth == 0) 
    {
        float eval = evaluate(board, playerColor);
        transpositionTable[zobristKey] = {eval, "", maxDepth, Flag::EXACT};
        return {eval, ""};
    }

    // 5) Genero la lista delle mosse valide e ne faccio una valutazione "statica" per l'ordinamento.
    
    //    Memorizzo in una sola struct: (Move, moveString, staticScore)
    struct MoveEntry {
        Move        m;
        std::string str;
        float       score;
    };

    std::vector<MoveEntry> movelist;
    
    // Suppongo che GetValidMoves() torni un puntatore/opzionale a std::vector<Move>
    auto validMovesPtr = board.GetValidMoves();
    if (!validMovesPtr || validMovesPtr->empty()) {
        // Se non ci sono mosse valide, è come se fossimo in un nodo terminale di stallo/draw
        return {0.0f, ""};
    }

    movelist.reserve(validMovesPtr->size());
    for (const auto& move : *validMovesPtr) {
        std::string moveStr;
        if (!board.TryGetMoveString(move, moveStr)) {
            continue;
        }

        // Provo ad applicare la mossa
        if (board.TryPlayMove(move, moveStr)) {
            // Calcolo la valutazione statica subito dopo aver giocato
            float staticScore = evaluate(board, playerColor);
            // Undo per tornare allo stato precedente
            board.TryUndoLastMove();

            // Memorizzo Move, la sua stringa e il punteggio per l'ordinamento
            movelist.push_back({ move, moveStr, staticScore });
        }
    }

    // Se dopo il filtro non ci sono mosse (es. nessuna era legale), trattalo come draw:
    if (movelist.empty()) {
        return {0.0f, ""};
    }

    // 6) Ordino movelist dal punteggio statico più alto al più basso
    std::sort(
        movelist.begin(), movelist.end(),
        [](MoveEntry const& a, MoveEntry const& b) {
            return a.score > b.score;
        }
    );

    
    // -----------------------VECCHIO ORDINAMENTO CON TT-----------------------
    // ----- NOTA: questo codice metteva come prima mossa quella della TT -----
    
    // auto validMoves = board.GetValidMoves();
    // std::vector<std::string> moves;
    // std::vector<Board> boards;
    // std::vector<float> scores;

    // // Recupero mossa dalla TT per metterla prima
    // std::string ttBestMove = "";
    // if (transpositionTable.find(zobristKey) != transpositionTable.end()) {
    //     ttBestMove = transpositionTable[zobristKey].bestMove;
    // }

    // // Aggiungi le mosse alla lista (escludendo la TT per ora)
    // for (const auto& move : *validMoves) 
    // {
    //     std::string moveStr;
    //     if (board.TryGetMoveString(move, moveStr)) 
    //     {
    //         Board newBoard = board;

    //         if (newBoard.TryPlayMove(move, moveStr)) 
    //         {
    //             newBoard.GetGameString();
    //             float score = evaluate(newBoard, playerColor);
    //             scores.push_back(score);
    //             moves.push_back(moveStr);
    //             boards.push_back(newBoard);
    //         }
    //     }
    // }

    // // Se esiste una mossa della TT, la metto in cima
    // if (!ttBestMove.empty()) 
    // {
    //     auto it = std::find(moves.begin(), moves.end(), ttBestMove);
    //     if (it != moves.end()) 
    //     {
    //         size_t index = std::distance(moves.begin(), it);
    //         std::swap(moves[0], moves[index]);
    //         std::swap(scores[0], scores[index]);
    //         std::swap(boards[0], boards[index]);
    //     }
    // }

    // -----------------------FINE VECCHIO ORDINAMENTO CON TT-----------------------


    // 7) Salvo alpha che potrebbe essere aggiornato
    float originalAlpha = alpha;   

    // 8) Negamax ricorsivo con alpha-beta

    float value = -std::numeric_limits<float>::infinity();
    std::string bestMove = "";

     for (auto& entry : movelist) 
    {
        if (timeUp){
            break;   
        }

        // Applico direttamente la mossa "entry.m" usando la stringa pre-memorizzata "entry.str"
        if (!board.TryPlayMove(entry.m, entry.str)) {
            // Se per qualche motivo TryPlayMove fallisce, skippo
            continue;
        }

        // Chiamata ricorsiva: cambio segno, scambio alpha/beta, profondità-1
        auto [childScore, _childMove] =
            negamax(board, -playerColor, -beta, -alpha, maxDepth - 1, timeUp);
        float score = -childScore;

        // Torno indietro al nodo padre
        board.TryUndoLastMove();
        
        // Se lo score è migliore del valore corrente, aggiorno
        // e salvo la mossa migliore
        if (score > value) 
        {
            value = score;
            bestMove = entry.str;
        }
        
        // Aggiorno alpha
        alpha = std::max(alpha, value);

        // Se alpha è maggiore o uguale a beta, faccio pruning
        if (alpha >= beta) { 
            break;
        } 
    }

    Flag flag;
    if (value <= originalAlpha) 
    {
        flag = Flag::UPPER_BOUND;  // valore minore dell'alpha originale
    } 
    else if (value >= beta) 
    {
        flag = Flag::LOWER_BOUND;  // valore maggiore di beta
    } 
    else 
    {
        flag = Flag::EXACT;        // valore esatto
    }

    // Salvo nella TT solo se la profondità è maggiore
    if (transpositionTable.find(zobristKey) == transpositionTable.end() || transpositionTable[zobristKey].depth < maxDepth) 
    {
        transpositionTable[zobristKey] = {value, bestMove, maxDepth, flag};
    }

    return {value, bestMove};
}

std::pair<float, std::string> MinMaxZobrist::negamaxStats(
    Board board, 
    int playerColor, 
    float alpha, 
    float beta, 
    int maxDepth, 
    std::atomic<bool>& timeUp,
    std::atomic<int>& nodeCounter) 
{
    // 0) Incrementa i nodi visitati
    nodeCounter++;

    // 1) Se il timer è scaduto, interrompo subito
    if (timeUp) {
        return {0, ""};
    }


    // 2) Controllo se presente nella TT con profondità sufficiente
    uint64_t zobristKey = zobristHasher.computeHash(board);

    
    if (transpositionTable.find(zobristKey) != transpositionTable.end()) 
    {
        const TTEntry& entry = transpositionTable[zobristKey];

        if (entry.depth >= maxDepth) 
        {
            if (entry.flag == Flag::EXACT) 
            {
                return {entry.value, entry.bestMove};
            } 
            else if (entry.flag == Flag::LOWER_BOUND && entry.value >= beta) 
            {
                return {entry.value, entry.bestMove};
            } 
            else if (entry.flag == Flag::UPPER_BOUND && entry.value <= alpha) 
            {
                return {entry.value, entry.bestMove}; 
            }
        }
    }

    // 3) Controllo se sono in nodo terminale (vittoria/pareggio)
   BoardState state = board.GetBoardState();
    if (state == BoardState::WhiteWins) {
        float inf = std::numeric_limits<float>::infinity();
        return (playerColor == 0)
            ? std::make_pair(-inf, "")
            : std::make_pair(+inf, "");
    }
    else if (state == BoardState::BlackWins) {
        float inf = std::numeric_limits<float>::infinity();
        return (playerColor == 1)
            ? std::make_pair(-inf, "")
            : std::make_pair(+inf, "");
    }
    else if (state == BoardState::Draw) {
        return {0.0f, ""};
    }

    // 4) Se ho raggiunto la profondità massima, valuto staticamente
    if (maxDepth == 0) 
    {
        float eval = evaluate(board, playerColor);
        transpositionTable[zobristKey] = {eval, "", maxDepth, Flag::EXACT};
        return {eval, ""};
    }

    // 5) Genero la lista delle mosse valide e ne faccio una valutazione "statica" per l'ordinamento.
    
    //    Memorizzo in una sola struct: (Move, moveString, staticScore)
    struct MoveEntry {
        Move        m;
        std::string str;
        float       score;
    };

    std::vector<MoveEntry> movelist;
    
    // Suppongo che GetValidMoves() torni un puntatore/opzionale a std::vector<Move>
    auto validMovesPtr = board.GetValidMoves();
    if (!validMovesPtr || validMovesPtr->empty()) {
        // Se non ci sono mosse valide, è come se fossimo in un nodo terminale di stallo/draw
        return {0.0f, ""};
    }

    movelist.reserve(validMovesPtr->size());
    for (const auto& move : *validMovesPtr) {
        std::string moveStr;
        if (!board.TryGetMoveString(move, moveStr)) {
            continue;
        }

        // Provo ad applicare la mossa
        if (board.TryPlayMove(move, moveStr)) {
            // Calcolo la valutazione statica subito dopo aver giocato
            float staticScore = evaluate(board, playerColor);
            // Undo per tornare allo stato precedente
            board.TryUndoLastMove();

            // Memorizzo Move, la sua stringa e il punteggio per l'ordinamento
            movelist.push_back({ move, moveStr, staticScore });
        }
    }

    // Se dopo il filtro non ci sono mosse (es. nessuna era legale), trattalo come draw:
    if (movelist.empty()) {
        return {0.0f, ""};
    }

    // 6) Ordino movelist dal punteggio statico più alto al più basso
    std::sort(
        movelist.begin(), movelist.end(),
        [](MoveEntry const& a, MoveEntry const& b) {
            return a.score > b.score;
        }
    );

    
    // -----------------------VECCHIO ORDINAMENTO CON TT-----------------------
    // ----- NOTA: questo codice metteva come prima mossa quella della TT -----
    
    // auto validMoves = board.GetValidMoves();
    // std::vector<std::string> moves;
    // std::vector<Board> boards;
    // std::vector<float> scores;

    // // Recupero mossa dalla TT per metterla prima
    // std::string ttBestMove = "";
    // if (transpositionTable.find(zobristKey) != transpositionTable.end()) {
    //     ttBestMove = transpositionTable[zobristKey].bestMove;
    // }

    // // Aggiungi le mosse alla lista (escludendo la TT per ora)
    // for (const auto& move : *validMoves) 
    // {
    //     std::string moveStr;
    //     if (board.TryGetMoveString(move, moveStr)) 
    //     {
    //         Board newBoard = board;

    //         if (newBoard.TryPlayMove(move, moveStr)) 
    //         {
    //             newBoard.GetGameString();
    //             float score = evaluate(newBoard, playerColor);
    //             scores.push_back(score);
    //             moves.push_back(moveStr);
    //             boards.push_back(newBoard);
    //         }
    //     }
    // }

    // // Se esiste una mossa della TT, la metto in cima
    // if (!ttBestMove.empty()) 
    // {
    //     auto it = std::find(moves.begin(), moves.end(), ttBestMove);
    //     if (it != moves.end()) 
    //     {
    //         size_t index = std::distance(moves.begin(), it);
    //         std::swap(moves[0], moves[index]);
    //         std::swap(scores[0], scores[index]);
    //         std::swap(boards[0], boards[index]);
    //     }
    // }

    // -----------------------FINE VECCHIO ORDINAMENTO CON TT-----------------------


    // 7) Salvo alpha che potrebbe essere aggiornato
    float originalAlpha = alpha;   

    // 8) Negamax ricorsivo con alpha-beta

    float value = -std::numeric_limits<float>::infinity();
    std::string bestMove = "";

     for (auto& entry : movelist) 
    {
        if (timeUp){
            break;   
        }

        // Applico direttamente la mossa "entry.m" usando la stringa pre-memorizzata "entry.str"
        if (!board.TryPlayMove(entry.m, entry.str)) {
            // Se per qualche motivo TryPlayMove fallisce, skippo
            continue;
        }

        // Chiamata ricorsiva: cambio segno, scambio alpha/beta, profondità-1
        auto [childScore, _childMove] =
            negamaxStats(board, -playerColor, -beta, -alpha, maxDepth - 1, timeUp, nodeCounter);
        float score = -childScore;

        // Torno indietro al nodo padre
        board.TryUndoLastMove();
        
        // Se lo score è migliore del valore corrente, aggiorno
        // e salvo la mossa migliore
        if (score > value) 
        {
            value = score;
            bestMove = entry.str;
        }
        
        // Aggiorno alpha
        alpha = std::max(alpha, value);

        // Se alpha è maggiore o uguale a beta, faccio pruning
        if (alpha >= beta) { 
            break;
        } 
    }

    Flag flag;
    if (value <= originalAlpha) 
    {
        flag = Flag::UPPER_BOUND;  // valore minore dell'alpha originale
    } 
    else if (value >= beta) 
    {
        flag = Flag::LOWER_BOUND;  // valore maggiore di beta
    } 
    else 
    {
        flag = Flag::EXACT;        // valore esatto
    }

    // Salvo nella TT solo se la profondità è maggiore
    if (transpositionTable.find(zobristKey) == transpositionTable.end() || transpositionTable[zobristKey].depth < maxDepth) 
    {
        transpositionTable[zobristKey] = {value, bestMove, maxDepth, flag};
    }

    return {value, bestMove};
}

float MinMaxZobrist::evaluate(Board& board, int playerColor) {

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
