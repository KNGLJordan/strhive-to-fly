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

// Costruttore con solo flag booleano (usa tutti i default)
MinMaxZobrist::MinMaxZobrist(bool useEnhancedEval) 
    : weights(), useEnhanced(useEnhancedEval)
{
}

// Costruttore con struct EvaluationWeights
MinMaxZobrist::MinMaxZobrist(const EvaluationWeights& w, bool useEnhancedEval) 
    : weights(w), useEnhanced(useEnhancedEval)
{
}

// Costruttore legacy con vector (mantieni per compatibilità)
MinMaxZobrist::MinMaxZobrist(std::vector<float> w, bool useEnhancedEval) 
    : weights(), useEnhanced(useEnhancedEval)
{
    if (!w.empty()) {
        if (w.size() >= 1) weights.k_qn = w[0];
        if (w.size() >= 2) weights.k_mq = w[1];
        if (w.size() >= 3) weights.k_ms = w[2];
        if (w.size() >= 4) weights.k_mb = w[3];
        if (w.size() >= 5) weights.k_ma = w[4];
        if (w.size() >= 6) weights.k_mg = w[5];
        if (w.size() >= 7) weights.k_mm = w[6];
        if (w.size() >= 8) weights.k_ml = w[7];
        if (w.size() >= 9) weights.k_mp = w[8];
        if (w.size() >= 10) weights.k_nm = w[9];
        
        if (useEnhanced) {
            if (w.size() >= 11) weights.k_pieces_in_play = w[10];
            if (w.size() >= 12) weights.k_pieces_in_hand = w[11];
            if (w.size() >= 13) weights.k_pinned_pieces = w[12];
            if (w.size() >= 14) weights.k_covered_pieces = w[13];
            if (w.size() >= 15) weights.k_noisy_moves = w[14];
            if (w.size() >= 16) weights.k_quiet_moves = w[15];
            if (w.size() >= 17) weights.k_friendly_neighbors = w[16];
            if (w.size() >= 18) weights.k_enemy_neighbors = w[17];
            if (w.size() >= 19) weights.k_queen_safety = w[18];
            if (w.size() >= 20) weights.k_ant_mobility = w[19];
            if (w.size() >= 21) weights.k_beetle_coverage = w[20];
            if (w.size() >= 22) weights.k_spider_positioning = w[21];
        }
    }
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
        float eval = useEnhanced ? evaluate2(board, playerColor) : evaluate(board, playerColor);
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
            float staticScore = useEnhanced ? evaluate2(board, playerColor) : evaluate(board, playerColor);
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
        float eval = useEnhanced ? evaluate2(board, playerColor) : evaluate(board, playerColor);
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
            float staticScore = useEnhanced ? evaluate2(board, playerColor) : evaluate(board, playerColor);
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
        evaluation += (bQ_neighbors - wQ_neighbors) * weights.k_qn;

        // Mobility
        evaluation += (wQ_moves - bQ_moves) * weights.k_mq;
        evaluation += (wS_moves - bS_moves) * weights.k_ms;
        evaluation += (wB_moves - bB_moves) * weights.k_mb;
        evaluation += (wA_moves - bA_moves) * weights.k_ma;
        evaluation += (wG_moves - bG_moves) * weights.k_mg;
        evaluation += (wM_moves - bM_moves) * weights.k_mm;
        evaluation += (wL_moves - bL_moves) * weights.k_ml;
        evaluation += (wP_moves - bP_moves) * weights.k_mp;

        // Number of moves
        evaluation += (w_moves - b_moves) * weights.k_nm;
    }
    else // maximizing Black
    {
        // Queen neighbors
        evaluation += (wQ_neighbors - bQ_neighbors) * weights.k_qn;

        // Mobility
        evaluation += (bQ_moves - wQ_moves) * weights.k_mq;
        evaluation += (bS_moves - wS_moves) * weights.k_ms;
        evaluation += (bB_moves - wB_moves) * weights.k_mb;
        evaluation += (bA_moves - wA_moves) * weights.k_ma;
        evaluation += (bG_moves - wG_moves) * weights.k_mg;
        evaluation += (bM_moves - wM_moves) * weights.k_mm;
        evaluation += (bL_moves - wL_moves) * weights.k_ml;
        evaluation += (bP_moves - wP_moves) * weights.k_mp;

        // Number of moves
        evaluation += (b_moves - w_moves) * weights.k_nm;

    }

    // std::ofstream log("/home/francesco/Desktop/franz/scuolaOrtogonale/hive/strhive-to-fly/log/logcpp.txt", std::ios::app);
    // log << "Evaluation -> " << evaluation << "\n";
    // log.close();

    return evaluation;
}

float MinMaxZobrist::evaluate2(Board& board, int playerColor) {
    // Get comprehensive board metrics
    BoardMetrics metrics = board.GetBoardMetrics();
    
    float evaluation = 0.0f;
    
    // Determine which color we're maximizing vs minimizing
    Color maxColor = (playerColor == 1) ? Color::White : Color::Black;
    Color minColor = (playerColor == 1) ? Color::Black : Color::White;
    
    // Basic piece count metrics
    int maxPiecesInPlay = 0, minPiecesInPlay = 0;
    int maxPiecesInHand = 0, minPiecesInHand = 0;
    int maxPinnedPieces = 0, minPinnedPieces = 0;
    int maxCoveredPieces = 0, minCoveredPieces = 0;
    int maxNoisyMoves = 0, minNoisyMoves = 0;
    int maxQuietMoves = 0, minQuietMoves = 0;
    int maxFriendlyNeighbors = 0, minFriendlyNeighbors = 0;
    int maxEnemyNeighbors = 0, minEnemyNeighbors = 0;
    
    // Queen-specific metrics
    PieceName maxQueen = (maxColor == Color::White) ? PieceName::wQ : PieceName::bQ;
    PieceName minQueen = (maxColor == Color::White) ? PieceName::bQ : PieceName::wQ;
    
    int maxQueenNeighbors = metrics[maxQueen].FriendlyNeighborCount + metrics[maxQueen].EnemyNeighborCount;
    int minQueenNeighbors = metrics[minQueen].FriendlyNeighborCount + metrics[minQueen].EnemyNeighborCount;
    
    // Piece-type specific metrics
    struct PieceTypeMetrics {
        int mobility = 0;
        int noisyMoves = 0;
        int quietMoves = 0;
        int pinned = 0;
        int covered = 0;
        int friendlyNeighbors = 0;
        int enemyNeighbors = 0;
    };
    
    PieceTypeMetrics maxPieces[(int)BugType::NumBugTypes] = {};
    PieceTypeMetrics minPieces[(int)BugType::NumBugTypes] = {};
    
    // Collect metrics for all pieces
    for (int pn = 0; pn < (int)PieceName::NumPieceNames; pn++) {
        PieceName piece = (PieceName)pn;
        Color pieceColor = GetColor(piece);
        BugType bugType = GetBugType(piece);
        
        if (!PieceNameIsEnabledForGameType(piece, board.GetGameType())) {
            continue;
        }
        
        const PieceMetrics& pm = metrics[piece];
        
        if (pieceColor == maxColor) {
            maxPiecesInPlay += pm.InPlay;
            maxPiecesInHand += (1 - pm.InPlay);
            maxPinnedPieces += pm.IsPinned;
            maxCoveredPieces += pm.IsCovered;
            maxNoisyMoves += pm.NoisyMoveCount;
            maxQuietMoves += pm.QuietMoveCount;
            maxFriendlyNeighbors += pm.FriendlyNeighborCount;
            maxEnemyNeighbors += pm.EnemyNeighborCount;
            
            maxPieces[(int)bugType].mobility += pm.NoisyMoveCount + pm.QuietMoveCount;
            maxPieces[(int)bugType].noisyMoves += pm.NoisyMoveCount;
            maxPieces[(int)bugType].quietMoves += pm.QuietMoveCount;
            maxPieces[(int)bugType].pinned += pm.IsPinned;
            maxPieces[(int)bugType].covered += pm.IsCovered;
            maxPieces[(int)bugType].friendlyNeighbors += pm.FriendlyNeighborCount;
            maxPieces[(int)bugType].enemyNeighbors += pm.EnemyNeighborCount;
        } else {
            minPiecesInPlay += pm.InPlay;
            minPiecesInHand += (1 - pm.InPlay);
            minPinnedPieces += pm.IsPinned;
            minCoveredPieces += pm.IsCovered;
            minNoisyMoves += pm.NoisyMoveCount;
            minQuietMoves += pm.QuietMoveCount;
            minFriendlyNeighbors += pm.FriendlyNeighborCount;
            minEnemyNeighbors += pm.EnemyNeighborCount;
            
            minPieces[(int)bugType].mobility += pm.NoisyMoveCount + pm.QuietMoveCount;
            minPieces[(int)bugType].noisyMoves += pm.NoisyMoveCount;
            minPieces[(int)bugType].quietMoves += pm.QuietMoveCount;
            minPieces[(int)bugType].pinned += pm.IsPinned;
            minPieces[(int)bugType].covered += pm.IsCovered;
            minPieces[(int)bugType].friendlyNeighbors += pm.FriendlyNeighborCount;
            minPieces[(int)bugType].enemyNeighbors += pm.EnemyNeighborCount;
        }
    }
    
    // // 1. Queen Safety (existing logic but enhanced)
    // evaluation += (minQueenNeighbors - maxQueenNeighbors) * weights.k_qn;
    
    // // Extra penalty if our queen is highly threatened
    // if (maxQueenNeighbors >= 4) {
    //     evaluation += weights.k_queen_safety * (maxQueenNeighbors - 3);
    // }
    
    // // 2. Basic Mobility (similar to original but using metrics data)
    // evaluation += (maxPieces[(int)BugType::QueenBee].mobility - minPieces[(int)BugType::QueenBee].mobility) * weights.k_mq;
    // evaluation += (maxPieces[(int)BugType::Spider].mobility - minPieces[(int)BugType::Spider].mobility) * weights.k_ms;
    // evaluation += (maxPieces[(int)BugType::Beetle].mobility - minPieces[(int)BugType::Beetle].mobility) * weights.k_mb;
    // evaluation += (maxPieces[(int)BugType::SoldierAnt].mobility - minPieces[(int)BugType::SoldierAnt].mobility) * weights.k_ma;
    // evaluation += (maxPieces[(int)BugType::Grasshopper].mobility - minPieces[(int)BugType::Grasshopper].mobility) * weights.k_mg;
    // evaluation += (maxPieces[(int)BugType::Mosquito].mobility - minPieces[(int)BugType::Mosquito].mobility) * weights.k_mm;
    // evaluation += (maxPieces[(int)BugType::Ladybug].mobility - minPieces[(int)BugType::Ladybug].mobility) * weights.k_ml;
    // evaluation += (maxPieces[(int)BugType::Pillbug].mobility - minPieces[(int)BugType::Pillbug].mobility) * weights.k_mp;
    
    // 3. Enhanced Strategic Metrics
    evaluation += (maxPiecesInPlay - minPiecesInPlay) * weights.k_pieces_in_play;
    evaluation += (maxPiecesInHand - minPiecesInHand) * weights.k_pieces_in_hand;
    evaluation += (maxPinnedPieces - minPinnedPieces) * weights.k_pinned_pieces;
    evaluation += (maxCoveredPieces - minCoveredPieces) * weights.k_covered_pieces;
    evaluation += (maxNoisyMoves - minNoisyMoves) * weights.k_noisy_moves;
    evaluation += (maxQuietMoves - minQuietMoves) * weights.k_quiet_moves;
    evaluation += (maxFriendlyNeighbors - minFriendlyNeighbors) * weights.k_friendly_neighbors;
    evaluation += (maxEnemyNeighbors - minEnemyNeighbors) * weights.k_enemy_neighbors;
    
    // 4. Piece-Specific Strategic Bonuses
    
    // Ant mobility bonus (ants are extremely valuable when mobile)
    evaluation += (maxPieces[(int)BugType::SoldierAnt].mobility - minPieces[(int)BugType::SoldierAnt].mobility) * weights.k_ant_mobility;
    
    // Beetle coverage bonus (beetles on top of enemy pieces)
    evaluation += (maxPieces[(int)BugType::Beetle].enemyNeighbors - minPieces[(int)BugType::Beetle].enemyNeighbors) * weights.k_beetle_coverage;
    
    // Spider positioning (spiders with good neighbor count)
    evaluation += (maxPieces[(int)BugType::Spider].friendlyNeighbors - minPieces[(int)BugType::Spider].friendlyNeighbors) * weights.k_spider_positioning;
    
    // Total mobility (similar to original weights.k_nm)
    int maxTotalMoves = maxNoisyMoves + maxQuietMoves;
    int minTotalMoves = minNoisyMoves + minQuietMoves;
    evaluation += (maxTotalMoves - minTotalMoves) * weights.k_nm;
    
    return evaluation;
}
