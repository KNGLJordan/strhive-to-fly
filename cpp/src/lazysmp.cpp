#include <cstring>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <thread>
#include <random>
#include <limits>
#include "lazysmp.h"

namespace MzingaCpp
{

// ThreadData implementation
MinMaxLocklessLazySMP::ThreadData::ThreadData(int id)
    : thread_id(id),
      rng(id * 12345),  // Different seed per thread
      best_score(-std::numeric_limits<float>::infinity()),
      best_move_hash(0),
      completed_depth(0),
      nodes_searched(0),
      tb_hits(0),
      beta_cutoffs(0) {
    
    std::memset(best_move_str, 0, sizeof(best_move_str));
    
    // Create diversity in search parameters
    // This is the KEY to Lazy SMP's effectiveness!
    
    if (id == 0) {
        // Main thread: searches exact depth, standard parameters
        depth_offset = 0;
        use_aspiration = true;
        aspiration_delta = 50.0f;
        id_step = 1;  // Increment by 1 each iteration
    } 
    else if (id < 4) {
        // Threads 1-3: Search 1 ply deeper with varying aspiration
        depth_offset = 1;
        use_aspiration = true;
        aspiration_delta = 50.0f + id * 25.0f;
        id_step = 1;
    }
    else if (id < 8) {
        // Threads 4-7: Search exact depth with wider windows
        depth_offset = 0;
        use_aspiration = true;
        aspiration_delta = 100.0f + (id - 4) * 50.0f;
        id_step = 1;
    }
    else if (id < 12) {
        // Threads 8-11: Search 1 ply shallower, no aspiration
        depth_offset = -1;
        use_aspiration = false;
        aspiration_delta = 0;
        id_step = 1;
    }
    else if (id < 16) {
        // Threads 12-15: Search 2 plies deeper with standard windows
        depth_offset = 2;
        use_aspiration = true;
        aspiration_delta = 75.0f;
        id_step = 1;
    }
    else if (id < 20) {
        // Threads 16-19: Skip even depths (1, 3, 5, 7...)
        depth_offset = 0;
        use_aspiration = false;
        aspiration_delta = 0;
        id_step = 2;  // Skip every other depth
    }
    else {
        // Remaining threads: Mixed strategies
        depth_offset = (id % 5) - 2;  // -2 to +2
        use_aspiration = (id % 3 != 0);
        aspiration_delta = 50.0f + (id % 4) * 50.0f;
        id_step = (id % 8 == 0) ? 2 : 1;
    }
}

void MinMaxLocklessLazySMP::ThreadData::resetSearch() {
    best_score.store(-std::numeric_limits<float>::infinity(), std::memory_order_relaxed);
    best_move_hash.store(0, std::memory_order_relaxed);
    completed_depth.store(0, std::memory_order_relaxed);
    nodes_searched.store(0, std::memory_order_relaxed);
    tb_hits.store(0, std::memory_order_relaxed);
    beta_cutoffs.store(0, std::memory_order_relaxed);
    std::memset(best_move_str, 0, sizeof(best_move_str));
}

void MinMaxLocklessLazySMP::ThreadData::updateBestMove(float score, const std::string& move, int depth) {
    // Only update if this is a better score at a deeper or equal depth
    float current = best_score.load(std::memory_order_relaxed);
    int current_depth = completed_depth.load(std::memory_order_relaxed);
    
    if (depth > current_depth || (depth == current_depth && score > current)) {
        best_score.store(score, std::memory_order_relaxed);
        completed_depth.store(depth, std::memory_order_relaxed);
        
        uint64_t moveHash = std::hash<std::string>{}(move);
        best_move_hash.store(moveHash, std::memory_order_relaxed);
        
        size_t len = std::min(move.length(), size_t(31));
        std::memcpy(best_move_str, move.c_str(), len);
        best_move_str[len] = '\0';
    }
}

std::string MinMaxLocklessLazySMP::ThreadData::getBestMove() const {
    return std::string(best_move_str);
}

// Main class implementation
MinMaxLocklessLazySMP::MinMaxLocklessLazySMP(bool useEnhancedEval, int threads)
    : MinMaxZobrist(useEnhancedEval),
      num_threads(std::min(threads, MAX_THREADS)) {
    
    thread_data.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        thread_data.push_back(std::make_unique<ThreadData>(i));
    }
    startWorkerThreads();
}

MinMaxLocklessLazySMP::MinMaxLocklessLazySMP(const EvaluationWeights& w, bool useEnhancedEval, int threads)
    : MinMaxZobrist(w, useEnhancedEval),
      num_threads(std::min(threads, MAX_THREADS)) {
    
    thread_data.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        thread_data.push_back(std::make_unique<ThreadData>(i));
    }
    startWorkerThreads();
}

MinMaxLocklessLazySMP::~MinMaxLocklessLazySMP() {
    stopWorkerThreads();
}

void MinMaxLocklessLazySMP::initializeBoards(GameType gameType) {
    for (auto& td : thread_data) {
        td->board = Board(gameType);
    }
}

void MinMaxLocklessLazySMP::syncBoards(const Board& board) {
    for (auto& td : thread_data) {
        td->board = board;
    }
}

void MinMaxLocklessLazySMP::applyMoveToAllBoards(const Move& move, const std::string& moveStr) {
    for (auto& td : thread_data) {
        td->board.TryPlayMove(move, moveStr);
    }
}

void MinMaxLocklessLazySMP::applyUndoToAllBoards(int numMoves) {
    for (auto& td : thread_data) {
        for (int i = 0; i < numMoves; ++i) {
            td->board.TryUndoLastMove();
        }
    }
}

std::string MinMaxLocklessLazySMP::calculateBestMove(Board& board, int maxDepth, int timeLimit) {
    // Sync boards with current position
    // syncBoards(board);
    return searchLockless(maxDepth, timeLimit);
}

std::string MinMaxLocklessLazySMP::searchLockless(int maxDepth, int timeLimit) {
    // Clear transposition table and reset thread data
    // tt.clear();
    for (auto& td : thread_data) {
        td->resetSearch();
    }
    // Set up search parameters
    current_player.store((thread_data[0]->board.GetCurrentTurn() % 2 == 1) ? -1 : 1, 
                        std::memory_order_relaxed);
    threads_finished.store(0, std::memory_order_relaxed);
    time_limit_reached.store(false, std::memory_order_relaxed);
    
    // Determine search mode
    if (timeLimit > 0) {
        // TIME-LIMITED SEARCH: Use iterative deepening
        search_mode.store(SearchMode::TIME_LIMITED, std::memory_order_relaxed);
        search_start_time.store(getTimeMs(), std::memory_order_relaxed);
        time_limit_ms.store(timeLimit * 1000, std::memory_order_relaxed);
        target_depth.store(MAX_SEARCH_DEPTH, std::memory_order_relaxed);  // No depth limit
    } else {
        // DEPTH-LIMITED SEARCH: Search to exact depth
        search_mode.store(SearchMode::DEPTH_LIMITED, std::memory_order_relaxed);
        target_depth.store(maxDepth, std::memory_order_relaxed);
        time_limit_ms.store(0, std::memory_order_relaxed);
    }
    
    // Start search
    search_active.store(true, std::memory_order_release);
    
    // Wait for all threads to finish
    while (threads_finished.load(std::memory_order_acquire) < num_threads) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));  // Better than yield()
    }
    
    search_active.store(false, std::memory_order_relaxed);
    search_mode.store(SearchMode::NONE, std::memory_order_relaxed);
    
    // Collect best result from all threads
    float best_score = -std::numeric_limits<float>::infinity();
    std::string best_move;
    int best_thread = -1;
    int best_depth = 0;
    
    for (int t = 0; t < num_threads; ++t) {
        float score = thread_data[t]->best_score.load(std::memory_order_relaxed);
        int depth = thread_data[t]->completed_depth.load(std::memory_order_relaxed);
        
        // Prefer deeper searches with good scores
        if (depth > best_depth || (depth == best_depth && score > best_score)) {
            best_score = score;
            best_move = thread_data[t]->getBestMove();
            best_thread = t;
            best_depth = depth;
        }
    }
    
    // #ifdef DEBUG_LAZY_SMP
    // std::cout << "\n=== Lazy SMP Search Complete ===\n";
    // std::cout << "Mode: " << (timeLimit > 0 ? "Time-limited" : "Depth-limited") << "\n";
    // std::cout << "Best thread: " << best_thread << " (depth " << best_depth << ", score " << best_score << ")\n";
    // std::cout << "TT usage: ~" << tt.approximateUsage() << " entries\n\n";
    
    // // Show top threads
    // std::cout << "Thread statistics:\n";
    // for (int t = 0; t < num_threads; ++t) {
    //     std::cout << "  Thread " << t 
    //               << ": depth=" << thread_data[t]->completed_depth.load()
    //               << ", nodes=" << thread_data[t]->nodes_searched.load()
    //               << ", tt_hits=" << thread_data[t]->tb_hits.load()
    //               << ", cutoffs=" << thread_data[t]->beta_cutoffs.load() << "\n";
    // }
    // std::cout << "\n";
    // #endif
    
    return best_move;
}

void MinMaxLocklessLazySMP::startWorkerThreads() {
    stop_threads.store(false, std::memory_order_relaxed);
    worker_threads.reserve(num_threads);
    
    for (int t = 0; t < num_threads; ++t) {
        worker_threads.emplace_back([this, t]() {
            #ifdef __linux__
            // Pin thread to CPU for better cache performance
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(t % MAX_THREADS, &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
            #endif
            
            workerThreadMain(t);
        });
    }
}

void MinMaxLocklessLazySMP::stopWorkerThreads() {
    stop_threads.store(true, std::memory_order_relaxed);
    for (auto& t : worker_threads) {
        if (t.joinable()) {
            t.join();
        }
    }
}

void MinMaxLocklessLazySMP::workerThreadMain(int thread_id) {
    ThreadData& td = *thread_data[thread_id];
    
    while (!stop_threads.load(std::memory_order_relaxed)) {
        // Wait for search to start
        while (!search_active.load(std::memory_order_acquire) &&
               !stop_threads.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        
        if (stop_threads.load(std::memory_order_relaxed)) break;
        
        // Execute appropriate search based on mode
        SearchMode mode = search_mode.load(std::memory_order_relaxed);
        if (mode == SearchMode::TIME_LIMITED) {
            searchThreadTimeLimited(td);
        } else if (mode == SearchMode::DEPTH_LIMITED) {
            searchThreadDepthLimited(td);
        }
        
        // Signal completion
        threads_finished.fetch_add(1, std::memory_order_release);
        
        // Wait for search to end
        while (search_active.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
}

void MinMaxLocklessLazySMP::searchThreadDepthLimited(ThreadData& td) {
    // DEPTH-LIMITED: Search to exact target depth (with thread's offset)
    int target = target_depth.load(std::memory_order_relaxed);
    int search_depth = std::max(1, target + td.depth_offset);
    int player = current_player.load(std::memory_order_relaxed);
    
    // Single search to target depth
    auto [score, move] = locklessNegamax(td, td.board, player,
                                         -std::numeric_limits<float>::infinity(),
                                         std::numeric_limits<float>::infinity(),
                                         search_depth);
    
    if (!move.empty()) {
        td.updateBestMove(score, move, search_depth);
    }
}

void MinMaxLocklessLazySMP::searchThreadTimeLimited(ThreadData& td) {
    // TIME-LIMITED: Iterative deepening until time runs out
    int player = current_player.load(std::memory_order_relaxed);
    
    // Start from depth 1 (or 2 if using step=2)
    int start_depth = (td.id_step == 2) ? 2 : 1;
    
    for (int depth = start_depth; depth <= MAX_SEARCH_DEPTH; depth += td.id_step) {
        if (shouldStop()) break;
        
        // Adjust depth based on thread's offset
        int search_depth = std::max(1, depth + td.depth_offset);
        
        float alpha = -std::numeric_limits<float>::infinity();
        float beta = std::numeric_limits<float>::infinity();
        
        // Use aspiration windows at higher depths
        if (td.use_aspiration && depth >= 4) {
            float current_best = td.best_score.load(std::memory_order_relaxed);
            if (current_best > -std::numeric_limits<float>::infinity()) {
                alpha = current_best - td.aspiration_delta;
                beta = current_best + td.aspiration_delta;
            }
        }
        
        auto [score, move] = locklessNegamax(td, td.board, player, alpha, beta, search_depth);
        
        // Re-search if aspiration window failed
        if (td.use_aspiration && depth >= 4) {
            if (score <= alpha) {
                // Failed low - search with open lower bound
                std::tie(score, move) = locklessNegamax(td, td.board, player,
                                                        -std::numeric_limits<float>::infinity(),
                                                        beta, search_depth);
            } else if (score >= beta) {
                // Failed high - search with open upper bound
                std::tie(score, move) = locklessNegamax(td, td.board, player,
                                                        alpha,
                                                        std::numeric_limits<float>::infinity(),
                                                        search_depth);
            }
        }
        
        if (!move.empty()) {
            td.updateBestMove(score, move, search_depth);
        }
        
        // Early exit if we found a winning move
        if (std::abs(score) > 10000.0f) {
            break;
        }

        // if (td.thread_id == 0) {
        //     // Main thread prints progress
        //     std::cout << "Thread " << td.thread_id << " completed depth " 
        //               << search_depth << ": score = " << score 
        //               << ", move = " << move << "\n";
        // }
    }
}

int64_t MinMaxLocklessLazySMP::getTimeMs() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

bool MinMaxLocklessLazySMP::shouldStop() const {
    if (search_mode.load(std::memory_order_relaxed) == SearchMode::TIME_LIMITED) {
        int64_t elapsed = getTimeMs() - search_start_time.load(std::memory_order_relaxed);
        return elapsed >= time_limit_ms.load(std::memory_order_relaxed);
    }
    return false;
}

// std::pair<float, std::string> MinMaxLocklessLazySMP::locklessNegamax(
//     ThreadData& td,
//     Board& board,
//     int playerColor,
//     float alpha,
//     float beta,
//     int depth
// ) {
//     td.nodes_searched.fetch_add(1, std::memory_order_relaxed);
    
//     // Check time limit for time-based searches
//     if (shouldStop()) {
//         return {0, ""};
//     }    

//     // uint64_t hash = zobristHasher.computeHash(board); 
//     uint64_t hash = td.zobristHasher.computeHash(td.board);

//     float tt_value;
//     std::string tt_move;
//     int tt_depth;
//     Flag tt_flag;

//     bool tt_hit = tt.probe(hash, tt_value, tt_move, tt_depth, tt_flag);

//     if (tt_hit) {
//         td.tb_hits.fetch_add(1, std::memory_order_relaxed);
//         if (tt_depth >= depth) {
//             if (tt_flag == Flag::EXACT) {
//                 return {tt_value, tt_move};
//             } else if (tt_flag == Flag::LOWER_BOUND && tt_value >= beta) {
//                 return {tt_value, tt_move};
//             } else if (tt_flag == Flag::UPPER_BOUND && tt_value <= alpha) {
//                 return {tt_value, tt_move};
//             }
//         }
//     }

//     BoardState state = board.GetBoardState();
//     if (state == BoardState::WhiteWins) {
//         float inf = std::numeric_limits<float>::infinity();
//         return (playerColor == 1) ? std::make_pair(inf, "") : std::make_pair(-inf, "");
//     } else if (state == BoardState::BlackWins) {
//         float inf = std::numeric_limits<float>::infinity();
//         return (playerColor == -1) ? std::make_pair(inf, "") : std::make_pair(-inf, "");
//     } else if (state == BoardState::Draw) {
//         return {0.0f, ""};
//     }

//     if (depth == 0) {
//         float eval = useEnhanced ? evaluate2(board, playerColor) : evaluate(board, playerColor);
//         tt.store(hash, eval, "", 0, Flag::EXACT);
//         return {eval, ""};
//     }

//     struct MoveEntry {
//         Move m;
//         std::string str;
//         float score;
//         float noise;
//     };

//     std::vector<MoveEntry> movelist;
//     auto validMovesPtr = board.GetValidMoves();

//     if (!validMovesPtr || validMovesPtr->empty()) {
//         return {0.0f, ""};
//     }

//     movelist.reserve(validMovesPtr->size());

//     for (const auto& move : *validMovesPtr) {
//         std::string moveStr;
//         if (!board.TryGetMoveString(move, moveStr)) continue;
//         if (board.TryPlayMove(move, moveStr)) {
//             float staticScore = useEnhanced ? evaluate2(board, playerColor) : evaluate(board, playerColor);
//             board.TryUndoLastMove();
//             if (tt_hit && moveStr == tt_move) {
//                 staticScore += 10000.0f;
//             }
//             movelist.push_back({move, moveStr, staticScore, 0.0f});
//         }
//     }

//     if (movelist.empty()) {
//         return {0.0f, ""};
//     }

//     // std::sort(movelist.begin(), movelist.end(),
//     //     [&td](const MoveEntry& a, const MoveEntry& b) {
//     //         if (td.thread_id == 0) {
//     //             return a.score > b.score;
//     //         }
//     //         float noise = std::uniform_real_distribution<float>(-0.5f, 0.5f)(td.rng);
//     //         return (a.score + noise) > b.score;
//     //     }
//     // );
//     // Add per-move jitter once (non–main threads only), then sort deterministically.
//     if (td.thread_id != 0) {
//         std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
//         for (auto& e : movelist) {
//             e.noise = dist(td.rng);
//         }
//     }
//     std::sort(movelist.begin(), movelist.end(),
//         [](const MoveEntry& a, const MoveEntry& b) {
//             return (a.score + a.noise) > (b.score + b.noise);
//         });

//     float originalAlpha = alpha;
//     float value = -std::numeric_limits<float>::infinity();
//     std::string bestMove = "";

//     for (const auto& entry : movelist) {
//         if (time_limit_reached.load(std::memory_order_relaxed)) break;
//         if (board.TryPlayMove(entry.m, entry.str)) {
//             auto [childScore, _] = locklessNegamax(
//                 td, board, -playerColor, -beta, -alpha, depth - 1
//             );
//             float score = -childScore;
//             board.TryUndoLastMove();
//             if (score > value) {
//                 value = score;
//                 bestMove = entry.str;
//             }
//             alpha = std::max(alpha, value);
//             if (alpha >= beta) break;
//         }
//     }

//     Flag flag = (value <= originalAlpha) ? Flag::UPPER_BOUND :
//                 (value >= beta) ? Flag::LOWER_BOUND : Flag::EXACT;

//     tt.store(hash, value, bestMove, depth, flag);

//     return {value, bestMove};
// }

std::pair<float, std::string> MinMaxLocklessLazySMP::locklessNegamax(
    ThreadData& td,
    Board& board,
    int playerColor,
    float alpha,
    float beta,
    int depth
) {
    td.nodes_searched.fetch_add(1, std::memory_order_relaxed);
    
    if (shouldStop()) {
        return {0, ""};
    }
    
    // Check TT
    uint64_t hash = td.zobristHasher.computeHash(board);
    float tt_value;
    std::string tt_move;
    int tt_depth;
    Flag tt_flag;
    
    bool tt_hit = tt.probe(hash, tt_value, tt_move, tt_depth, tt_flag);
    if (tt_hit) {
        td.tb_hits.fetch_add(1, std::memory_order_relaxed);
        if (tt_depth >= depth) {
            if (tt_flag == Flag::EXACT) {
                return {tt_value, tt_move};
            } else if (tt_flag == Flag::LOWER_BOUND && tt_value >= beta) {
                td.beta_cutoffs.fetch_add(1, std::memory_order_relaxed);  // COUNT CUTOFF!
                return {tt_value, tt_move};
            } else if (tt_flag == Flag::UPPER_BOUND && tt_value <= alpha) {
                return {tt_value, tt_move};
            }
        }
    }
    
    // Terminal states
    BoardState state = board.GetBoardState();
    if (state == BoardState::WhiteWins) {
        float value = (playerColor == 1) ? 100000.0f : -100000.0f;
        return {value, ""};
    } else if (state == BoardState::BlackWins) {
        float value = (playerColor == -1) ? 100000.0f : -100000.0f;
        return {value, ""};
    } else if (state == BoardState::Draw) {
        return {0.0f, ""};
    }
    
    // Leaf evaluation
    if (depth == 0) {
        float eval = evaluateFast(board, playerColor);  // Use fast eval!
        tt.store(hash, eval, "", 0, Flag::EXACT);
        return {eval, ""};
    }
    
    // Move generation
    struct MoveEntry {
        Move m;
        std::string str;
        float score;
        float noise = 0.0f;
    };
    
    std::vector<MoveEntry> movelist;
    auto validMovesPtr = board.GetValidMoves();
    
    if (!validMovesPtr || validMovesPtr->empty()) {
        return {0.0f, ""};
    }
    
    movelist.reserve(validMovesPtr->size());
    
    // Generate moves with BETTER static ordering
    for (const auto& move : *validMovesPtr) {
        std::string moveStr;
        if (!board.TryGetMoveString(move, moveStr)) continue;
        
        // Static move ordering heuristics
        float staticScore = 0.0f;
        
        // TT move gets huge bonus
        if (tt_hit && moveStr == tt_move) {
            staticScore += 10000.0f;
        }
        
        // Move ordering heuristics WITHOUT playing the move
        BugType bugType = GetBugType(move.PieceName);
        
        // Prioritize queen moves when enemy queen is surrounded
        if (bugType == BugType::QueenBee) {
            staticScore += 100.0f;
        }
        
        // Prioritize placing attacking pieces (ant, beetle)
        if (bugType == BugType::SoldierAnt || bugType == BugType::Beetle) {
            staticScore += 50.0f;
        }
        
        // Prioritize moves that place pieces next to enemy queen
        // (This would require checking move destination)
        
        movelist.push_back({move, moveStr, staticScore, 0.0f});
    }
    
    if (movelist.empty()) {
        return {0.0f, ""};
    }
    
    // Add noise for non-main threads
    if (td.thread_id != 0) {
        std::uniform_real_distribution<float> dist(-5.0f, 5.0f);  // Larger noise range
        for (auto& e : movelist) {
            e.noise = dist(td.rng);
        }
    }
    
    // Sort moves
    std::sort(movelist.begin(), movelist.end(),
        [](const MoveEntry& a, const MoveEntry& b) {
            return (a.score + a.noise) > (b.score + b.noise);
        });
    
    // Search moves
    float originalAlpha = alpha;
    float value = -std::numeric_limits<float>::infinity();
    std::string bestMove = "";
    int moveCount = 0;
    
    for (const auto& entry : movelist) {
        if (shouldStop()) break;
        
        if (board.TryPlayMove(entry.m, entry.str)) {
            moveCount++;
            
            // Late Move Reduction (LMR)
            int reduction = 0;
            if (moveCount > 4 && depth > 3) {
                reduction = 1;  // Search later moves shallower
            }
            
            auto [childScore, _] = locklessNegamax(
                td, board, -playerColor, -beta, -alpha, depth - 1 - reduction
            );
            float score = -childScore;
            
            // Re-search if LMR found something good
            if (reduction > 0 && score > alpha) {
                std::tie(childScore, _) = locklessNegamax(
                    td, board, -playerColor, -beta, -alpha, depth - 1
                );
                score = -childScore;
            }
            
            board.TryUndoLastMove();
            
            if (score > value) {
                value = score;
                bestMove = entry.str;
            }
            
            alpha = std::max(alpha, value);
            
            // Beta cutoff - THIS IS CRITICAL!
            if (alpha >= beta) {
                td.beta_cutoffs.fetch_add(1, std::memory_order_relaxed);  // COUNT IT!
                
                // Killer move heuristic: remember good cutoff moves
                // (You could store this move as a "killer" for this depth)
                
                break;  // PRUNE!
            }
        }
    }
    
    // Store in TT
    Flag flag;
    if (value <= originalAlpha) {
        flag = Flag::UPPER_BOUND;  // Failed low
    } else if (value >= beta) {
        flag = Flag::LOWER_BOUND;  // Failed high (beta cutoff)
    } else {
        flag = Flag::EXACT;
    }
    
    tt.store(hash, value, bestMove, depth, flag);
    
    return {value, bestMove};
}

} // namespace MzingaCpp