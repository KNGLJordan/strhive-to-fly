#ifndef LAZYSMP_H
#define LAZYSMP_H

#include "MinMaxZobrist.h"
#include "ZobristHasher.h"
#include "lltt.h"
#include <atomic>
#include <thread>
#include <vector>
#include <memory>
#include <random>

namespace MzingaCpp {

class MinMaxLocklessLazySMP : public MinMaxZobrist {
private:
    static constexpr int MAX_THREADS = 48;
    static constexpr int DEFAULT_THREADS = 44;
    static constexpr int MAX_SEARCH_DEPTH = 100;  // Maximum depth for iterative deepening
    
    // Thread data structure
    struct alignas(64) ThreadData {  // Cache line aligned
        int thread_id;
        Board board;    
        ZobristHasher zobristHasher;
        std::mt19937 rng;
        
        // Search parameters for diversity
        int depth_offset;        // How much to adjust base depth
        float aspiration_delta;  // Aspiration window size
        bool use_aspiration;     // Whether to use aspiration windows
        int id_step;            // Step size for iterative deepening (1 or 2)
        
        // Results for current search
        std::atomic<float> best_score;
        char best_move_str[32];
        std::atomic<uint64_t> best_move_hash;
        std::atomic<int> completed_depth;  // Highest depth completed
        
        // Statistics
        std::atomic<uint64_t> nodes_searched;
        std::atomic<uint32_t> tb_hits;
        std::atomic<uint32_t> beta_cutoffs;
        
        ThreadData(int id);
        void resetSearch();
        void updateBestMove(float score, const std::string& move, int depth);
        std::string getBestMove() const;
    };
    
    // Shared resources
    std::vector<std::unique_ptr<ThreadData>> thread_data;
    int num_threads;
    
    // Worker thread management
    std::vector<std::thread> worker_threads;
    std::atomic<bool> stop_threads{false};
    
    // Search control flags
    enum class SearchMode { NONE, DEPTH_LIMITED, TIME_LIMITED };
    std::atomic<SearchMode> search_mode{SearchMode::NONE};
    std::atomic<bool> search_active{false};
    std::atomic<bool> time_limit_reached{false};
    std::atomic<int> target_depth{0};        // For depth-limited search
    std::atomic<int> current_player{1};
    std::atomic<int> threads_finished{0};
    std::atomic<int64_t> search_start_time{0};
    std::atomic<int> time_limit_ms{0};
    
public:
    LocklessTranspositionTable tt;
    MinMaxLocklessLazySMP(bool useEnhancedEval, int threads = DEFAULT_THREADS);
    MinMaxLocklessLazySMP(const EvaluationWeights& w, bool useEnhancedEval, int threads = DEFAULT_THREADS);
    ~MinMaxLocklessLazySMP();
    
    // Board management
    void initializeBoards(GameType gameType);
    void syncBoards(Board& board);
    void applyMoveToAllBoards(const Move& move, const std::string& moveStr);
    void applyUndoToAllBoards(int numMoves = 1);
    
    // Main search interface
    std::string calculateBestMove(Board& board, int maxDepth, int timeLimit) override;
    
private:
    std::string searchLockless(int maxDepth, int timeLimit);
    void startWorkerThreads();
    void stopWorkerThreads();
    void workerThreadMain(int thread_id);
    void searchThreadDepthLimited(ThreadData& td);
    void searchThreadTimeLimited(ThreadData& td);
    
    std::pair<float, std::string> locklessNegamax(
        ThreadData& td,
        Board& board,
        int playerColor,
        float alpha,
        float beta,
        int depth
    );
    
    // Helper functions
    int64_t getTimeMs() const;
    bool shouldStop() const;
};

} // namespace MzingaCpp
#endif // LAZYSMP_H