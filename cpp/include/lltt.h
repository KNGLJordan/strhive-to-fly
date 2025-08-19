#pragma once

#include <vector>
#include <atomic>
#include <string>
#include <cstring>
#include <cstdint>
#include "Enums.h"
#include "MinMaxZobrist.h"

namespace MzingaCpp {

class LocklessTranspositionTable {
private:
    // Solution 1: Split the entry into two parts (Stockfish-style)
    // First 8 bytes: key (for validation)
    // Next 8 bytes: packed data (value, depth, flag, move index)
    struct TTEntry {
        std::atomic<uint64_t> key;
        std::atomic<uint64_t> data;
        
        TTEntry() : key(0), data(0) {}
    };
    
    // Pack/unpack helpers for the 64-bit data field
    static uint64_t packData(int16_t value, uint16_t depth, uint16_t flag, uint16_t moveIndex) {
        return (uint64_t(value + 32768) << 48) |  // Store value + offset to make it unsigned
               (uint64_t(depth) << 32) |
               (uint64_t(flag) << 16) |
               uint64_t(moveIndex);
    }
    
    static void unpackData(uint64_t data, int16_t& value, uint16_t& depth, uint16_t& flag, uint16_t& moveIndex) {
        value = int16_t((data >> 48) - 32768);
        depth = uint16_t(data >> 32);
        flag = uint16_t(data >> 16);
        moveIndex = uint16_t(data);
    }
    
    static constexpr size_t TT_SIZE = 16 * 1024 * 1024;  // 16M entries
    static constexpr size_t TT_MASK = TT_SIZE - 1;
    
    std::vector<TTEntry> entries;
    
    // Separate move storage (indexed by hash)
    struct MoveEntry {
        char move[16];
        MoveEntry() { std::memset(move, 0, sizeof(move)); }
    };
    std::vector<MoveEntry> moves;
    
public:
    LocklessTranspositionTable();
    bool probe(uint64_t hash, float& value, std::string& move, int& depth, Flag& flag);
    void store(uint64_t hash, float value, const std::string& move, int depth, Flag flag);
    void clear();
    size_t approximateUsage() const;
};


// // Alternative Solution 2: Use 128-bit atomics (if available on your platform)
// #ifdef __x86_64__
// class LocklessTranspositionTable128 {
// private:
//     // Use 128-bit atomic compare-exchange (available on x86-64)
//     struct alignas(16) TTEntry128 {
//         uint64_t key;
//         union {
//             struct {
//                 float value;
//                 uint16_t depth;
//                 uint16_t flag;
//             };
//             uint64_t data;
//         };
        
//         TTEntry128() : key(0), data(0) {}
//     };
    
//     static_assert(sizeof(TTEntry128) == 16, "TTEntry128 must be exactly 16 bytes");
    
//     std::vector<TTEntry128> entries;
//     std::vector<std::string> move_table;  // Separate move storage
//     std::hash<std::string> move_hasher;
    
//     static constexpr size_t TT_SIZE = 16 * 1024 * 1024;
//     static constexpr size_t TT_MASK = TT_SIZE - 1;
    
// public:
//     LocklessTranspositionTable128() : entries(TT_SIZE), move_table(65536) {}
    
//     bool probe(uint64_t hash, float& value, std::string& move, int& depth, Flag& flag) {
//         size_t index = hash & TT_MASK;
//         TTEntry128 entry;
        
//         // Use 128-bit atomic load (x86-64 specific)
//         __atomic_load(&entries[index], &entry, __ATOMIC_RELAXED);
        
//         if (entry.key == hash) {
//             value = entry.value;
//             depth = entry.depth;
//             flag = static_cast<Flag>(entry.flag);
            
//             // Retrieve move from separate table
//             size_t move_idx = hash % move_table.size();
//             move = move_table[move_idx];
//             return true;
//         }
//         return false;
//     }
    
//     void store(uint64_t hash, float value, const std::string& move, int depth, Flag flag) {
//         size_t index = hash & TT_MASK;
        
//         TTEntry128 newEntry;
//         newEntry.key = hash;
//         newEntry.value = value;
//         newEntry.depth = depth;
//         newEntry.flag = static_cast<uint16_t>(flag);
        
//         // Store with 128-bit atomic operation
//         __atomic_store(&entries[index], &newEntry, __ATOMIC_RELAXED);
        
//         // Store move separately (not atomic, but collisions are OK)
//         size_t move_idx = hash % move_table.size();
//         move_table[move_idx] = move;
//     }
// };
// #endif

} // namespace MzingaCpp
