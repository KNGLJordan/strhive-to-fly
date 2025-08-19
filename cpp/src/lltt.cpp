
// lltt.cpp
#include "lltt.h"
#include <cstring>
#include <algorithm>

namespace MzingaCpp {

LocklessTranspositionTable::LocklessTranspositionTable()
    : entries(TT_SIZE), moves(TT_SIZE) {}

bool LocklessTranspositionTable::probe(uint64_t hash, float& value, std::string& move, int& depth, Flag& flag) {
    size_t index = hash & TT_MASK;
    
    // Load both 64-bit values atomically
    uint64_t stored_key = entries[index].key.load(std::memory_order_relaxed);
    uint64_t stored_data = entries[index].data.load(std::memory_order_relaxed);
    
    if (stored_key == hash) {
        int16_t packed_value;
        uint16_t packed_depth, packed_flag, move_index;
        unpackData(stored_data, packed_value, packed_depth, packed_flag, move_index);
        
        value = static_cast<float>(packed_value) / 100.0f;  // Convert from centipawns
        depth = packed_depth;
        flag = static_cast<Flag>(packed_flag);
        
        // Retrieve move from separate storage
        if (move_index < TT_SIZE) {
            move = std::string(moves[move_index].move);
        } else {
            move = "";
        }
        
        return true;
    }
    return false;
}

void LocklessTranspositionTable::store(uint64_t hash, float value, const std::string& move, int depth, Flag flag) {
    size_t index = hash & TT_MASK;
    
    // Check current entry
    uint64_t current_key = entries[index].key.load(std::memory_order_relaxed);
    uint64_t current_data = entries[index].data.load(std::memory_order_relaxed);
    
    int16_t current_value;
    uint16_t current_depth, current_flag, current_move_index;
    unpackData(current_data, current_value, current_depth, current_flag, current_move_index);
    
    // Replace if empty, same position, or deeper search
    if (current_key == 0 || current_key == hash || current_depth < depth) {
        // Store move in separate table
        uint16_t move_index = index;  // Simple strategy: use same index
        if (!move.empty()) {
            size_t move_len = std::min(move.length(), size_t(31));
            std::memcpy(moves[move_index].move, move.c_str(), move_len);
            moves[move_index].move[move_len] = '\0';
        }
        
        // Convert value to centipawns (fixed point)
        int16_t packed_value = static_cast<int16_t>(std::min(32767.0f, std::max(-32768.0f, value * 100)));
        
        // Pack and store
        uint64_t new_data = packData(packed_value, depth, static_cast<uint16_t>(flag), move_index);
        
        entries[index].key.store(hash, std::memory_order_relaxed);
        entries[index].data.store(new_data, std::memory_order_relaxed);
    }
}

void LocklessTranspositionTable::clear() {
    for (auto& entry : entries) {
        entry.key.store(0, std::memory_order_relaxed);
        entry.data.store(0, std::memory_order_relaxed);
    }
    for (auto& move : moves) {
        std::memset(move.move, 0, sizeof(move.move));
    }
}

size_t LocklessTranspositionTable::approximateUsage() const {
    size_t used = 0;
    for (size_t i = 0; i < 1000; ++i) {
        size_t idx = (i * 1000) & TT_MASK;
        if (entries[idx].key.load(std::memory_order_relaxed) != 0) {
            used++;
        }
    }
    return (used * TT_SIZE) / 1000;
}

} // namespace MzingaCpp