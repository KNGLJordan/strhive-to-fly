#ifndef ZOBRISTHASHER_H
#define ZOBRISTHASHER_H

#include "Board.h"
#include "Position.h"
#include <unordered_map>
#include <random>
#include <utility>
namespace MzingaCpp {

// mixing veloce e di qualità
static inline uint64_t splitmix64(uint64_t x) noexcept {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// zig-zag per coordinate negative
static inline uint64_t zz(int v) noexcept {
    return (static_cast<uint32_t>(v) << 1) ^ static_cast<uint32_t>(v >> 31);
}

class ZobristHasher {
public:
    uint64_t computeHash( Board& board) ;

    uint64_t updateHash(uint64_t h, PieceName piece, Position oldPos, Position newPos) const;

private:
    static constexpr uint64_t SEED = 0x123456789ABCDEF0ull;

    static constexpr uint64_t TOGGLE_TURN = 2658527535540318469ull; 
    
    static inline uint64_t zobristValue(PieceName p, const Position& pos) noexcept;
};

} // namespace MzingaCpp


#endif // ZOBRISTHASHER_H
