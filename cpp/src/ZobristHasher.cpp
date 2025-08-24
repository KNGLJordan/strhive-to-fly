#include "ZobristHasher.h"
#include "Board.h"
#include "Position.h"

namespace MzingaCpp {

// Calcola l'hash iniziale della board
uint64_t ZobristHasher::computeHash(Board& board) {
    uint64_t h = 0;
    for (const auto& [piece, pos] : board.GetPiecesAndPositions()) {
        h ^= zobristValue(piece, pos);
    }
    if (board.GetCurrentTurn() % 2 == 0) {
        h ^= TOGGLE_TURN; // cambia il turno
    }
    auto lastPiece_Moved = board.m_lastPieceMoved;
    // if there is a pillbug and the lastpiece moved is near the pillbub, we add the info about the last piece moved
    h = splitmix64(h ^ zz(static_cast<int>(lastPiece_Moved)));
    return h;
}

inline uint64_t ZobristHasher::zobristValue(PieceName p, const Position& pos) noexcept {
    // combino pezzo e coordinate con un paio di mix
    uint64_t x = ZobristHasher::SEED ^ static_cast<uint64_t>(static_cast<int>(p));
    x = splitmix64(x ^ zz(pos.Q));
    x = splitmix64(x ^ zz(pos.R));
    x = splitmix64(x ^ static_cast<uint64_t>(pos.Stack));
    return x;
}

// Aggiorna l'hash quando un pezzo si muove
uint64_t ZobristHasher::updateHash(uint64_t h, PieceName piece, Position oldPos, Position newPos) const {
    h ^= zobristValue(piece, oldPos);
    h ^= zobristValue(piece, newPos);
    return h;
}

} // namespace MzingaCpp