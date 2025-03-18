#include "ZobristHasher.h"
#include "Board.h"
#include "Position.h"

namespace MzingaCpp {

// Generatore di numeri casuali per valori Zobrist
std::mt19937_64 rng(123456789);
std::uniform_int_distribution<uint64_t> dist;

ZobristHasher::ZobristHasher() {}

// Funzione per ottenere o generare un valore Zobrist per una coppia (pezzo, posizione)
uint64_t ZobristHasher::getOrGenerateHash(PieceName piece, Position pos) {
    auto key = std::make_pair(piece, pos);
    if (zobristTable.find(key) == zobristTable.end()) {
        zobristTable[key] = dist(rng);
    }
    return zobristTable[key];
}

// Calcola l'hash iniziale della board
uint64_t ZobristHasher::computeHash(Board& board) {
    uint64_t hash = 0;
    for (const auto& [piece, pos] : board.GetPiecesAndPositions()) {
        hash ^= getOrGenerateHash(piece, pos);
    }
    return hash;
}

// Aggiorna l'hash quando un pezzo si muove
uint64_t ZobristHasher::updateHash(uint64_t currentHash, PieceName piece, Position oldPos, Position newPos) {
    currentHash ^= getOrGenerateHash(piece, oldPos);  // Rimuove il pezzo dalla vecchia posizione
    currentHash ^= getOrGenerateHash(piece, newPos);  // Aggiunge il pezzo alla nuova posizione
    return currentHash;
}

} // namespace MzingaCpp
