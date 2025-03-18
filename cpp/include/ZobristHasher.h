#ifndef ZOBRISTHASHER_H
#define ZOBRISTHASHER_H

#include "Board.h"
#include "Position.h"
#include <unordered_map>
#include <random>
#include <utility>

namespace MzingaCpp {

// Specializzazione dell'hash per PieceName
struct PieceNameHasher {
    size_t operator()(PieceName piece) const {
        return std::hash<int>{}(static_cast<int>(piece));  // Usa un cast per ottenere un intero da PieceName
    }
};

// Functor per l'hash delle posizioni
struct ZobristPositionHasher {
    size_t operator()(const Position& pos) const {
        size_t h1 = std::hash<int>{}(pos.Q);
        size_t h2 = std::hash<int>{}(pos.R);
        size_t h3 = std::hash<int>{}(pos.Stack);
        return h1 ^ (h2 << 1) ^ (h3 << 2);  // Combina gli hash dei componenti Q, R e Stack
    }
};

// Functor per l'hash della coppia PieceName e Position
struct PairHasher {
    size_t operator()(const std::pair<PieceName, Position>& p) const {
        size_t h1 = PieceNameHasher{}(p.first);
        size_t h2 = ZobristPositionHasher{}(p.second);
        return h1 ^ (h2 << 1);  // Combina gli hash di PieceName e Position
    }
};

class ZobristHasher {
public:
    ZobristHasher();  // Costruttore

    // Funzione per ottenere o generare un valore Zobrist per una coppia (pezzo, posizione)
    uint64_t getOrGenerateHash(PieceName piece, Position pos);

    // Calcola l'hash iniziale della board
    uint64_t computeHash(Board& board);

    // Aggiorna l'hash quando un pezzo si muove
    uint64_t updateHash(uint64_t currentHash, PieceName piece, Position oldPos, Position newPos);

private:
    // Tabella Zobrist che mappa la coppia (PieceName, Position) a un valore hash
    std::unordered_map<std::pair<PieceName, Position>, uint64_t, PairHasher> zobristTable;
};

} // namespace MzingaCpp

#endif // ZOBRISTHASHER_H
