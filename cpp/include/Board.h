// Copyright (c) Jon Thysell <http://jonthysell.com>
// Licensed under the MIT License.

#ifndef BOARD_H
#define BOARD_H

#include <memory>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <functional>

#include "Constants.h"
#include "Enums.h"
#include "Move.h"
#include "MoveSet.h"
#include "Position.h"
#include "PositionSet.h"

namespace MzingaCpp
{
// Hash wrapper for Position to use in unordered containers
struct PositionHashWrapper {
    size_t operator()(Position const &p) const noexcept
    {
        return MzingaCpp::hash(p);
    }
};

struct PieceMetrics
{
    int InPlay = 0;
    int IsPinned = 0; 
    int IsCovered = 0;
    int NoisyMoveCount = 0;
    int QuietMoveCount = 0;
    int FriendlyNeighborCount = 0;
    int EnemyNeighborCount = 0;
};

struct BoardMetrics
{
    MzingaCpp::BoardState BoardState;
    int PiecesInPlay = 0;
    int PiecesInHand = 0;
    
    // Metrics for each piece type - you can index by PieceName
    PieceMetrics pieceMetrics[(int)PieceName::NumPieceNames];
    
    // Accessor for easier syntax
    PieceMetrics& operator[](PieceName pieceName) 
    {
        return pieceMetrics[(int)pieceName];
    }
    
    const PieceMetrics& operator[](PieceName pieceName) const 
    {
        return pieceMetrics[(int)pieceName];
    }
};

class Board
{
  public:
  
    Board();
    Board(GameType gameType);

    BoardState GetBoardState();
    int GetCurrentTurn();
    GameType GetGameType() const;

    std::string GetGameString() const;
    std::shared_ptr<MoveSet> GetValidMoves();

    
    void FastPlay(Move const &move, std::string moveString);
    bool TryPlayMove(Move const &move, std::string moveString);
    bool TryUndoLastMove();

    bool TryGetMoveString(Move const &move, std::string &result);
    bool TryParseMove(std::string moveString, Move &result, std::string &resultString);

    long CalculatePerft(int depth);

    std::shared_ptr<Board> Clone();
    
    int CountNeighbors(PieceName const &pieceName);

    std::vector<std::pair<PieceName, Position>> GetPiecesAndPositions();

    BoardMetrics GetBoardMetrics();
    bool IsNoisyMove(const Move& move);
    void SetCurrentPlayerMetrics(BoardMetrics& boardMetrics, std::shared_ptr<MoveSet> moveSet);
    bool IsPinned(PieceName pieceName, std::shared_ptr<MoveSet> moveSet, int& noisyCount, int& quietCount);
    int CountNeighbors(PieceName pieceName, int& friendlyCount, int& enemyCount);
    std::vector<std::pair<Position, PieceName>> GetPiecesInPlay();

    PieceName m_lastPieceMoved = PieceName::INVALID;
    
  private:
    // Articulation points (cut vertices) cache
    mutable bool m_articulationPositionsReady = false;
    mutable std::unordered_set<Position, PositionHashWrapper> m_articulationPositions;
    
    // Cached valid placements for pieces around enemy queen
    mutable std::unordered_set<Position, PositionHashWrapper> m_cachedEnemyQueenNeighbors;
    mutable bool m_cachedEnemyQueenNeighborsReady = false;

    // Updates articulation points using Tarjan's algorithm
    void UpdateArticulationPoints() const;
    
    void GetValidMoves(PieceName const &pieceName, std::shared_ptr<MoveSet> moveSet);
    void CalculateValidPlacements();

    void GetValidQueenBeeMoves(PieceName const &pieceName, std::shared_ptr<MoveSet> moveSet);
    void GetValidSpiderMoves(PieceName const &pieceName, std::shared_ptr<MoveSet> moveSet);
    void GetValidBeetleMoves(PieceName const &pieceName, std::shared_ptr<MoveSet> moveSet);
    void GetValidGrasshopperMoves(PieceName const &pieceName, std::shared_ptr<MoveSet> moveSet);
    void GetValidSoldierAntMoves(PieceName const &pieceName, std::shared_ptr<MoveSet> moveSet);
    void GetValidMosquitoMoves(PieceName const &pieceName, std::shared_ptr<MoveSet> moveSet,
                               bool const &specialAbilityOnly);
    void GetValidLadybugMoves(PieceName const &pieceName, std::shared_ptr<MoveSet> moveSet);
    void GetValidPillbugBasicMoves(PieceName const &pieceName, std::shared_ptr<MoveSet> moveSet);
    void GetValidPillbugSpecialMoves(PieceName const &pieceName, std::shared_ptr<MoveSet> moveSet);

    void GetValidSlides(PieceName const &pieceName, std::shared_ptr<MoveSet> moveSet, int fixedRange);
    void GetValidSlides(PieceName const &pieceName, std::shared_ptr<MoveSet> moveSet, Position const &startingPosition,
                        Position const &lastPosition, Position const &currentPosition);
    void GetValidSlides(PieceName const &pieceName, std::shared_ptr<MoveSet> moveSet, Position const &startingPosition,
                        Position const &lastPosition, Position const &currentPosition, int remainingSlides);

    void TrustedPlay(Move const &move);

    bool PlacingPieceInOrder(PieceName const &pieceName);

    Position GetPosition(PieceName const &pieceName);
    void SetPosition(PieceName const &pieceName, Position const &position);

    PieceName GetPieceAt(Position const &position);
    PieceName GetPieceAt(Position const &position, Direction const &direction);
    PieceName GetPieceOnTopAt(Position const &position);
    bool HasPieceAt(Position const &position);
    bool HasPieceAt(Position const &position, Direction const &direction);

    bool PieceInHand(PieceName const &pieceName);
    bool PieceInPlay(PieceName const &pieceName);

    bool PieceIsOnTop(PieceName const &pieceName);

    // Improved hive connectivity checking
    bool CanMoveWithoutBreakingHive(PieceName const &pieceName);
    bool IsOneHive();

    void ResetState();
    void ResetCaches();

    GameType m_gameType = GameType::Base;
    BoardState m_boardState = BoardState::NotStarted;
    Color m_currentColor = Color::White;
    int m_currentTurn = 0;


    Position m_piecePositions[(int)PieceName::NumPieceNames];
    PieceName m_pieceGrid[BoardSize][BoardSize][BoardStackSize];

    std::vector<Move> m_moveHistory;
    std::vector<std::string> m_moveHistoryStr;

    PositionSet m_cachedValidPlacements;
    bool m_cachedValidPlacementsReady = false;
};
} // namespace MzingaCpp

#endif