// Copyright (c) Jon Thysell <http://jonthysell.com>
// Licensed under the MIT License.

#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <algorithm>

#include "Engine.h"
#include "Board.h"
#include "lazysmp.h"

void WriteLine(std::string line)
{
    std::cout << line << std::endl;
}

struct TestCase {
    std::string start;
    std::vector<std::string> correct_moves;
    std::string description;
    bool is_winning;
};

// Helper function to parse game string and create board
std::shared_ptr<MzingaCpp::Board> CreateBoardFromGameString(const std::string& gameString, MzingaCpp::MinMaxLocklessLazySMP& lazysmp) {
    // Parse the game string to extract game type and moves
    size_t firstSemicolon = gameString.find(';');
    std::string gameTypeStr = gameString.substr(0, firstSemicolon);
    
    MzingaCpp::GameType gameType = MzingaCpp::GameType::BaseMLP;
    
    auto board = std::make_shared<MzingaCpp::Board>(gameType);
    
    // If it's just the starting position, return the board
    if (gameString == "Base+MLP") {
        return board;
    }
    
    // Parse and play moves
    size_t pos = gameString.find(';', firstSemicolon + 1); // Skip board state
    pos = gameString.find(';', pos + 1); // Skip current player
    pos++;
    
    while (pos < gameString.length()) {
        size_t nextPos = gameString.find(';', pos);
        if (nextPos == std::string::npos) {
            nextPos = gameString.length();
        }
        
        std::string moveStr = gameString.substr(pos, nextPos - pos);
        if (!moveStr.empty()) {
            MzingaCpp::Move move;
            std::string resultString;
            if (board->TryParseMove(moveStr, move, resultString)) {
                board->TryPlayMove(move, resultString);
                // apply to all threads
                lazysmp.applyMoveToAllBoards(move, resultString);
            } else {
                std::cout << "Failed to parse move: " << moveStr << std::endl;
            }
        }
        
        pos = nextPos + 1;
    }
    
    return board;
}

void printFirstLayer(MzingaCpp::Board& board, MzingaCpp::MinMaxLocklessLazySMP& lazysmp) {
    // get valid moves
    auto validMoves = board.GetValidMoves();
    // for each move, play it and get the value from lazysmp.tt
    MzingaCpp::ZobristHasher hasher;
    for (const auto& move : *validMoves) {
        std::string moveStr;
        if (board.TryGetMoveString(move, moveStr)) {
            if (!board.TryPlayMove(move, moveStr)) {
                std::cout << "Failed to play move: " << moveStr << std::endl;
                continue;
            }
            auto hash = lazysmp.zobristHasher.computeHash(board);
            auto hash2 = hasher.computeHash(board);

            if (!board.TryUndoLastMove())
            {
                std::cout << "Failed to undo move: " << moveStr << std::endl;
                continue;
            }

            std::cout << "Hash: " << hash << ", HashCheck: " << hash2 << ", Move: " << moveStr;
            // float& value, std::string& move, int& depth, Flag& flag
            float value = 0.0f;
            std::string moveStr;
            int depth = 0;
            MzingaCpp::Flag flag = MzingaCpp::Flag::EXACT; // Assuming exact for simplicity
            auto result = lazysmp.tt.probe(hash, value, moveStr, depth, flag);

            if (!result) {
                std::cout << ". NO entry found!!! " << std::endl;
                continue;
            }
            
            // Get the value from the lazySMP
            std::cout << ", Value: " << value << ", Depth: " << depth << ", Flag: " << static_cast<int>(flag) << std::endl;
        }
    }
}

// Function to run a single test case using engine's bestmove
bool RunTestCase(const TestCase& testCase, int testNum) {
    std::cout << "\n=== Test Case " << testNum << " ===" << std::endl;
    std::cout << "Description: " << testCase.description << std::endl;
    std::cout << "Expected winning: " << (testCase.is_winning ? "Yes" : "No") << std::endl;
    std::cout << "Expected moves: ";
    for (const auto& move : testCase.correct_moves) {
        std::cout << move << " ";
    }
    std::cout << std::endl;

    MzingaCpp::MinMaxLocklessLazySMP lazysmp = MzingaCpp::MinMaxLocklessLazySMP(false);

    bool engineMoveIsCorrect = false;

    try {
        // Create board from game string
        auto board = CreateBoardFromGameString(testCase.start, lazysmp);

        const auto board_str = board->GetGameString(); 
        // Set the position in the engine
        std::cout << "Setting position: " << board_str << std::endl;
        
        // Get the best move from engine with 5 second time limit
        std::cout << "Getting best move (5 seconds)..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();

        auto engineMove = lazysmp.calculateBestMove(*board, 0, 5);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "Bestmove took: " << duration.count() << " ms" << std::endl;
        
        // Check against all expected moves
        if(testCase.is_winning){
            MzingaCpp::Move move;
            std::string moveString;
            if (board->TryParseMove(engineMove, move, moveString) && board->TryPlayMove(move, moveString))
            {
                if(board->GetBoardState() != MzingaCpp::BoardState::WhiteWins && 
                   board->GetBoardState() != MzingaCpp::BoardState::BlackWins) {
                    std::cout << "❌ Winning move did not lead to a win!" << std::endl;
                    engineMoveIsCorrect = false;
                }
                else{
                    std::cout << "✅ Winning move selected: " << moveString << std::endl;
                    engineMoveIsCorrect = true;
                    
                }
                
            } else {
                std::cout << "[WARNING] Error playing the move selected" << std::endl;
            }

        }
        else{
            for (const auto& expectedMove : testCase.correct_moves) {
                std::string cleanExpected = expectedMove;
                
                // Handle special "play" prefix
                if (cleanExpected.substr(0, 5) == "play ") {
                    cleanExpected = cleanExpected.substr(5);
                }
                
                std::cout << "Comparing '" << engineMove << "' with '" << cleanExpected << "'" << std::endl;
                
                if (engineMove == cleanExpected) {
                    engineMoveIsCorrect = true;
                    std::cout << "✅ Engine move matches expected move: " << cleanExpected << std::endl;
                    break;
                }
            }
        }
        
        if (!engineMoveIsCorrect) {
            std::cout << "❌ Engine move '" << engineMove << "' is not among expected moves" << std::endl;
            
            printFirstLayer(*board, lazysmp);
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ Test FAILED with exception: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cout << "❌ Test FAILED with unknown exception" << std::endl;
        return false;
    }
    return engineMoveIsCorrect;
}

int main(int argc, char *argv[])
{
    std::cout << "=== Hive Engine Test Suite ===" << std::endl;
    
    // Define test cases
    std::vector<TestCase> testCases = {
        {
            "Base+MLP;InProgress;Black[17];wP;bA1 wP-;wL /wP;bQ bA1-;wM \\wP;bQ bA1/;wQ wM/;bA2 bQ-;wL bQ\\;bB1 \\bA2;wB1 -wM;bQ -bB1;wQ \\bQ;bA3 bB1/;wA1 \\wM;bA3 bA2\\;wA1 wL\\;bA3 wA1-;wB1 wM;bB2 bA2-;wS1 \\wB1;bB2 bA2/;wA2 /wS1;bB2 bB1/;wA2 bB2/;bA3 \\wS1;wA1 \\bA3;bM bB2\\;wB1 /bQ;bA1 /bA3;wL bQ\\;bA2 -wQ;wM -bA1",
            {"palle"},
            "esplodeva",
            false
        },
        {
            "newgame Base+MLP;InProgress;Black[9];wS1;bB1 wS1-;wQ \\wS1;bG1 bB1\\;wA1 /wQ;bG2 bG1-;wB1 wQ/;bQ \\bG2;wG1 \\wB1;bG2 wS1\\;wA1 -wQ;bG2 /wG1;wG1 /wA1;bG1 wB1\\;wA2 -wA1;bA1 bQ/;wA2 wB1/",
            {"bA1 -wS1"},
            "winning move selected",
            true
        },
        {
            "newgame Base+MLP",
            {"wS1", "wB1", "wG1", "wA1", "wM", "wL", "wP"},
            "starting position, all moves are correct. v should be 0.5",
            false
        },
        {
            "newgame Base+MLP;InProgress;White[31];wS1;bB1 wS1-;wQ \\wS1;bG1 bB1\\;wA1 /wQ;bG2 bG1-;wB1 wQ/;bQ \\bG2;wG1 \\wB1;bG2 wS1\\;wA1 -wQ;bG2 /wG1;wG1 /wA1;bG1 wB1\\;wA2 -wA1;bA1 bQ/;wA2 wB1/;bA1 bQ-;wG1 -wA2;bA1 bQ/;wA2 -wA1;bA1 bQ-;wB1 bG1;bS1 \\bA1;wB1 -bS1;bS1 wG1\\;wB1 bG1;bA1 bQ/;wS1 bQ\\;bA1 wS1/;wB1 bS1;bB2 /bB1;wB1 bG1;bB2 wQ\\;wS1 /bB2;bA1 bQ/;wS1 bQ\\;bA1 wS1/;wS1 /bB2;bA1 bQ\\;wA2 -bG2;bA1 bB2\\;wS1 bB1\\;bA1 wS1\\;wA2 -wA1;bB2 bB1;wA2 -bG2;bB2 wQ\\;wA2 /wA1;bA1 bQ-;wA2 -wA1;bA1 wS1\\;wA2 -bG2;bA1 /wS1;wA2 -wA1;bA1 bQ\\;wA2 -bG2;bA1 bQ-;wA2 -wA1;bA1 /wS1",
            {"wA2 -bA1", "wA2 bA1\\", "wA2 /bA1"},
            "pinned opponent ant",
            false
        },
        {
            "newgame Base+MLP;InProgress;Black[18];wA1;bP wA1\\;wL \\wA1;bA1 bP-;wG1 -wL;bQ bP\\;wQ \\wG1;bA2 /bP;wA2 wL/;bA1 \\wA2;wS1 wG1\\;bA2 -wQ;wS1 /bQ;bA3 bQ/;wA3 wA2\\;bA3 /wG1;wP wS1\\;bA3 wA3-;wP bQ\\;bG1 /bA2;wG2 /wS1;bG1 \\wQ;wS2 wS1\\;bG2 -bG1;wG2 bQ/;bA1 wS2\\;wM wG1\\;bG2 wQ/;wG3 wP-;bA1 bA2\\;wA2 bA3-;bG3 -bA2;wM -bG3;bL -bG1;wS2 wG2\\",
            {"bL bG2\\"},
            "winning move selected",
            true
        },
        {
            "newgame Base+MLP;InProgress;Black[17];wA1;bP wA1\\;wL \\wA1;bA1 /bP;wQ wA1/;bQ bP\\;wA2 /wL;bA1 wQ/;wL /bP;bA2 bA1/;wA2 bQ-;wL \\wA2;wA2 \\bA2;bA3 bA2\\;wA3 wL-;bM bA3/;wG1 -wQ;bM -wG1;wM \\wA3;bA3 wA3/;wM /bQ;bG1 -bM;wP -wM;bB1 bG1\\;wP /bP;bQ bQ\\;wS1 /wP;bG1 bA1\\;wA2 \\bA3;wL /wA1;wA3 wS1\\;bP wQ\\;bQ wM\\",
            {"bA3 -bA1", "bA2 -bA1"},
            "winning move selected",
            true
        },
        {
            "newgame Base+MLP;InProgress;Black[25];wA1;bP /wA1;wA2 \\wA1;bQ bP\\;wL wA1/;bA1 -bP;wQ wL/;bA1 \\wQ;wA2 bQ-;bA2 bA1/;wP wL\\;bA2 wP-;wA3 wA2\\;bS1 bA2-;wG1 wA3/;bM /bP;wA3 bQ\\;bM -wA1;wG1 /bP;bM -wL;wS1 wQ-;wG1 wA1\\;wA3 /bQ;bS1 wS1/;wG2 wA2-;bA3 \\bA1;wS2 wG2-;bM /bA1;wA1 bA2-;bP /bP;wM wA1\\;bA2 wQ\\;wA3 \\bA3;bA2 bQ\\;wG3 wS1\\;bP -wG1;wG1 /wG3;bG1 /bA3;wA3 /bQ;bG1 -bS1;wG1 wP\\;bG2 bS1/;wL -wP;bG2 wQ\\;wM -bP;bL /bA3;wA1 -bL;bA2 wA1\\;wA1 bQ\\",
            {"bL bM\\"},
            "winning move selected",
            true
        },
        {
            "newgame Base+MLP;InProgress;Black[24];wS1;bG1 wS1-;wP \\wS1;bA1 bG1\\;wS2 \\wP;bS1 bA1-;wQ wS2-;bQ /bS1;wQ -wP;bA2 bG1/;wM -wS2;bS2 bA2-;wL /wQ;bA3 bQ\\;wB1 \\wM;bB1 bS1\\;wA1 -wL;bA3 -wS1;wB2 wS2-;bP bA2/;wA2 wB1/;bM bP-;wA1 bM\\;bM \\bP;wL wS2/;bB2 bB1\\;wA2 bM-;bM /bA3;wA2 wL/;bQ -bB2;wA1 \\bP;bG2 /bA1;wA3 wA2-;bM bS2\\;wA2 -wM;bL /bQ;wA2 \\wL;bG3 bM\\;wA2 -bA3;bB2 bG3\\;wG1 wA3/;bA3 wG1\\;wG2 -wA1;bA3 bL\\;wA1 -bA1;bA2 bP-;wG1 -wS1",
            {"play bA3 /wM"},
            "winning move selected",
            true
        },
        {
            "newgame Base+MLP;InProgress;Black[11];wM;bG1 wM-;wQ \\wM;bP bG1\\;wA1 /wQ;bM bG1-;wA1 bP\\;bQ bM-;wB1 /wA1;bA1 bG1/;wB1 wA1;bA1 \\wQ;wA2 /wQ;bA1 wQ-;wA2 bA1/;bQ bM/;wB1 bP;bA2 bQ\\;wM bA2-;bQ bA2/;wA1 \\bQ",
            {"pass"},
            "pass selected",
            false
        }
    };
    
    // Run all test cases
    int passed = 0;
    int total = testCases.size();
    // int total = 1;
    
    for (size_t i = 0; i < total; ++i) {
        if (RunTestCase(testCases[i], i + 1)) {
            passed++;
        } 
    }
    
    // Summary
    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "Passed: " << passed << "/" << total << std::endl;
    std::cout << "Success Rate: " << (100.0 * passed / total) << "%" << std::endl;
    
    if (passed == total) {
        std::cout << "🎉 All tests passed!" << std::endl;
    } else {
        std::cout << "❌ Some tests failed." << std::endl;
    }

    return (passed == total) ? 0 : 1;
}