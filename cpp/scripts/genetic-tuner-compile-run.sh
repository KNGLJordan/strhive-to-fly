#!/bin/bash

pushd $(dirname $0)/..

g++ -std=c++17 -O2 -o genetic_tuner src/genetic_tuner.cpp src/Board.cpp src/MinMax.cpp src/Enums.cpp src/Move.cpp src/Position.cpp -Iinclude

./genetic_tuner