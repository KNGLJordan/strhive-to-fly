#!/bin/bash

pushd $(dirname $0)/..

rm -rf build 

mkdir -p build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=Release/
cmake --build . --config Release

popd