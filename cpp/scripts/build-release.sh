#!/bin/bash

pushd $(dirname $0)/..

mkdir -p build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=Release/ -DDEBUG_LAZY_SMP=OFF
cmake --build . --config Release

popd