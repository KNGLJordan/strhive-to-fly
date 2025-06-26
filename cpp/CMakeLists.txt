cmake_minimum_required(VERSION 3.16.0)

project(strivecpp VERSION 0.0.1)

add_executable(strivecpp)

target_compile_definitions(strivecpp PRIVATE VERSION_STR="${PROJECT_VERSION}")

target_include_directories(strivecpp PRIVATE include)

target_sources(strivecpp PRIVATE
    src/main.cpp
    src/Board.cpp
    src/Engine.cpp
    src/Enums.cpp
    src/MinMax.cpp
    src/MinMaxZobrist.cpp
    src/Move.cpp
    src/Position.cpp
    src/ZobristHasher.cpp
)

set_property(TARGET strivecpp PROPERTY CXX_STANDARD 17)
