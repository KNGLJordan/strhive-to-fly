cmake_minimum_required(VERSION 3.16.0)

project(strivecpp VERSION 0.0.1)

option(DEBUG_LAZY_SMP "Enable Lazy SMP debug output" OFF)
add_executable(strivecpp)
target_compile_options(strivecpp PRIVATE -fsanitize=address)
target_link_options(strivecpp PRIVATE -fsanitize=address)
target_compile_definitions(strivecpp PRIVATE VERSION_STR="${PROJECT_VERSION}" $<$<BOOL:DEBUG_LAZY_SMP>:DEBUG_LAZY_SMP>)

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
    src/lazysmp.cpp
    src/lltt.cpp
)

set_property(TARGET strivecpp PROPERTY CXX_STANDARD 17)
