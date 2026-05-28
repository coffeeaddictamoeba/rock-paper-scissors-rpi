cmake_minimum_required(VERSION 3.16)
project(rock-paper-scissors LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE ROCKPAPERSCISSORS_SRC CONFIGURE_DEPENDS "src/*.cpp")
add_executable(rock-paper-scissors ${ROCKPAPERSCISSORS_SRC})

target_include_directories(rock-paper-scissors PRIVATE
    include
)