cmake_minimum_required(VERSION 3.24)
project(mochila_ga LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
else()
    message(FATAL_ERROR "OpenMP not found!")
endif()

include(FetchContent)

FetchContent_Declare(
    fast-cpp-csv-parser
    GIT_REPOSITORY https://github.com/ben-strasser/fast-cpp-csv-parser.git
    GIT_TAG master
    GIT_SHALLOW TRUE
)

FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
    GIT_TAG v3.1
    GIT_SHALLOW TRUE
)

FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 11.2.0
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(fast-cpp-csv-parser argparse fmt)

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/cli.cpp
    src/instance_loader.cpp
    src/fitness.cpp
    src/selection.cpp
    src/operators.cpp
    src/genetic_algorithm.cpp
    src/parallel_genetic_algorithm.cpp
    src/island_model.cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(${PROJECT_NAME} PRIVATE fmt::fmt)
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/_deps/fast-cpp-csv-parser-src
    ${argparse_SOURCE_DIR}/include
)

# ── Tests ────────────────────────────────────────────────────────────────────
enable_testing()

add_executable(test_fitness
    tests/test_fitness.cpp
    src/fitness.cpp
)

target_include_directories(test_fitness PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/_deps/fast-cpp-csv-parser-src
    ${argparse_SOURCE_DIR}/include
)

add_test(NAME fitness_unit COMMAND test_fitness)

add_test(
    NAME ga_integration
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_integration.sh
            $<TARGET_FILE:${PROJECT_NAME}>
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(ga_integration PROPERTIES DEPENDS ${PROJECT_NAME})
