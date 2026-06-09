cmake_minimum_required(VERSION 3.25.2)
project(mochila_ga_cuda LANGUAGES CXX CUDA)

# ─── Opciones generales ───────────────────────────────────────────────────────
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Estándar C++
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Estándar CUDA 
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Arquitecturas GPU objetivo
# 52 → Maxwell, 60/61 → Pascal, 75 → Turing, 86/89 → Ampere/Ada
# Quitar las que no se necesiten para reducir el tiempo de compilación
set(CMAKE_CUDA_ARCHITECTURES 52 60 61 75 86 89)

# ─── Directorio de salida para resultados del programa ────────────────────────
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/../save)

# ─── Dependencias del sistema ─────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
else()
    message(FATAL_ERROR "OpenMP not found!")
endif()

# ─── Dependencias descargadas con FetchContent ────────────────────────────────
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

# ─── Fuentes ──────────────────────────────────────────────────────────────────
# Se recolectan automáticamente todos los .cpp, .c y .cu bajo src/,
# y luego se excluye main.cpp para que solo quede en el ejecutable principal.
file(GLOB_RECURSE all_SRCS
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
    "${PROJECT_SOURCE_DIR}/src/*.c"
    "${PROJECT_SOURCE_DIR}/src/*.cu"
)
list(REMOVE_ITEM all_SRCS "${PROJECT_SOURCE_DIR}/src/main.cpp")

# ─── Ejecutable principal ─────────────────────────────────────────────────────
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/cli.cpp
    src/instance_loader.cpp
    src/fitness.cpp
    src/selection.cpp
    src/operators.cpp
    src/genetic_algorithm.cpp
    ${all_SRCS}   # incorpora los .cu y cualquier fuente futura en src/
)

# ─── Includes ─────────────────────────────────────────────────────────────────
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/_deps/fast-cpp-csv-parser-src
    ${argparse_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# ─── Enlace de librerías ──────────────────────────────────────────────────────
target_link_libraries(${PROJECT_NAME} PRIVATE
    OpenMP::OpenMP_CXX
    fmt::fmt
    CUDA::cudart
)

# ─── Propiedades CUDA ─────────────────────────────────────────────────────────
# Compilación separable: necesaria para llamadas entre archivos .cu distintos
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# -lineinfo: agrega info de línea al binario CUDA (útil para Nsight/cuda-gdb)
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -lineinfo
    >
)

# ─── Empaquetado ─────────────────────────────────────────────────────────────
include(CPack)