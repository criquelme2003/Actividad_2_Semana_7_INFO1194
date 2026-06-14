#pragma once

// Header C++-compatible (sin tipos CUDA) que expone la interfaz de la
// Variante 2. Se incluye desde archivos .cpp (main.cpp, cli.cpp, etc.).
// Los tipos CUDA completos están en genetic_algorithm_cuda.cuh.

#include "models.hpp"

GARunResult run_genetic_algorithm_cuda_basic(const ProblemInstance &instance, const GAConfig &config, int seed,
                                             int block_size);
