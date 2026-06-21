#pragma once

// Header C++-compatible (sin tipos CUDA) que expone la interfaz de las
// Variantes 2 y 3. Se incluye desde archivos .cpp (main.cpp, cli.cpp, etc.).
// Los tipos CUDA completos están en genetic_algorithm_cuda.cuh.

#include "models.hpp"

// Variante 2: CUDA básico
GARunResult run_genetic_algorithm_cuda_basic(const ProblemInstance &instance, const GAConfig &config, int seed,
                                             int block_size);

// Variante 3: CUDA optimizado
// Optimizaciones: memoria constante, pase único O(n_items) para categorías,
// reducción con shared memory, elitismo GPU-side, sin divergencia de warps.
GARunResult run_genetic_algorithm_cuda_optimized(const ProblemInstance &instance, const GAConfig &config, int seed,
                                                 int block_size);
