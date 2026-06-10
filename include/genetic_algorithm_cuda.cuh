#pragma once

// Este header solo debe incluirse desde archivos .cu (compilados por nvcc).
// Para incluir desde .cpp, usar genetic_algorithm_cuda_interface.hpp.

#include "models.hpp"
#include <cstdint>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// ─── Penalizaciones empaquetadas para pasar al kernel ────────────────────────
// Se pasa como parámetro por valor
struct GpuPenalties {
	double alpha;
	double beta;
	double gamma;
	double delta;
	double epsilon;
};

// ─── Instancia del problema en device ────────────────────────────────────────
// Todos los punteros apuntan a memoria de GPU (cudaMalloc).
// Se construye una vez antes del loop y se libera al final.
struct GpuInstance {
	// Propiedades de ítems (indexados por posición 0..n_items-1)
	double *d_values;  // [n_items]  valor de cada ítem
	double *d_weights; // [n_items]  peso de cada ítem
	double *d_volumes; // [n_items]  volumen de cada ítem
	int *d_item_cat;   // [n_items]  índice de categoría (0-based)

	// Reglas por categoría (indexadas por índice de categoría)
	int *d_cat_min; // [n_cats]   mínimo de ítems requeridos
	int *d_cat_max; // [n_cats]   máximo permitido (-1 = sin límite)
	int n_cats;

	// Incompatibilidades: pares de índices de ítems que no pueden coexistir
	int *d_incomp_a; // [n_incomp]
	int *d_incomp_b; // [n_incomp]
	int n_incomp;

	// Dependencias: si genes[dep_item]==1 entonces genes[dep_req] debe ser 1
	int *d_dep_item; // [n_dep]
	int *d_dep_req;  // [n_dep]
	int n_dep;

	double max_weight;
	double max_volume;
	int n_items;
};

// ─── Contexto del GA en device ────────────────────────────────────────────────
// Agrupa todos los buffers de device del algoritmo genético.
// La representación de la población es row-major:
//
//   genes[individuo_id * n_items + item_id]
//
// Esto coloca los genes de un mismo individuo contiguos en memoria,
// ideal para el kernel de fitness donde un hilo procesa un individuo completo.
struct GACudaContext {
	uint8_t *d_population;     // [pop_size * n_items]  generación actual
	uint8_t *d_new_population; // [pop_size * n_items]  próxima generación
	double *d_fitness;         // [pop_size]
	uint8_t *d_feasible;       // [pop_size] 1=factible respecto duras, 0=no
	curandState *d_rng_states; // [pop_size] un estado cuRAND por individuo
	int *d_selected_a;         // [pop_size] índices de padres A (torneo)
	int *d_selected_b;         // [pop_size] índices de padres B (torneo)
	
	// Resultados parciales (1 valor por bloque) de la reducción
	// "mejor individuo" / "mejor individuo factible".
	int *d_block_best_idx;                  // [grid_pop]
	double *d_block_best_fitness;           // [grid_pop]
	int *d_block_best_feasible_idx;         // [grid_pop]
	double *d_block_best_feasible_fitness;  // [grid_pop]
	int grid_pop;                           // = ceil(pop_size / block_size)

	int pop_size;
	int n_items;
	int block_size;
};

// ─────────────────────────────────────────────────────────────────────────────
// KERNELS DE INFRAESTRUCTURA BASE
// ─────────────────────────────────────────────────────────────────────────────

// Inicializa los estados cuRAND.
// Un hilo por individuo; sequence = thread_id garantiza independencia estadística.
__global__ void kernel_init_rng(curandState *d_rng_states, int pop_size, unsigned long long seed);

// Genera la población inicial aleatoriamente (50 % de probabilidad por gen).
// Debe llamarse después de kernel_init_rng.
__global__ void kernel_init_population(uint8_t *d_population, int pop_size, int n_items, curandState *d_rng_states);

// Evalúa el fitness de todos los individuos.
// Estrategia: 1 hilo = 1 individuo.
// Cada hilo itera sobre n_items genes y acumula:
//   - valor/peso/volumen totales
//   - excesos de capacidad → restricciones duras (alpha, beta)
//   - violaciones de categoría → restricción blanda (gamma)
//   - violaciones de incompatibilidad → restricción dura (delta)
//   - violaciones de dependencia → restricción dura (epsilon)
__global__ void kernel_evaluate_fitness(const uint8_t *__restrict__ d_population, const double *__restrict__ d_values,
                                        const double *__restrict__ d_weights, const double *__restrict__ d_volumes,
                                        const int *__restrict__ d_item_cat, const int *__restrict__ d_cat_min,
                                        const int *__restrict__ d_cat_max, int n_cats,
                                        const int *__restrict__ d_incomp_a, const int *__restrict__ d_incomp_b,
                                        int n_incomp, const int *__restrict__ d_dep_item,
                                        const int *__restrict__ d_dep_req, int n_dep, double max_weight,
                                        double max_volume, GpuPenalties penalties, double *d_fitness,
                                        uint8_t *d_feasible, int pop_size, int n_items);

// ─────────────────────────────────────────────────────────────────────────────
// KERNELS DE OPERADORES GENÉTICOS
// Declarados aquí para que el loop principal pueda invocarlos.
// ─────────────────────────────────────────────────────────────────────────────

// Selección por torneo: para cada posición en la nueva generación, elige
// dos padres seleccionando el mejor de `tournament_size` candidatos aleatorios.
// Escribe los índices ganadores en d_selected_a y d_selected_b.
__global__ void kernel_tournament_selection(const double *__restrict__ d_fitness, int *d_selected_a, int *d_selected_b,
                                            curandState *d_rng_states, int pop_size, int tournament_size);

// Cruzamiento de un punto: genera d_new_population a partir de los pares
// (d_selected_a[i], d_selected_b[i]) de d_population.
__global__ void kernel_crossover(const uint8_t *__restrict__ d_population, uint8_t *d_new_population,
                                 const int *__restrict__ d_selected_a, const int *__restrict__ d_selected_b,
                                 curandState *d_rng_states, int pop_size, int n_items, double crossover_rate);

// Mutación bit-flip: invierte un gen aleatorio con probabilidad mutation_rate.
__global__ void kernel_mutation(uint8_t *d_new_population, curandState *d_rng_states, int pop_size, int n_items,
                                double mutation_rate);

// Preserva los elitism_count mejores individuos: copia
// d_population[d_elite_indices[i]] -> d_new_population[i] para i < elitism_count.
// Debe ejecutarse después de cruzamiento/mutación y antes de kernel_update_population.
__global__ void kernel_preserve_elites(const uint8_t *__restrict__ d_population, uint8_t *__restrict__ d_new_population,
                                       const int *__restrict__ d_elite_indices, int elitism_count, int n_items);

// Actualización de la generación: copia d_new_population sobre d_population.
__global__ void kernel_update_population(uint8_t *__restrict__ d_population,
                                         const uint8_t *__restrict__ d_new_population, int pop_size, int n_items);

// Reducción paralela (a nivel de bloque) que calcula, para cada bloque:
//   - el índice y fitness del mejor individuo (cualquier factibilidad)
//   - el índice y fitness del mejor individuo factible (-1 si ninguno)
// Resultados parciales (1 por bloque) en los arreglos d_block_best_*.
// Asume blockDim.x potencia de 2.
__global__ void kernel_find_best_feasible(const double *__restrict__ d_fitness, const uint8_t *__restrict__ d_feasible,
                                          int pop_size, int *d_block_best_idx, double *d_block_best_fitness,
                                          int *d_block_best_feasible_idx, double *d_block_best_feasible_fitness);

// FUNCIÓN DE ENTRADA — Variante 2: CUDA básico
// Declarada aquí e implementada en src/genetic_algorithm_cuda.cu
GARunResult run_genetic_algorithm_cuda_basic(const ProblemInstance &instance, const GAConfig &config, int seed,
                                             int block_size);
