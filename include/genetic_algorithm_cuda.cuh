#pragma once

// Este header solo debe incluirse desde archivos .cu (compilados por nvcc).
// Para incluir desde .cpp, usar genetic_algorithm_cuda_interface.hpp.

#include "models.hpp"
#include <cstdint>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// ─── Par fitness+índice para búsqueda atómica del mejor ─────────────────────
// Packeado en 64 bits para atomicCAS: idx (int, 4B) + fitness (float, 4B).
// La precisión float es suficiente para comparar máximos (fitness en [0,1]).
// Se alinea naturalmente a 8 bytes.
struct BestPair {
	int idx;
	float fitness;
};

// ─── Penalizaciones empaquetadas para pasar al kernel ────────────────────────
// Se pasa como parámetro por valor
struct GpuPenalties {
	double alpha;      // α: peso del valor normalizado      (α + β = 1)
	double beta;       // β: peso de la violación normalizada
	double w_weight;   // w1: exceso de peso                 (Σwi = 1)
	double w_volume;   // w2: exceso de volumen
	double w_category; // w3: violaciones de categoría
	double w_incompat; // w4: incompatibilidades
	double w_dep;      // w5: dependencias incumplidas
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
	double total_weight;       // suma de pesos de todos los items (denominador de normalizacion)
	double total_volume;       // suma de volumenes de todos los items (denominador de normalizacion)
	double max_possible_value; // suma de valores de todos los items (denominador de normalizacion)
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
	int *d_selected_a;         // [pop_size] índices de padres A (rank selection)
	int *d_selected_b;         // [pop_size] índices de padres B (rank selection)
	
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
// HELPERS DE HOST COMPARTIDOS ENTRE V2 Y V3
// Declarados aquí para ser accesibles desde genetic_algorithm_cuda_opt.cu.
// ─────────────────────────────────────────────────────────────────────────────

// Aplana la instancia y la transfiere al device (única vez antes del loop).
GpuInstance upload_instance(const ProblemInstance &inst, const PenaltyConfig &pen);

// Libera la memoria de device asociada a un GpuInstance.
void free_gpu_instance(GpuInstance &gi);

// Reserva todos los buffers del GA en device (sin d_block_best_*, que V3 gestiona aparte).
GACudaContext alloc_cuda_context(int pop_size, int n_items, int block_size);

// Libera los buffers del contexto (excepto d_block_best_*, que V3 libera antes de llamar aquí).
void free_cuda_context(GACudaContext &ctx);

// ─────────────────────────────────────────────────────────────────────────────
// KERNELS DE INFRAESTRUCTURA BASE
// ─────────────────────────────────────────────────────────────────────────────

// Inicializa los estados cuRAND.
// Un hilo por individuo; sequence = thread_id garantiza independencia estadística.
__global__ void kernel_init_rng(curandState *d_rng_states, int pop_size, unsigned long long seed);

// Evalúa el fitness de todos los individuos.
// Estrategia: 1 hilo = 1 individuo.
// Cada hilo itera sobre n_items genes y acumula:
//   - valor/peso/volumen totales
//   - excesos de capacidad → restricciones duras
//   - violaciones de categoría → restricción blanda
//   - violaciones de incompatibilidad → restricción dura
//   - violaciones de dependencia → restricción dura
// y calcula el fitness normalizado: fitness = alpha*valor_norm - beta*violacion_norm
__global__ void kernel_evaluate_fitness(const uint8_t *__restrict__ d_population, const double *__restrict__ d_values,
                                        const double *__restrict__ d_weights, const double *__restrict__ d_volumes,
                                        const int *__restrict__ d_item_cat, const int *__restrict__ d_cat_min,
                                        const int *__restrict__ d_cat_max, int n_cats,
                                        const int *__restrict__ d_incomp_a, const int *__restrict__ d_incomp_b,
                                        int n_incomp, const int *__restrict__ d_dep_item,
                                        const int *__restrict__ d_dep_req, int n_dep, double max_weight,
                                        double max_volume, double total_weight, double total_volume,
                                        double max_possible_value, GpuPenalties penalties, double *d_fitness,
                                        uint8_t *d_feasible, int pop_size, int n_items);

// ─────────────────────────────────────────────────────────────────────────────
// KERNELS DE OPERADORES GENÉTICOS
// Declarados aquí para que el loop principal pueda invocarlos.
// ─────────────────────────────────────────────────────────────────────────────

// Selección por ranking geométrico: para cada posición en la nueva generación,
// elige dos padres según distribución de probabilidades geométrica sobre los
// mejores R individuos (ordenados por fitness descendente).
// d_sorted_indices[0..R-1] contiene los índices de los R mejores.
// d_cum_probs contiene las probabilidades acumuladas normalizadas [0..R-1].
__global__ void kernel_rank_selection(const int *__restrict__ d_sorted_indices, int *d_selected_a, int *d_selected_b,
                                      curandState *d_rng_states, int pop_size, int rank_count,
                                      const double *__restrict__ d_cum_probs);

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

// Búsqueda paralela simple del mejor individuo (global y factible) usando
// atomicCAS sobre BestPair (64 bits). Sin shared memory, sin reducción en árbol
// — apropiado para V2. Cada hilo compite atómicamente por actualizar el mejor.
__global__ void kernel_find_best_simple(const double *__restrict__ d_fitness, const uint8_t *__restrict__ d_feasible,
                                        int pop_size, unsigned long long *d_best_global,
                                        unsigned long long *d_best_feas);

// FUNCIÓN DE ENTRADA — Variante 2: CUDA básico
// Declarada aquí e implementada en src/genetic_algorithm_cuda.cu
GARunResult run_genetic_algorithm_cuda_basic(const ProblemInstance &instance, const GAConfig &config, int seed,
                                             int block_size);

// FUNCIÓN DE ENTRADA — Variante 3: CUDA optimizado
// Implementada en src/genetic_algorithm_cuda_opt.cu.
// Optimizaciones respecto a V2:
//   1. Memoria constante para escalares (penalizaciones, límites, n_items, etc.)
//   2. Conteo de categorías en un solo pase O(n_items) (vs O(n_cats × n_items) en V2)
//   3. Reducción con shared memory para búsqueda del mejor (kernel_find_best_feasible)
//   4. Elitismo GPU-side: usa d_sorted_indices de CUB sort → elimina transfer pop_size × 8B
//   5. Reducción de divergencia de warps en cruzamiento y mutación
GARunResult run_genetic_algorithm_cuda_optimized(const ProblemInstance &instance, const GAConfig &config, int seed,
                                                 int block_size);
