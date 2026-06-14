// ─────────────────────────────────────────────────────────────────────────────
// genetic_algorithm_operators_cuda.cu
// Variante 2 — Operadores genéticos en GPU
//
// Implementación de los kernels de operadores genéticos:
//   - kernel_tournament_selection   Selección por torneo
//   - kernel_crossover              Cruzamiento de un punto
//   - kernel_mutation                Mutación bit-flip
//   - kernel_preserve_elites        Preservación de élites
//   - kernel_update_population      Actualización de la población
//   - kernel_find_best_feasible     Reducción: mejor individuo / mejor factible
// ─────────────────────────────────────────────────────────────────────────────

#include "genetic_algorithm_cuda.cuh"
#include <cfloat>
#include <cstdint>
#include <curand_kernel.h>

// ─────────────────────────────────────────────────────────────────────────────
// kernel_tournament_selection
//
// Para cada posición i en la nueva generación, selecciona dos padres mediante
// torneo: de tournament_size candidatos aleatorios, gana el de mayor fitness.
// Los torneos para el padre A y el padre B son independientes entre sí, por lo
// que ambos padres pueden coincidir (igual que en la versión secuencial).
//
// Salida:
//   d_selected_a[i] → índice del padre A para la posición i
//   d_selected_b[i] → índice del padre B para la posición i
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_tournament_selection(const double *__restrict__ d_fitness, int *d_selected_a, int *d_selected_b,
                                            curandState *d_rng_states, int pop_size, int tournament_size) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= pop_size)
		return;

	curandState local_state = d_rng_states[idx];

	// Torneo para el padre A
	int best_a = curand(&local_state) % pop_size;
	double best_a_fit = d_fitness[best_a];
	for (int t = 1; t < tournament_size; ++t) {
		int candidate = curand(&local_state) % pop_size;
		double cand_fit = d_fitness[candidate];
		if (cand_fit > best_a_fit) {
			best_a_fit = cand_fit;
			best_a = candidate;
		}
	}

	// Torneo para el padre B (independiente del de A)
	int best_b = curand(&local_state) % pop_size;
	double best_b_fit = d_fitness[best_b];
	for (int t = 1; t < tournament_size; ++t) {
		int candidate = curand(&local_state) % pop_size;
		double cand_fit = d_fitness[candidate];
		if (cand_fit > best_b_fit) {
			best_b_fit = cand_fit;
			best_b = candidate;
		}
	}

	d_selected_a[idx] = best_a;
	d_selected_b[idx] = best_b;

	d_rng_states[idx] = local_state;
}

// ─────────────────────────────────────────────────────────────────────────────
// kernel_crossover
//
// Cruzamiento de un punto para la posición idx de la nueva generación:
//   - Con probabilidad crossover_rate: elige un punto de corte aleatorio
//     en [1, n_items-1] y combina los genes de parent_a (antes del punto)
//     con los de parent_b (desde el punto en adelante).
//   - En caso contrario (o si n_items < 2): copia directamente al padre A.
//
// Entrada:  d_population    (generación actual)
//           d_selected_a/b  (índices de padres para cada posición)
// Salida:   d_new_population
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_crossover(const uint8_t *__restrict__ d_population, uint8_t *d_new_population,
                                 const int *__restrict__ d_selected_a, const int *__restrict__ d_selected_b,
                                 curandState *d_rng_states, int pop_size, int n_items, double crossover_rate) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= pop_size)
		return;

	const uint8_t *parent_a = d_population + static_cast<size_t>(d_selected_a[idx]) * n_items;
	const uint8_t *parent_b = d_population + static_cast<size_t>(d_selected_b[idx]) * n_items;
	uint8_t *child = d_new_population + static_cast<size_t>(idx) * n_items;

	curandState local_state = d_rng_states[idx];
	double coin = curand_uniform_double(&local_state);

	if (coin > crossover_rate || n_items < 2) {
		// Sin cruzamiento: copiar directamente al padre A.
		for (int i = 0; i < n_items; ++i) {
			child[i] = parent_a[i];
		}
	} else {
		// Punto de corte aleatorio en [1, n_items - 1]
		int point = 1 + static_cast<int>(curand(&local_state) % static_cast<unsigned int>(n_items - 1));
		for (int i = 0; i < n_items; ++i) {
			child[i] = (i < point) ? parent_a[i] : parent_b[i];
		}
	}

	d_rng_states[idx] = local_state;
}

// ─────────────────────────────────────────────────────────────────────────────
// kernel_mutation
//
// Mutación bit-flip sobre d_new_population:
//   - Con probabilidad mutation_rate por individuo, invierte un gen aleatorio.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_mutation(uint8_t *d_new_population, curandState *d_rng_states, int pop_size, int n_items,
                                double mutation_rate) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= pop_size)
		return;

	curandState local_state = d_rng_states[idx];
	double coin = curand_uniform_double(&local_state);

	if (coin <= mutation_rate) {
		uint8_t *chromosome = d_new_population + static_cast<size_t>(idx) * n_items;
		int gene = static_cast<int>(curand(&local_state) % static_cast<unsigned int>(n_items));
		chromosome[gene] ^= 1U;
	}

	d_rng_states[idx] = local_state;
}

// ─────────────────────────────────────────────────────────────────────────────
// kernel_preserve_elites
//
// Copia los elitism_count mejores individuos de d_population (generación
// actual) hacia las primeras posiciones de d_new_population:
//
//   d_new_population[i] = d_population[d_elite_indices[i]]   para i < elitism_count
//
// Debe ejecutarse DESPUÉS de cruzamiento/mutación (que escriben sobre
// d_new_population) y ANTES de kernel_update_population, evitando así la
// condición de carrera de leer d_population mientras otro hilo ya lo
// sobrescribió.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_preserve_elites(const uint8_t *__restrict__ d_population, uint8_t *__restrict__ d_new_population,
                                       const int *__restrict__ d_elite_indices, int elitism_count, int n_items) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= elitism_count)
		return;

	const uint8_t *src = d_population + static_cast<size_t>(d_elite_indices[idx]) * n_items;
	uint8_t *dst = d_new_population + static_cast<size_t>(idx) * n_items;

	for (int i = 0; i < n_items; ++i) {
		dst[i] = src[i];
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// kernel_update_population
//
// Reemplaza d_population por d_new_population (copia directa por individuo).
// El elitismo ya fue resuelto previamente por kernel_preserve_elites, que
// dejó los mejores individuos al inicio de d_new_population.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_update_population(uint8_t *__restrict__ d_population,
                                         const uint8_t *__restrict__ d_new_population, int pop_size, int n_items) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= pop_size)
		return;

	uint8_t *dst = d_population + static_cast<size_t>(idx) * n_items;
	const uint8_t *src = d_new_population + static_cast<size_t>(idx) * n_items;
	for (int i = 0; i < n_items; ++i) {
		dst[i] = src[i];
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// kernel_find_best_feasible
//
// Reducción en árbol a nivel de bloque (memoria compartida) que calcula, para
// cada bloque, dos resultados:
//   - el mejor individuo del bloque (cualquier factibilidad): índice + fitness
//   - el mejor individuo factible del bloque: índice (-1 si ninguno) + fitness
//
// La reducción final entre bloques (sobre los grid_pop resultados parciales)
// se realiza en el host, ya que grid_pop es pequeño.
//
// Layout de memoria compartida (en bytes, tamaño total =
// blockDim.x * (2*sizeof(double) + 2*sizeof(int))):
//   [0 .. blockDim.x)                doubles  -> s_fit       (mejor fitness global)
//   [blockDim.x .. 2*blockDim.x)     doubles  -> s_fit_feas  (mejor fitness factible)
//   [2*blockDim.x .. 3*blockDim.x)   ints     -> s_idx       (índice del mejor global)
//   [3*blockDim.x .. 4*blockDim.x)   ints     -> s_idx_feas  (índice del mejor factible)
//
// NOTA: se asume blockDim.x potencia de 2 (cumplido por DEFAULT_BLOCK_SIZE=256
// y por los tamaños de bloque usados en los experimentos de la actividad).
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_find_best_feasible(const double *__restrict__ d_fitness, const uint8_t *__restrict__ d_feasible,
                                          int pop_size, int *d_block_best_idx, double *d_block_best_fitness,
                                          int *d_block_best_feasible_idx, double *d_block_best_feasible_fitness) {

	extern __shared__ unsigned char s_mem[];

	double *s_fit = reinterpret_cast<double *>(s_mem);
	double *s_fit_feas = s_fit + blockDim.x;
	int *s_idx = reinterpret_cast<int *>(s_fit_feas + blockDim.x);
	int *s_idx_feas = s_idx + blockDim.x;

	int tid = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + tid;

	if (idx < pop_size) {
		s_fit[tid] = d_fitness[idx];
		s_idx[tid] = idx;

		if (d_feasible[idx] == 1U) {
			s_fit_feas[tid] = d_fitness[idx];
			s_idx_feas[tid] = idx;
		} else {
			s_fit_feas[tid] = -DBL_MAX;
			s_idx_feas[tid] = -1;
		}
	} else {
		s_fit[tid] = -DBL_MAX;
		s_idx[tid] = -1;
		s_fit_feas[tid] = -DBL_MAX;
		s_idx_feas[tid] = -1;
	}

	__syncthreads();

	// Reducción en árbol (asume blockDim.x potencia de 2)
	for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (tid < stride) {
			if (s_fit[tid + stride] > s_fit[tid]) {
				s_fit[tid] = s_fit[tid + stride];
				s_idx[tid] = s_idx[tid + stride];
			}
			if (s_fit_feas[tid + stride] > s_fit_feas[tid]) {
				s_fit_feas[tid] = s_fit_feas[tid + stride];
				s_idx_feas[tid] = s_idx_feas[tid + stride];
			}
		}
		__syncthreads();
	}

	if (tid == 0) {
		d_block_best_idx[blockIdx.x] = s_idx[0];
		d_block_best_fitness[blockIdx.x] = s_fit[0];
		d_block_best_feasible_idx[blockIdx.x] = s_idx_feas[0];
		d_block_best_feasible_fitness[blockIdx.x] = s_fit_feas[0];
	}
}
