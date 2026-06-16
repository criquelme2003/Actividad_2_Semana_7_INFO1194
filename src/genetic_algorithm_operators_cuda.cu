// ─────────────────────────────────────────────────────────────────────────────
// genetic_algorithm_operators_cuda.cu
// Variante 2 — Operadores genéticos en GPU
//
// Implementación de los kernels de operadores genéticos:
//   - kernel_rank_selection         Selección por ranking geométrico
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
// kernel_rank_selection
//
// Para cada posición i en la nueva generación, selecciona dos padres mediante
// ranking geométrico: elige un rank aleatorio (0..rank_count-1) de acuerdo a
// la distribución de probabilidades acumuladas d_cum_probs, y luego obtiene el
// índice del individuo correspondiente desde d_sorted_indices[rank].
//
// Los rankings de ambos padres se generan con el mismo generador pseudoaleatorio
// (secuencial, no paralelo) y se asegura que sean distintos: si rank_b == rank_a,
// se sigue sorteando hasta obtener uno diferente (misma semántica que V1).
//
// d_sorted_indices debe haber sido generado por cub::DeviceRadixSort::SortPairsDescending
// sobre los fitness en la generación actual.
//
// Salida:
//   d_selected_a[i] → índice del padre A para la posición i
//   d_selected_b[i] → índice del padre B para la posición i
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_rank_selection(const int *__restrict__ d_sorted_indices, int *d_selected_a, int *d_selected_b,
                                      curandState *d_rng_states, int pop_size, int rank_count,
                                      const double *__restrict__ d_cum_probs) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= pop_size)
		return;

	curandState local_state = d_rng_states[idx];

	// --- Selección del padre A (rank aleatorio según prob. acumulada) ---
	double u_a = curand_uniform_double(&local_state);
	int rank_a = 0;
	for (int r = 0; r < rank_count; ++r) {
		if (u_a < d_cum_probs[r]) {
			rank_a = r;
			break;
		}
		rank_a = r;
	}

	// --- Selección del padre B (distinto de A) ---
	int rank_b;
	do {
		double u_b = curand_uniform_double(&local_state);
		rank_b = 0;
		for (int r = 0; r < rank_count; ++r) {
			if (u_b < d_cum_probs[r]) {
				rank_b = r;
				break;
			}
			rank_b = r;
		}
	} while (rank_b == rank_a);

	d_selected_a[idx] = d_sorted_indices[rank_a];
	d_selected_b[idx] = d_sorted_indices[rank_b];

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
// kernel_find_best_simple
//
// Búsqueda paralela del mejor individuo (global y factible) mediante atomicCAS
// sobre BestPair (64 bits: índice + fitness en float). Sin shared memory, sin
// reducción en árbol — cada hilo compite atómicamente por ser el mejor.
//
// Adecuado para V2 (CUDA básico). V3 usará kernel_find_best_feasible con
// reducción paralela en shared memory.
//
// d_best_global y d_best_feas deben inicializarse con BestPair{-1, -FLT_MAX}
// ANTES de lanzar este kernel (se hace en el host con cudaMemcpy cada gen.).
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_find_best_simple(const double *__restrict__ d_fitness, const uint8_t *__restrict__ d_feasible,
                                        int pop_size, unsigned long long *d_best_global,
                                        unsigned long long *d_best_feas) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= pop_size)
		return;

	float f = static_cast<float>(d_fitness[idx]);

	BestPair local;
	local.idx = idx;
	local.fitness = f;
	unsigned long long local_val = *reinterpret_cast<unsigned long long *>(&local);

	// Competir atómicamente por el mejor global
	unsigned long long old_val = *d_best_global;
	BestPair old = *reinterpret_cast<BestPair *>(&old_val);

	while (f > old.fitness || (f == old.fitness && idx < old.idx)) {
		unsigned long long expected = old_val;
		unsigned long long prev = atomicCAS(d_best_global, expected, local_val);
		if (prev == expected)
			break;
		old_val = prev;
		old = *reinterpret_cast<BestPair *>(&old_val);
	}

	// Competir atómicamente por el mejor factible
	if (d_feasible[idx]) {
		old_val = *d_best_feas;
		old = *reinterpret_cast<BestPair *>(&old_val);

		while (f > old.fitness || (f == old.fitness && idx < old.idx)) {
			unsigned long long expected = old_val;
			unsigned long long prev = atomicCAS(d_best_feas, expected, local_val);
			if (prev == expected)
				break;
			old_val = prev;
			old = *reinterpret_cast<BestPair *>(&old_val);
		}
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
