// ─────────────────────────────────────────────────────────────────────────────
// genetic_algorithm_operators_cuda.cu
// Variante 2 — Operadores genéticos en GPU
//
// Este archivo contiene los stubs de los kernels de operadores genéticos.
// Cada función está marcada con TODO y debe ser implementada por Cristóbal.
//
// Kernels a implementar:
//   - kernel_tournament_selection   Selección por torneo
//   - kernel_crossover              Cruzamiento de un punto
//   - kernel_mutation               Mutación bit-flip
//   - kernel_update_population      Actualización con elitismo
// ─────────────────────────────────────────────────────────────────────────────

#include "genetic_algorithm_cuda.cuh"
#include <cstdint>
#include <curand_kernel.h>

// ─────────────────────────────────────────────────────────────────────────────
// kernel_tournament_selection
//
// Para cada posición i en la nueva generación, selecciona dos padres
// distintos mediante torneo: de tournament_size candidatos aleatorios,
// el que tenga mayor fitness gana.
//
// Salida:
//   d_selected_a[i] → índice del padre A para la posición i
//   d_selected_b[i] → índice del padre B para la posición i
//
// TODO: Cristóbal implementa este kernel.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_tournament_selection(const double *__restrict__ d_fitness, int *d_selected_a, int *d_selected_b,
                                            curandState *d_rng_states, int pop_size, int tournament_size) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= pop_size)
		return;

	// TODO: Cristóbal — implementar selección por torneo.
	// Stub: seleccionar índices 0 y 1 como padres para todas las posiciones.
	// Esto produce resultados incorrectos pero permite compilar y ejecutar.
	d_selected_a[idx] = 0;
	d_selected_b[idx] = (pop_size > 1) ? 1 : 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// kernel_crossover
//
// Cruzamiento de un punto para la posición idx de la nueva generación:
//   - Con probabilidad crossover_rate: elige un punto de corte aleatorio
//     y combina los genes de parent_a (antes del punto) con los de parent_b.
//   - En caso contrario: copia directamente al padre A.
//
// Entrada:  d_population    (generación actual)
//           d_selected_a/b  (índices de padres para cada posición)
// Salida:   d_new_population
//
// TODO: Cristóbal implementa este kernel.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_crossover(const uint8_t *__restrict__ d_population, uint8_t *d_new_population,
                                 const int *__restrict__ d_selected_a, const int *__restrict__ d_selected_b,
                                 curandState *d_rng_states, int pop_size, int n_items, double crossover_rate) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= pop_size)
		return;

	// TODO: Cristóbal — implementar cruzamiento de un punto con cuRAND.
	// Stub: copiar padre A sin modificar.
	const uint8_t *parent_a = d_population + static_cast<size_t>(d_selected_a[idx]) * n_items;
	uint8_t *child = d_new_population + static_cast<size_t>(idx) * n_items;
	for (int i = 0; i < n_items; ++i) {
		child[i] = parent_a[i];
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// kernel_mutation
//
// Mutación bit-flip sobre d_new_population:
//   - Con probabilidad mutation_rate por individuo, invierte un gen aleatorio.
//
// TODO: Cristóbal implementa este kernel.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_mutation(uint8_t *d_new_population, curandState *d_rng_states, int pop_size, int n_items,
                                double mutation_rate) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= pop_size)
		return;

	// TODO: Cristóbal — implementar mutación bit-flip con cuRAND.
	// Stub: no mutar nada.
	(void)d_new_population;
	(void)d_rng_states;
	(void)mutation_rate;
	(void)n_items;
}

// ─────────────────────────────────────────────────────────────────────────────
// kernel_update_population
//
// Reemplaza d_population por d_new_population respetando elitismo:
//   - Las primeras elitism_count posiciones de d_population reciben los
//     individuos de índices d_elite_indices[0..elitism_count-1].
//   - El resto se copia directamente de d_new_population.
//
// TODO: Cristóbal implementa este kernel.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_update_population(uint8_t *d_population, const uint8_t *__restrict__ d_new_population,
                                         const int *__restrict__ d_elite_indices, int elitism_count, int pop_size,
                                         int n_items) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= pop_size)
		return;

	// TODO: Cristóbal — implementar actualización con elitismo.
	// Stub: copiar d_new_population a d_population sin elitismo.
	uint8_t *dst = d_population + static_cast<size_t>(idx) * n_items;
	const uint8_t *src = d_new_population + static_cast<size_t>(idx) * n_items;
	for (int i = 0; i < n_items; ++i) {
		dst[i] = src[i];
	}
	(void)d_elite_indices;
	(void)elitism_count;
}
