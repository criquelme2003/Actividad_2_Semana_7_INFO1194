// genetic_algorithm_cuda_opt.cu — Variante 3: CUDA optimizado
//
// Optimizaciones implementadas respecto a V2 (CUDA básico):
//
//  1. Memoria constante (__constant__): penalizaciones y límites escalares se
//     almacenan en el caché L1 de constantes. Todos los hilos leen los mismos
//     valores → acceso broadcast, sin conflictos de banco, sin tráfico a DRAM.
//
//  2. Conteo de categorías en un solo pase O(n_items): V2 usaba un bucle anidado
//     O(n_cats × n_items). V3 acumula conteos en un arreglo local durante el
//     mismo recorrido de ítems. Para n_cats=20 y n_items=10000: 20× menos iteraciones.
//
//  3. Reducción con shared memory para búsqueda del mejor individuo:
//     kernel_find_best_feasible (árbol de reducción en shared memory) reemplaza
//     kernel_find_best_simple (atomicCAS sin shared memory de V2). Solo se
//     transfieren grid_pop resultados parciales (vs ningún registro extra en V2 más
//     el transfer completo de h_fitness).
//
//  4. Elitismo GPU-side (eliminación de transfer pop_size × 8 bytes por generación):
//     V2 transfería d_fitness al host para hacer partial_sort en CPU y luego
//     resubía los índices de élite. V3 usa los primeros elitism_count elementos de
//     d_sorted_indices (salida de CUB sort, ya en GPU) directamente en
//     kernel_preserve_elites → cero transferencias extra por generación.
//
//  5. Reducción de divergencia de warps en cruzamiento y mutación:
//     kernel_crossover_v3: el bucle de copia no tiene ramas — el punto de corte
//       se fija en n_items cuando no hay cruzamiento (toda la copia viene del
//       padre A), eliminando el if/else que diverge hilos del mismo warp.
//     kernel_mutation_v3: el XOR se aplica siempre (con 0 cuando no muta) →
//       sin branch, sin divergencia.

#include "constants.hpp"
#include "fitness.hpp"
#include "genetic_algorithm_cuda.cuh"
#include "operators.hpp"

#include <cub/cub.cuh>

#include <algorithm>
#include <cfloat>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using std::size_t;
using std::vector;

#define CUDA_CHECK(call)                                                                                               \
	do {                                                                                                               \
		cudaError_t _err = (call);                                                                                     \
		if (_err != cudaSuccess) {                                                                                     \
			throw std::runtime_error(std::string("[CUDA V3] ") + __FILE__ + ":" + std::to_string(__LINE__) +           \
			                         " → " + cudaGetErrorString(_err));                                                \
		}                                                                                                              \
	} while (0)

// ── Optimización 1: Memoria constante ────────────────────────────────────────
// Escalares leídos uniformemente por todos los hilos de cada kernel V3.
// El hardware los sirve desde el caché de constantes (broadcast, latencia ~4 ciclos)
// en lugar de cargarlos desde registros de kernel (que ocupan espacio en el
// registro de argumentos de 256 bytes por kernel).
__constant__ GpuPenalties c_pen;
__constant__ double       c_max_w;
__constant__ double       c_max_v;
__constant__ double       c_tot_w;
__constant__ double       c_tot_v;
__constant__ double       c_max_val;
__constant__ int          c_ni;      // n_items
__constant__ int          c_nc;      // n_cats
__constant__ int          c_ninc;    // n_incomp
__constant__ int          c_ndep;    // n_dep

// Máximo de categorías para el arreglo local de conteo (Optimización 2).
// Para n_cats ≤ 32 el compilador puede mantener el arreglo en registros;
// para valores mayores puede volcarlo a local memory (L1-cacheada).
#define V3_MAX_CATS 64

// ── Optimización 1 + 2: Kernel de evaluación con memoria constante y pase único ─
//
// Diferencias clave vs kernel_evaluate_fitness (V2):
//   - No recibe los escalares como parámetros: los lee de __constant__ memory.
//   - Categorías: un solo recorrido O(n_items) con arreglo local de conteos
//     (V2: dos bucles anidados O(n_cats × n_items)).
//   - Firma reducida: 13 parámetros menos → menor presión en el banco de
//     argumentos del kernel y menor uso de registros.
__global__ void kernel_evaluate_fitness_v3(const uint8_t *__restrict__ d_population,
                                           const double *__restrict__  d_values,
                                           const double *__restrict__  d_weights,
                                           const double *__restrict__  d_volumes,
                                           const int    *__restrict__  d_item_cat,
                                           const int    *__restrict__  d_cat_min,
                                           const int    *__restrict__  d_cat_max,
                                           const int    *__restrict__  d_incomp_a,
                                           const int    *__restrict__  d_incomp_b,
                                           const int    *__restrict__  d_dep_item,
                                           const int    *__restrict__  d_dep_req,
                                           double  *d_fitness,
                                           uint8_t *d_feasible,
                                           int pop_size) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= pop_size)
		return;

	// Leer escalares desde memoria constante (broadcast L1, cero tráfico a DRAM)
	const int n_items = c_ni;
	const int n_cats  = c_nc;
	const int n_incomp = c_ninc;
	const int n_dep    = c_ndep;

	const uint8_t *genes = d_population + (size_t)idx * n_items;

	// Optimización 2: pase único sobre los ítems acumulando valor, peso,
	// volumen y conteo de categorías simultáneamente.
	// V2 hacía esto en O(n_items) + O(n_cats × n_items) → V3: O(n_items).
	double total_value  = 0.0;
	double total_weight = 0.0;
	double total_volume = 0.0;
	int    cat_counts[V3_MAX_CATS] = {};  // arreglo local inicializado a 0

	for (int i = 0; i < n_items; ++i) {
		if (genes[i] != 0U) {
			total_value  += d_values[i];
			total_weight += d_weights[i];
			total_volume += d_volumes[i];
			int c = d_item_cat[i];
			if (c < V3_MAX_CATS)
				cat_counts[c]++;
		}
	}

	// Excesos de capacidad (restricciones duras, leídos de __constant__)
	double excess_w = (total_weight > c_max_w) ? (total_weight - c_max_w) : 0.0;
	double excess_v = (total_volume > c_max_v) ? (total_volume - c_max_v) : 0.0;

	// Violaciones de categoría (bucle corto O(n_cats))
	int cat_viol = 0;
	for (int c = 0; c < n_cats; ++c) {
		int cnt   = (c < V3_MAX_CATS) ? cat_counts[c] : 0;
		int under = d_cat_min[c] - cnt;
		if (under > 0)
			cat_viol += under;
		int mx = d_cat_max[c];
		if (mx >= 0 && cnt > mx)
			cat_viol += (cnt - mx);
	}

	// Incompatibilidades (restricción dura)
	int incomp_viol = 0;
	for (int k = 0; k < n_incomp; ++k)
		if (genes[d_incomp_a[k]] != 0U && genes[d_incomp_b[k]] != 0U)
			++incomp_viol;

	// Dependencias (restricción dura)
	int dep_viol = 0;
	for (int k = 0; k < n_dep; ++k)
		if (genes[d_dep_item[k]] != 0U && genes[d_dep_req[k]] == 0U)
			++dep_viol;

	// Fitness normalizado (denominadores y penalizaciones desde __constant__)
	double max_ew  = c_tot_w - c_max_w;
	double max_ev  = c_tot_v - c_max_v;
	double max_cat = (double)n_items;
	double max_inc = (double)(n_incomp > 0 ? n_incomp : 1);
	double max_dep = (double)(n_dep    > 0 ? n_dep    : 1);

	double p1 = fmin(1.0, excess_w / max_ew);
	double p2 = fmin(1.0, excess_v / max_ev);
	double p3 = fmin(1.0, (double)cat_viol  / max_cat);
	double p4 = (n_incomp == 0) ? 0.0 : fmin(1.0, (double)incomp_viol / max_inc);
	double p5 = (n_dep    == 0) ? 0.0 : fmin(1.0, (double)dep_viol    / max_dep);

	double viol_norm = c_pen.w_weight   * p1 +
	                   c_pen.w_volume   * p2 +
	                   c_pen.w_category * p3 +
	                   c_pen.w_incompat * p4 +
	                   c_pen.w_dep      * p5;

	double val_norm = (c_max_val > 0.0) ? (total_value / c_max_val) : 0.0;
	d_fitness[idx]  = c_pen.alpha * val_norm - c_pen.beta * viol_norm;

	d_feasible[idx] = (total_weight <= c_max_w + constants::FEASIBILITY_EPSILON &&
	                   total_volume <= c_max_v + constants::FEASIBILITY_EPSILON &&
	                   incomp_viol == 0 && dep_viol == 0) ? 1U : 0U;
}

// ── Optimización 5: Cruzamiento sin divergencia de warps ─────────────────────
//
// V2 (kernel_crossover): tiene if/else para decidir si cruzar o copiar padre A.
// Cuando un warp mezcla hilos que cruzan con hilos que copian, los dos caminos
// se ejecutan serialmente → divergencia.
//
// V3: se calcula siempre un punto de corte. Cuando no hay cruzamiento, el punto
// se fija en n_items (todo el hijo copia parent_a). El bucle de copia es siempre
// "child[i] = (i < point) ? parent_a[i] : parent_b[i]" — sin rama, el hardware
// emite instrucciones predicadas o selects sin divergencia de warp.
__global__ void kernel_crossover_v3(const uint8_t *__restrict__ d_population,
                                    uint8_t       *d_new_population,
                                    const int     *__restrict__ d_selected_a,
                                    const int     *__restrict__ d_selected_b,
                                    curandState   *d_rng_states,
                                    int pop_size,
                                    double crossover_rate) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= pop_size)
		return;

	const int n_items = c_ni;  // desde memoria constante

	const uint8_t *parent_a = d_population + (size_t)d_selected_a[idx] * n_items;
	const uint8_t *parent_b = d_population + (size_t)d_selected_b[idx] * n_items;
	uint8_t *child           = d_new_population + (size_t)idx * n_items;

	curandState local_state = d_rng_states[idx];

	double coin      = curand_uniform_double(&local_state);
	int raw_point    = (n_items >= 2) ?
	    1 + (int)(curand(&local_state) % (unsigned)(n_items - 1)) : 0;

	// point = n_items cuando no hay cruzamiento → todo el child copia parent_a.
	// Sin if/else en el bucle de abajo → sin divergencia de warp.
	int point = (coin <= crossover_rate && n_items >= 2) ? raw_point : n_items;

	for (int i = 0; i < n_items; ++i) {
		child[i] = (i < point) ? parent_a[i] : parent_b[i];
	}

	d_rng_states[idx] = local_state;
}

// ── Optimización 5: Mutación sin divergencia de warps ────────────────────────
//
// V2 (kernel_mutation): if (coin <= mutation_rate) { ... } → divergencia cuando
// en el mismo warp algunos hilos mutan y otros no.
//
// V3: siempre se genera la posición del gen candidato. El XOR se aplica con
// máscara: chromosome[gene] ^= (coin <= mutation_rate) ? 1U : 0U.
// XOR con 0 es no-op. Sin branch → instrucción predicada, sin divergencia.
__global__ void kernel_mutation_v3(uint8_t     *d_new_population,
                                   curandState *d_rng_states,
                                   int pop_size,
                                   double mutation_rate) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= pop_size)
		return;

	const int n_items = c_ni;

	curandState local_state = d_rng_states[idx];

	double  coin  = curand_uniform_double(&local_state);
	int     gene  = (int)(curand(&local_state) % (unsigned)n_items);
	uint8_t *chrom = d_new_population + (size_t)idx * n_items;

	// XOR con 0 (cuando no muta) → no-op sin branch
	chrom[gene] ^= (coin <= mutation_rate) ? 1U : 0U;

	d_rng_states[idx] = local_state;
}

// ─────────────────────────────────────────────────────────────────────────────
// run_genetic_algorithm_cuda_optimized — Loop principal, Variante 3.
//
// CPU/GPU:
//   GPU → init_rng, evaluate_fitness_v3, find_best_feasible (shared mem),
//         sort (cub), rank_selection, crossover_v3, mutation_v3,
//         preserve_elites (con d_sorted_indices), update_population.
//   CPU → reducción final de resultados de bloque (grid_pop iteraciones),
//         tracking de estancamiento, re-evaluación del mejor (FitnessBreakdown).
//
// Transferencias por generación (V3 vs V2):
//   V2: d_fitness (pop_size × 8 B) + BestPair (16 B) + elite_indices (elic × 4 B)
//   V3: block_best (grid_pop × 24 B) + cromosoma del mejor SOLO cuando mejora
//       → para pop_size=16384, block_size=256: 1536 B vs ~131 KB en V2 (~85× menos)
// ─────────────────────────────────────────────────────────────────────────────
GARunResult run_genetic_algorithm_cuda_optimized(const ProblemInstance &instance, const GAConfig &config, int seed,
                                                 int block_size) {

	if (instance.items.empty())
		throw std::invalid_argument("La instancia no tiene ítems.");
	if ((block_size & (block_size - 1)) != 0)
		throw std::invalid_argument("V3 requiere block_size potencia de 2 (128, 256, 512...).");

	const int pop_size   = config.population_size;
	const int n_items    = static_cast<int>(instance.items.size());
	const int rank_count = std::max(10, pop_size / 10);
	const int grid_pop   = (pop_size + block_size - 1) / block_size;

	// ── Subir instancia al device (única vez) ──
	GpuInstance gi = upload_instance(instance, config.penalties);

	if (gi.n_cats > V3_MAX_CATS)
		throw std::runtime_error("n_cats (" + std::to_string(gi.n_cats) +
		                         ") supera V3_MAX_CATS=" + std::to_string(V3_MAX_CATS) +
		                         ". Aumentar V3_MAX_CATS y recompilar.");

	// ── Reservar contexto (buffers de población, fitness, rng, etc.) ──
	GACudaContext ctx = alloc_cuda_context(pop_size, n_items, block_size);
	ctx.grid_pop      = grid_pop;

	// Reservar buffers de resultados parciales (Optimización 3)
	CUDA_CHECK(cudaMalloc(&ctx.d_block_best_idx,              grid_pop * sizeof(int)));
	CUDA_CHECK(cudaMalloc(&ctx.d_block_best_fitness,          grid_pop * sizeof(double)));
	CUDA_CHECK(cudaMalloc(&ctx.d_block_best_feasible_idx,     grid_pop * sizeof(int)));
	CUDA_CHECK(cudaMalloc(&ctx.d_block_best_feasible_fitness, grid_pop * sizeof(double)));

	// ── Optimización 1: subir constantes al caché L1 (una sola vez) ──
	GpuPenalties h_pen{config.penalties.alpha,      config.penalties.beta,
	                   config.penalties.w_weight,   config.penalties.w_volume,
	                   config.penalties.w_category, config.penalties.w_incompat,
	                   config.penalties.w_dep};
	CUDA_CHECK(cudaMemcpyToSymbol(c_pen,     &h_pen,                 sizeof(GpuPenalties)));
	CUDA_CHECK(cudaMemcpyToSymbol(c_max_w,   &gi.max_weight,         sizeof(double)));
	CUDA_CHECK(cudaMemcpyToSymbol(c_max_v,   &gi.max_volume,         sizeof(double)));
	CUDA_CHECK(cudaMemcpyToSymbol(c_tot_w,   &gi.total_weight,       sizeof(double)));
	CUDA_CHECK(cudaMemcpyToSymbol(c_tot_v,   &gi.total_volume,       sizeof(double)));
	CUDA_CHECK(cudaMemcpyToSymbol(c_max_val, &gi.max_possible_value, sizeof(double)));
	CUDA_CHECK(cudaMemcpyToSymbol(c_ni,      &gi.n_items,            sizeof(int)));
	CUDA_CHECK(cudaMemcpyToSymbol(c_nc,      &gi.n_cats,             sizeof(int)));
	CUDA_CHECK(cudaMemcpyToSymbol(c_ninc,    &gi.n_incomp,           sizeof(int)));
	CUDA_CHECK(cudaMemcpyToSymbol(c_ndep,    &gi.n_dep,              sizeof(int)));

	// ── Buffers para CUB sort y rank selection ──
	int    *d_identity      = nullptr;
	int    *d_sorted_indices = nullptr;
	double *d_sorted_fitness = nullptr;
	void   *d_temp_storage  = nullptr;
	size_t  temp_bytes      = 0;
	double *d_cum_probs     = nullptr;

	CUDA_CHECK(cudaMalloc(&d_identity,       pop_size * sizeof(int)));
	CUDA_CHECK(cudaMalloc(&d_sorted_indices, pop_size * sizeof(int)));
	CUDA_CHECK(cudaMalloc(&d_sorted_fitness, pop_size * sizeof(double)));

	// Vector identidad [0..pop_size-1]: valor de entrada para CUB sort (no se modifica)
	{
		vector<int> h_id(pop_size);
		std::iota(h_id.begin(), h_id.end(), 0);
		CUDA_CHECK(cudaMemcpy(d_identity, h_id.data(), pop_size * sizeof(int), cudaMemcpyHostToDevice));
	}

	// Calcular tamaño del storage temporal de CUB
	CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
	    nullptr, temp_bytes, ctx.d_fitness, d_sorted_fitness, d_identity, d_sorted_indices, pop_size));
	CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_bytes));

	// Probabilidades acumuladas geométricas para rank_selection (calculadas en host, subidas una vez)
	{
		const int    rc = rank_count;
		const double p  = constants::RANK_SELECTION_P;
		vector<double> h_cum(rc);
		double sum = 0.0;
		for (int i = 0; i < rc; ++i)
			sum += p * std::pow(1.0 - p, (double)i);
		double acc = 0.0;
		for (int i = 0; i < rc; ++i) {
			double raw = p * std::pow(1.0 - p, (double)i);
			h_cum[i] = (acc + raw) / sum;
			acc += raw;
		}
		CUDA_CHECK(cudaMalloc(&d_cum_probs, rc * sizeof(double)));
		CUDA_CHECK(cudaMemcpy(d_cum_probs, h_cum.data(), rc * sizeof(double), cudaMemcpyHostToDevice));
	}

	// ── Inicializar cuRAND ──
	kernel_init_rng<<<grid_pop, block_size>>>(ctx.d_rng_states, pop_size,
	                                          static_cast<unsigned long long>(seed));
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	// ── Población inicial factible en host (misma estrategia que V1 y V2) ──
	{
		std::mt19937 init_rng(static_cast<unsigned long long>(seed));
		const size_t pop_bytes = (size_t)pop_size * n_items;
		vector<uint8_t> h_pop(pop_bytes);
		for (int i = 0; i < pop_size; ++i) {
			Chromosome ch = make_feasible_random_chromosome(instance, init_rng);
			std::copy(ch.genes.begin(), ch.genes.end(), h_pop.begin() + (size_t)i * n_items);
		}
		CUDA_CHECK(cudaMemcpy(ctx.d_population, h_pop.data(), pop_bytes, cudaMemcpyHostToDevice));
	}

	// ── Buffers host para reducción de resultados parciales (grid_pop elementos) ──
	// Optimización 3: solo grid_pop valores (≤ 64 típicamente) vs pop_size en V2.
	vector<int>    h_blk_idx(grid_pop);
	vector<double> h_blk_fit(grid_pop);
	vector<int>    h_blk_feas_idx(grid_pop);
	vector<double> h_blk_feas_fit(grid_pop);

	FitnessEvaluator cpu_eval(instance, config.penalties);
	GARunResult      result{};
	result.has_feasible       = false;
	double best_fit_ever      = -1e18;
	double best_feas_ever     = -1e18;
	int    stagnation_counter = 0;

	// Tamaño de shared memory para kernel_find_best_feasible (árbol de reducción)
	const size_t smem_best = (size_t)block_size * (2 * sizeof(double) + 2 * sizeof(int));

	// ── Loop principal ──────────────────────────────────────────────────────
	for (int gen = 0; gen < config.generations; ++gen) {

		// 1. Evaluación de aptitud — Optimizaciones 1 y 2
		kernel_evaluate_fitness_v3<<<grid_pop, block_size>>>(
		    ctx.d_population,
		    gi.d_values, gi.d_weights, gi.d_volumes,
		    gi.d_item_cat, gi.d_cat_min, gi.d_cat_max,
		    gi.d_incomp_a, gi.d_incomp_b,
		    gi.d_dep_item, gi.d_dep_req,
		    ctx.d_fitness, ctx.d_feasible, pop_size);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		// 2. Búsqueda del mejor con shared memory — Optimización 3
		kernel_find_best_feasible<<<grid_pop, block_size, smem_best>>>(
		    ctx.d_fitness, ctx.d_feasible, pop_size,
		    ctx.d_block_best_idx,             ctx.d_block_best_fitness,
		    ctx.d_block_best_feasible_idx,    ctx.d_block_best_feasible_fitness);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		// 3. Transferir resultados parciales (grid_pop × 24 B) — Optimización 3 y 4
		CUDA_CHECK(cudaMemcpy(h_blk_idx.data(),      ctx.d_block_best_idx,
		                      grid_pop * sizeof(int),    cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(h_blk_fit.data(),      ctx.d_block_best_fitness,
		                      grid_pop * sizeof(double), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(h_blk_feas_idx.data(), ctx.d_block_best_feasible_idx,
		                      grid_pop * sizeof(int),    cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(h_blk_feas_fit.data(), ctx.d_block_best_feasible_fitness,
		                      grid_pop * sizeof(double), cudaMemcpyDeviceToHost));

		// 4. Reducción final en CPU sobre grid_pop valores (negligible)
		int    best_idx      = -1;
		double best_fit      = -1e18;
		int    best_feas_idx = -1;
		double best_feas_fit = -1e18;

		for (int b = 0; b < grid_pop; ++b) {
			if (h_blk_idx[b] >= 0 && h_blk_fit[b] > best_fit) {
				best_fit = h_blk_fit[b];
				best_idx = h_blk_idx[b];
			}
			if (h_blk_feas_idx[b] >= 0 && h_blk_feas_fit[b] > best_feas_fit) {
				best_feas_fit = h_blk_feas_fit[b];
				best_feas_idx = h_blk_feas_idx[b];
			}
		}

		// 5. Tracking del mejor global
		if (gen == 0 || best_fit > best_fit_ever) {
			best_fit_ever     = best_fit;
			stagnation_counter = 0;

			// Copiar solo el cromosoma del mejor (n_items bytes)
			Chromosome best_chrom;
			best_chrom.genes.resize((size_t)n_items);
			CUDA_CHECK(cudaMemcpy(best_chrom.genes.data(),
			                      ctx.d_population + (size_t)best_idx * n_items,
			                      n_items * sizeof(uint8_t), cudaMemcpyDeviceToHost));
			result.best_by_fitness = {best_chrom, cpu_eval.evaluate(best_chrom)};
		} else {
			++stagnation_counter;
		}

		// 6. Tracking del mejor factible
		if (best_feas_idx >= 0 && best_feas_fit > best_feas_ever) {
			best_feas_ever = best_feas_fit;
			Chromosome feas_chrom;
			feas_chrom.genes.resize((size_t)n_items);
			CUDA_CHECK(cudaMemcpy(feas_chrom.genes.data(),
			                      ctx.d_population + (size_t)best_feas_idx * n_items,
			                      n_items * sizeof(uint8_t), cudaMemcpyDeviceToHost));
			result.best_feasible = {feas_chrom, cpu_eval.evaluate(feas_chrom)};
			result.has_feasible  = true;
		}

		result.generations_executed = gen + 1;

		if (stagnation_counter >= config.max_stagnation_generations)
			break;

		// 7. CUB sort (fitness descendente → d_sorted_indices)
		// Salida: d_sorted_indices[0..elitism_count-1] = índices de élite (Optimización 4)
		CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
		    d_temp_storage, temp_bytes,
		    ctx.d_fitness, d_sorted_fitness,
		    d_identity,    d_sorted_indices,
		    pop_size));
		CUDA_CHECK(cudaGetLastError());

		// 8. Selección por ranking geométrico (reutiliza d_sorted_indices de V2)
		kernel_rank_selection<<<grid_pop, block_size>>>(
		    d_sorted_indices, ctx.d_selected_a, ctx.d_selected_b,
		    ctx.d_rng_states, pop_size, rank_count, d_cum_probs);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		// 9. Cruzamiento V3 (sin divergencia de warps) — Optimización 5
		kernel_crossover_v3<<<grid_pop, block_size>>>(
		    ctx.d_population, ctx.d_new_population,
		    ctx.d_selected_a, ctx.d_selected_b,
		    ctx.d_rng_states, pop_size, config.crossover_rate);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		// 10. Mutación V3 (XOR predicado, sin divergencia de warps) — Optimización 5
		kernel_mutation_v3<<<grid_pop, block_size>>>(
		    ctx.d_new_population, ctx.d_rng_states, pop_size, config.mutation_rate);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		// 11. Preservar élites — Optimización 4: d_sorted_indices ya tiene los índices
		// de los mejores individuos (top-k del CUB sort). Se pasa directamente al kernel,
		// evitando el ciclo partial_sort en CPU + cudaMemcpy de V2.
		const int grid_elite = (config.elitism_count + block_size - 1) / block_size;
		kernel_preserve_elites<<<grid_elite, block_size>>>(
		    ctx.d_population, ctx.d_new_population,
		    d_sorted_indices,          // top elitism_count = élites, ya en GPU
		    config.elitism_count, n_items);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		// 12. Actualizar población
		kernel_update_population<<<grid_pop, block_size>>>(
		    ctx.d_population, ctx.d_new_population, pop_size, n_items);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
	}

	// ── Liberar memoria ──────────────────────────────────────────────────────
	cudaFree(d_identity);
	cudaFree(d_sorted_indices);
	cudaFree(d_sorted_fitness);
	cudaFree(d_temp_storage);
	cudaFree(d_cum_probs);

	// Liberar d_block_best_* antes de free_cuda_context (que no los toca)
	cudaFree(ctx.d_block_best_idx);
	cudaFree(ctx.d_block_best_fitness);
	cudaFree(ctx.d_block_best_feasible_idx);
	cudaFree(ctx.d_block_best_feasible_fitness);
	ctx.d_block_best_idx              = nullptr;
	ctx.d_block_best_fitness          = nullptr;
	ctx.d_block_best_feasible_idx     = nullptr;
	ctx.d_block_best_feasible_fitness = nullptr;

	free_cuda_context(ctx);
	free_gpu_instance(gi);

	return result;
}
