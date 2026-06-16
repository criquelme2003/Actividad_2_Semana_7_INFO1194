// genetic_algorithm_cuda.cu  —  Variante 2: CUDA básico
// Kernels: kernel_init_rng, kernel_evaluate_fitness
// Funciones host: upload_instance, alloc/free helpers, run_genetic_algorithm_cuda_basic
// La población inicial factible se genera en host (make_feasible_random_chromosome)
// y se transfiere al device antes del loop principal.

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
#include <unordered_map>
#include <vector>

using std::size_t;
using std::string;
using std::unordered_map;
using std::vector;

// Macro de verificación de errores CUDA: lanza runtime_error con ubicación y mensaje.
#define CUDA_CHECK(call)                                                                                               \
	do {                                                                                                               \
		cudaError_t _err = (call);                                                                                     \
		if (_err != cudaSuccess) {                                                                                     \
			throw std::runtime_error(std::string("[CUDA] ") + __FILE__ + ":" + std::to_string(__LINE__) + " → " +      \
			                         cudaGetErrorString(_err));                                                        \
		}                                                                                                              \
	} while (0)

// kernel_init_rng — Inicializa un estado cuRAND por individuo.
// sequence=idx garantiza secuencias estadísticamente independientes entre hilos.
__global__ void kernel_init_rng(curandState *d_rng_states, int pop_size, unsigned long long seed) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= pop_size)
		return;

	curand_init(seed, static_cast<unsigned long long>(idx), 0ULL, &d_rng_states[idx]);
}

// kernel_evaluate_fitness — Evalúa el fitness de todos los individuos.
// Estrategia: 1 hilo = 1 individuo (independencia total, sin sincronización).
// Pasos: (1) acumular valor/peso/volumen, (2) excesos de capacidad [duras],
//        (3) violaciones de categoría [blanda], (4) incompatibilidades [dura],
//        (5) dependencias rotas [dura], (6) calcular fitness y factibilidad.
__global__ void kernel_evaluate_fitness(const uint8_t *__restrict__ d_population, const double *__restrict__ d_values,
                                        const double *__restrict__ d_weights, const double *__restrict__ d_volumes,
                                        const int *__restrict__ d_item_cat, const int *__restrict__ d_cat_min,
                                        const int *__restrict__ d_cat_max, int n_cats,
                                        const int *__restrict__ d_incomp_a, const int *__restrict__ d_incomp_b,
                                        int n_incomp, const int *__restrict__ d_dep_item,
                                        const int *__restrict__ d_dep_req, int n_dep, double max_weight,
                                        double max_volume, double inst_total_weight, double inst_total_volume,
                                        double max_possible_value, GpuPenalties penalties, double *d_fitness,
                                        uint8_t *d_feasible, int pop_size, int n_items) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= pop_size)
		return;

	// Puntero al inicio de los genes de este individuo (layout row-major: individuo * n_items + item)
	const uint8_t *genes = d_population + static_cast<size_t>(idx) * n_items;

	// Paso 1: acumular valor, peso y volumen
	double total_value = 0.0;
	double total_weight = 0.0;
	double total_volume = 0.0;

	for (int i = 0; i < n_items; ++i) {
		if (genes[i] != 0U) {
			total_value += d_values[i];
			total_weight += d_weights[i];
			total_volume += d_volumes[i];
		}
	}

	// Paso 2: excesos de capacidad (restricciones duras)
	double excess_weight = (total_weight > max_weight) ? (total_weight - max_weight) : 0.0;
	double excess_volume = (total_volume > max_volume) ? (total_volume - max_volume) : 0.0;

	// Paso 3: violaciones de categoría (restricción blanda).
	// Complejidad O(n_cats * n_items). Variante optimizada puede usar un solo
	// pase con tabla de conteo en shared memory.
	int cat_violations = 0;
	for (int c = 0; c < n_cats; ++c) {
		int cnt = 0;
		for (int i = 0; i < n_items; ++i) {
			if (genes[i] != 0U && d_item_cat[i] == c) {
				++cnt;
			}
		}
		// Violación por debajo del mínimo
		int under = d_cat_min[c] - cnt;
		if (under > 0)
			cat_violations += under;
		int max_c = d_cat_max[c]; // -1 = sin límite superior
		if (max_c >= 0) {
			int over = cnt - max_c;
			if (over > 0)
				cat_violations += over;
		}
	}

	// Paso 4: incompatibilidades (restricción dura)
	int incomp_violations = 0;
	for (int k = 0; k < n_incomp; ++k) {
		if (genes[d_incomp_a[k]] != 0U && genes[d_incomp_b[k]] != 0U) {
			++incomp_violations;
		}
	}

	// Paso 5: dependencias rotas (restricción dura)
	int dep_violations = 0;
	for (int k = 0; k < n_dep; ++k) {
		if (genes[d_dep_item[k]] != 0U && genes[d_dep_req[k]] == 0U) {
			++dep_violations;
		}
	}

	// Paso 6: fitness normalizado y factibilidad dura
	// fitness = alpha*valor_norm - beta*violacion_norm.
	// Factibilidad dura: peso, volumen, incompatibilidades y dependencias.
	// Categoría es restricción blanda (penaliza pero no invalida).
	double valor_norm = (max_possible_value > 0.0) ? (total_value / max_possible_value) : 0.0;

	double max_excess_w = inst_total_weight - max_weight;
	double max_excess_v = inst_total_volume - max_volume;
	double max_cat = static_cast<double>(n_items);
	double max_incompat = static_cast<double>((n_incomp > 0) ? n_incomp : 1);
	double max_dep = static_cast<double>((n_dep > 0) ? n_dep : 1);

	double p1 = fmin(1.0, excess_weight / max_excess_w);
	double p2 = fmin(1.0, excess_volume / max_excess_v);
	double p3 = fmin(1.0, static_cast<double>(cat_violations) / max_cat);
	double p4 = (n_incomp == 0) ? 0.0 : fmin(1.0, static_cast<double>(incomp_violations) / max_incompat);
	double p5 = (n_dep == 0) ? 0.0 : fmin(1.0, static_cast<double>(dep_violations) / max_dep);

	double violacion_norm = penalties.w_weight * p1 + penalties.w_volume * p2 + penalties.w_category * p3 +
	                        penalties.w_incompat * p4 + penalties.w_dep * p5;

	d_fitness[idx] = penalties.alpha * valor_norm - penalties.beta * violacion_norm;

	d_feasible[idx] = (total_weight <= max_weight + constants::FEASIBILITY_EPSILON &&
	                   total_volume <= max_volume + constants::FEASIBILITY_EPSILON && incomp_violations == 0 &&
	                   dep_violations == 0)
	                      ? 1U
	                      : 0U;
}

// ─── Helpers de host ─────────────────────────────────────────────────────────

// Construye el mapa categoría-string → entero y los vectores planos de reglas.
static void build_category_mapping(const ProblemInstance &inst, vector<int> &h_item_cat, vector<int> &h_cat_min,
                                   vector<int> &h_cat_max, int &n_cats_out) {

	unordered_map<string, int> cat_map;
	int next_id = 0;

	for (const auto &item : inst.items) {
		if (cat_map.find(item.category) == cat_map.end()) {
			cat_map[item.category] = next_id++;
		}
	}

	n_cats_out = next_id;
	h_cat_min.assign(static_cast<size_t>(n_cats_out), 0);
	h_cat_max.assign(static_cast<size_t>(n_cats_out), -1); // -1 = sin límite superior

	for (const auto &rule : inst.category_rules) {
		auto it = cat_map.find(rule.category);
		if (it != cat_map.end()) {
			int cid = it->second;
			h_cat_min[static_cast<size_t>(cid)] = rule.minimum;
			h_cat_max[static_cast<size_t>(cid)] = rule.maximum;
		}
	}

	h_item_cat.resize(inst.items.size());
	for (size_t i = 0; i < inst.items.size(); ++i) {
		h_item_cat[i] = cat_map.at(inst.items[i].category);
	}
}

// upload_instance — Aplana la instancia y la transfiere al device (única vez antes del loop).
static GpuInstance upload_instance(const ProblemInstance &inst, const PenaltyConfig & /*pen*/) {
	const int n = static_cast<int>(inst.items.size());

	vector<double> h_values(n), h_weights(n), h_volumes(n);
	for (int i = 0; i < n; ++i) {
		h_values[i] = inst.items[i].value;
		h_weights[i] = inst.items[i].weight;
		h_volumes[i] = inst.items[i].volume;
	}

	vector<int> h_item_cat, h_cat_min, h_cat_max;
	int n_cats = 0;
	build_category_mapping(inst, h_item_cat, h_cat_min, h_cat_max, n_cats);

	// Aplanar incompatibilidades
	const int n_incomp = static_cast<int>(inst.incompatibility_indices.size());
	vector<int> h_incomp_a(n_incomp), h_incomp_b(n_incomp);
	for (int k = 0; k < n_incomp; ++k) {
		h_incomp_a[k] = static_cast<int>(inst.incompatibility_indices[k].first);
		h_incomp_b[k] = static_cast<int>(inst.incompatibility_indices[k].second);
	}

	// Aplanar dependencias
	const int n_dep = static_cast<int>(inst.dependency_indices.size());
	vector<int> h_dep_item(n_dep), h_dep_req(n_dep);
	for (int k = 0; k < n_dep; ++k) {
		h_dep_item[k] = static_cast<int>(inst.dependency_indices[k].first);
		h_dep_req[k] = static_cast<int>(inst.dependency_indices[k].second);
	}

	// Reservar y transferir
	GpuInstance gi{};
	gi.n_items = n;
	gi.n_cats = n_cats;
	gi.n_incomp = n_incomp;
	gi.n_dep = n_dep;
	gi.max_weight = inst.max_weight;
	gi.max_volume = inst.max_volume;
	gi.total_weight = inst.total_weight;
	gi.total_volume = inst.total_volume;
	gi.max_possible_value = inst.max_possible_value;

	CUDA_CHECK(cudaMalloc(&gi.d_values, n * sizeof(double)));
	CUDA_CHECK(cudaMalloc(&gi.d_weights, n * sizeof(double)));
	CUDA_CHECK(cudaMalloc(&gi.d_volumes, n * sizeof(double)));
	CUDA_CHECK(cudaMalloc(&gi.d_item_cat, n * sizeof(int)));

	CUDA_CHECK(cudaMemcpy(gi.d_values, h_values.data(), n * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(gi.d_weights, h_weights.data(), n * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(gi.d_volumes, h_volumes.data(), n * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(gi.d_item_cat, h_item_cat.data(), n * sizeof(int), cudaMemcpyHostToDevice));

	if (n_cats > 0) {
		CUDA_CHECK(cudaMalloc(&gi.d_cat_min, n_cats * sizeof(int)));
		CUDA_CHECK(cudaMalloc(&gi.d_cat_max, n_cats * sizeof(int)));
		CUDA_CHECK(cudaMemcpy(gi.d_cat_min, h_cat_min.data(), n_cats * sizeof(int), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(gi.d_cat_max, h_cat_max.data(), n_cats * sizeof(int), cudaMemcpyHostToDevice));
	}

	if (n_incomp > 0) {
		CUDA_CHECK(cudaMalloc(&gi.d_incomp_a, n_incomp * sizeof(int)));
		CUDA_CHECK(cudaMalloc(&gi.d_incomp_b, n_incomp * sizeof(int)));
		CUDA_CHECK(cudaMemcpy(gi.d_incomp_a, h_incomp_a.data(), n_incomp * sizeof(int), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(gi.d_incomp_b, h_incomp_b.data(), n_incomp * sizeof(int), cudaMemcpyHostToDevice));
	}

	if (n_dep > 0) {
		CUDA_CHECK(cudaMalloc(&gi.d_dep_item, n_dep * sizeof(int)));
		CUDA_CHECK(cudaMalloc(&gi.d_dep_req, n_dep * sizeof(int)));
		CUDA_CHECK(cudaMemcpy(gi.d_dep_item, h_dep_item.data(), n_dep * sizeof(int), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(gi.d_dep_req, h_dep_req.data(), n_dep * sizeof(int), cudaMemcpyHostToDevice));
	}

	return gi;
}

static void free_gpu_instance(GpuInstance &gi) {
	cudaFree(gi.d_values);
	cudaFree(gi.d_weights);
	cudaFree(gi.d_volumes);
	cudaFree(gi.d_item_cat);
	cudaFree(gi.d_cat_min);
	cudaFree(gi.d_cat_max);
	cudaFree(gi.d_incomp_a);
	cudaFree(gi.d_incomp_b);
	cudaFree(gi.d_dep_item);
	cudaFree(gi.d_dep_req);
	gi = GpuInstance{};
}

// alloc_cuda_context — Reserva todos los buffers del GA en device.
// Población: layout row-major [pop_size * n_items], individuo i en genes[i*n_items..].
static GACudaContext alloc_cuda_context(int pop_size, int n_items, int block_size) {
	GACudaContext ctx{};
	ctx.pop_size = pop_size;
	ctx.n_items = n_items;
	ctx.block_size = block_size;

	const size_t pop_bytes = static_cast<size_t>(pop_size) * n_items * sizeof(uint8_t);

	CUDA_CHECK(cudaMalloc(&ctx.d_population, pop_bytes));
	CUDA_CHECK(cudaMalloc(&ctx.d_new_population, pop_bytes));
	CUDA_CHECK(cudaMalloc(&ctx.d_fitness, pop_size * sizeof(double)));
	CUDA_CHECK(cudaMalloc(&ctx.d_feasible, pop_size * sizeof(uint8_t)));
	CUDA_CHECK(cudaMalloc(&ctx.d_rng_states, pop_size * sizeof(curandState)));
	CUDA_CHECK(cudaMalloc(&ctx.d_selected_a, pop_size * sizeof(int)));
	CUDA_CHECK(cudaMalloc(&ctx.d_selected_b, pop_size * sizeof(int)));

	// grid_pop se calcula en run_genetic_algorithm_cuda_basic como variable local.
	// ctx.grid_pop se deja sin inicializar (0); la reducción se hará en CPU para V2.

	return ctx;
}

static void free_cuda_context(GACudaContext &ctx) {
	cudaFree(ctx.d_population);
	cudaFree(ctx.d_new_population);
	cudaFree(ctx.d_fitness);
	cudaFree(ctx.d_feasible);
	cudaFree(ctx.d_rng_states);
	cudaFree(ctx.d_selected_a);
	cudaFree(ctx.d_selected_b);
	// ctx.d_block_best_* no se usan en V2 (mejor se busca en CPU para mantener V2 básica).
	ctx = GACudaContext{};
}

// run_genetic_algorithm_cuda_basic — Loop principal, Variante 2.
//
// CPU/GPU:
//   GPU → init_rng, evaluate_fitness, find_best_simple, sort (cub),
//         rank_selection, crossover, mutation, preserve_elites, update
//   CPU → tracking de estancamiento y re-evaluación del mejor (FitnessBreakdown)
//
// Transferencias por generación:
//   device→host: d_fitness (pop_size × 8 bytes)
//                best_global + best_feas (2 × BestPair = 16 bytes)
//                cromosoma del mejor solo cuando mejora (n_items bytes)
//   host→device: d_elite_indices (elitism_count × 4 bytes)
//                inicialización de d_best_global/d_best_feas (2 × BestPair = 16 bytes)
//   La población completa NUNCA se transfiere durante el loop.
GARunResult run_genetic_algorithm_cuda_basic(const ProblemInstance &instance, const GAConfig &config, int seed,
                                             int block_size) {

	if (instance.items.empty()) {
		throw std::invalid_argument("La instancia no tiene ítems.");
	}

	const int pop_size = config.population_size;
	const int n_items = static_cast<int>(instance.items.size());
	const int rank_count = std::max(10, pop_size / 10); // R dinámico: 10% de la población, mínimo 10

	// Subir instancia al device (única vez)
	GpuInstance gi = upload_instance(instance, config.penalties);
	GACudaContext ctx = alloc_cuda_context(pop_size, n_items, block_size);

	int *d_elite_indices = nullptr;
	CUDA_CHECK(cudaMalloc(&d_elite_indices, config.elitism_count * sizeof(int)));

	// Buffers para rank selection (cub sort + kernel)
	int *d_identity = nullptr;
	int *d_sorted_indices = nullptr;
	double *d_sorted_fitness = nullptr;
	void *d_temp_storage = nullptr;
	size_t temp_storage_bytes = 0;
	double *d_cum_probs = nullptr;

	// Buffers para búsqueda atómica del mejor (V2: paralelo, sin shared memory)
	unsigned long long *d_best_global = nullptr;
	unsigned long long *d_best_feas = nullptr;

	CUDA_CHECK(cudaMalloc(&d_identity, pop_size * sizeof(int)));
	CUDA_CHECK(cudaMalloc(&d_sorted_indices, pop_size * sizeof(int)));
	CUDA_CHECK(cudaMalloc(&d_sorted_fitness, pop_size * sizeof(double)));
	CUDA_CHECK(cudaMalloc(&d_best_global, sizeof(unsigned long long)));
	CUDA_CHECK(cudaMalloc(&d_best_feas, sizeof(unsigned long long)));

	// Precalcular probabilidades acumuladas para ranking geométrico (host→device, una vez)
	// R = rank_count = max(10, pop_size/10): selección sobre el top R (justifica cub sort + kernel).
	{
		const int rc = rank_count;
		const double p = constants::RANK_SELECTION_P;
		vector<double> h_cum_probs(rc);
		double sum = 0.0;
		for (int i = 0; i < rc; ++i) {
			sum += p * std::pow(1.0 - p, static_cast<double>(i));
		}
		double acc = 0.0;
		for (int i = 0; i < rc; ++i) {
			double raw = p * std::pow(1.0 - p, static_cast<double>(i));
			h_cum_probs[i] = (acc + raw) / sum;
			acc += raw;
		}
		CUDA_CHECK(cudaMalloc(&d_cum_probs, rc * sizeof(double)));
		CUDA_CHECK(cudaMemcpy(d_cum_probs, h_cum_probs.data(), rc * sizeof(double), cudaMemcpyHostToDevice));
	}

	// Inicializar vector identidad [0..pop_size-1] y consultar tamaño de temp_storage para cub
	{
		vector<int> h_identity(pop_size);
		std::iota(h_identity.begin(), h_identity.end(), 0);
		CUDA_CHECK(cudaMemcpy(d_identity, h_identity.data(), pop_size * sizeof(int), cudaMemcpyHostToDevice));

		CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
		    nullptr, temp_storage_bytes, ctx.d_fitness, d_sorted_fitness, d_identity, d_sorted_indices, pop_size));
		CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
	}

	GpuPenalties penalties{config.penalties.alpha,     config.penalties.beta,       config.penalties.w_weight,
	                       config.penalties.w_volume,  config.penalties.w_category, config.penalties.w_incompat,
	                       config.penalties.w_dep};

	// 1 hilo por individuo; hilos sobrantes descartados con el guard idx>=pop_size
	const int grid_pop = (pop_size + block_size - 1) / block_size;

	// Inicializar cuRAND
	kernel_init_rng<<<grid_pop, block_size>>>(ctx.d_rng_states, pop_size, static_cast<unsigned long long>(seed));
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	// Generar población inicial factible en host (misma reparación que la variante secuencial)
	// y transferirla al device. Garantiza que todos los individuos cumplan las
	// restricciones duras desde el arranque.
	{
		std::mt19937 init_rng(static_cast<unsigned long long>(seed));
		const size_t pop_bytes = static_cast<size_t>(pop_size) * static_cast<size_t>(n_items);
		vector<uint8_t> h_population(pop_bytes);

		for (int i = 0; i < pop_size; ++i) {
			Chromosome chrom = make_feasible_random_chromosome(instance, init_rng);
			std::copy(chrom.genes.begin(), chrom.genes.end(),
			          h_population.begin() + static_cast<size_t>(i) * n_items);
		}

		CUDA_CHECK(cudaMemcpy(ctx.d_population, h_population.data(), pop_bytes, cudaMemcpyHostToDevice));
	}

	// Buffers host: fitness para elitismo (partial_sort en CPU) y resultados de búsqueda
	vector<double> h_fitness(pop_size);

	// Evaluador CPU para recalcular FitnessBreakdown completo del mejor individuo
	FitnessEvaluator cpu_evaluator(instance, config.penalties);

	GARunResult result{};
	result.has_feasible = false;
	double best_fitness_ever = -1e18;
	double best_feasible_ever = -1e18;
	int stagnation_counter = 0;

	// Loop principal
	for (int gen = 0; gen < config.generations; ++gen) {

		// Evaluación de aptitud en GPU
		kernel_evaluate_fitness<<<grid_pop, block_size>>>(
		    ctx.d_population, gi.d_values, gi.d_weights, gi.d_volumes, gi.d_item_cat, gi.d_cat_min, gi.d_cat_max,
		    gi.n_cats, gi.d_incomp_a, gi.d_incomp_b, gi.n_incomp, gi.d_dep_item, gi.d_dep_req, gi.n_dep, gi.max_weight,
		    gi.max_volume, gi.total_weight, gi.total_volume, gi.max_possible_value, penalties, ctx.d_fitness,
		    ctx.d_feasible, pop_size, n_items);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		// Transferir fitness al host (necesario para partial_sort del elitismo)
		CUDA_CHECK(cudaMemcpy(h_fitness.data(), ctx.d_fitness, pop_size * sizeof(double), cudaMemcpyDeviceToHost));

		// Búsqueda paralela del mejor (global y factible) — kernel sin shared memory
		{
			BestPair init = {-1, -FLT_MAX};
			CUDA_CHECK(cudaMemcpy(d_best_global, &init, sizeof(BestPair), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(d_best_feas, &init, sizeof(BestPair), cudaMemcpyHostToDevice));

			kernel_find_best_simple<<<grid_pop, block_size>>>(ctx.d_fitness, ctx.d_feasible, pop_size,
			                                                  d_best_global, d_best_feas);
			CUDA_CHECK(cudaGetLastError());
			CUDA_CHECK(cudaDeviceSynchronize());
		}

		// Traer resultados al host (2 × 8 bytes = 16 bytes por generación)
		BestPair best_global;
		BestPair best_feas;
		CUDA_CHECK(cudaMemcpy(&best_global, d_best_global, sizeof(BestPair), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(&best_feas, d_best_feas, sizeof(BestPair), cudaMemcpyDeviceToHost));

		const int best_idx = best_global.idx;
		const double gen_best_fit = h_fitness[best_idx];
		const int best_feasible_idx = best_feas.idx;
		const double gen_best_feasible_fit = (best_feasible_idx >= 0) ? h_fitness[best_feasible_idx] : -1e18;

		// Tracking del mejor (estancamiento)
		if (gen == 0 || gen_best_fit > best_fitness_ever) {
			best_fitness_ever = gen_best_fit;
			stagnation_counter = 0;

			// Copiar cromosoma ganador al host y re-evaluar en CPU para FitnessBreakdown completo
			Chromosome best_chrom;
			best_chrom.genes.resize(static_cast<size_t>(n_items));
			CUDA_CHECK(cudaMemcpy(best_chrom.genes.data(), ctx.d_population + static_cast<size_t>(best_idx) * n_items,
			                      n_items * sizeof(uint8_t), cudaMemcpyDeviceToHost));
			result.best_by_fitness = {best_chrom, cpu_evaluator.evaluate(best_chrom)};
		} else {
			++stagnation_counter;
		}

		// Tracking del mejor individuo factible
		if (best_feasible_idx >= 0 && gen_best_feasible_fit > best_feasible_ever) {
			best_feasible_ever = gen_best_feasible_fit;

			// Copiar su cromosoma y re-evaluar en CPU
			Chromosome feas_chrom;
			feas_chrom.genes.resize(static_cast<size_t>(n_items));
			CUDA_CHECK(cudaMemcpy(feas_chrom.genes.data(),
			                      ctx.d_population + static_cast<size_t>(best_feasible_idx) * n_items,
			                      n_items * sizeof(uint8_t), cudaMemcpyDeviceToHost));
			result.best_feasible = {feas_chrom, cpu_evaluator.evaluate(feas_chrom)};
			result.has_feasible = true;
		}

		result.generations_executed = gen + 1;

		// Criterio de término por estancamiento
		if (stagnation_counter >= config.max_stagnation_generations) {
			break;
		}

		// Calcular índices de élite en CPU y subirlos al device.
		// Se hace aquí (después del break) porque d_elite_indices solo es
		// necesario para kernel_preserve_elites, que viene a continuación.
		// Hacerlo antes del break era trabajo innecesario en la última generación.
		vector<int> indices(pop_size);
		std::iota(indices.begin(), indices.end(), 0);
		std::partial_sort(indices.begin(), indices.begin() + config.elitism_count, indices.end(),
		                  [&h_fitness](int a, int b) { return h_fitness[a] > h_fitness[b]; });

		CUDA_CHECK(
		    cudaMemcpy(d_elite_indices, indices.data(), config.elitism_count * sizeof(int), cudaMemcpyHostToDevice));

		// Operadores genéticos

		// Ordenar fitness descendente con cub (ranking para selección geométrica)
		CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
		    d_temp_storage, temp_storage_bytes,
		    ctx.d_fitness, d_sorted_fitness,
		    d_identity, d_sorted_indices,
		    pop_size));
		CUDA_CHECK(cudaGetLastError());

		// Selección por ranking geométrico (misma semántica que Variante 1)
		kernel_rank_selection<<<grid_pop, block_size>>>(
		    d_sorted_indices, ctx.d_selected_a, ctx.d_selected_b, ctx.d_rng_states,
		    pop_size, rank_count, d_cum_probs);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		// Cruzamiento de un punto
		kernel_crossover<<<grid_pop, block_size>>>(ctx.d_population, ctx.d_new_population, ctx.d_selected_a,
		                                           ctx.d_selected_b, ctx.d_rng_states, pop_size, n_items,
		                                           config.crossover_rate);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		// Mutación bit-flip
		kernel_mutation<<<grid_pop, block_size>>>(ctx.d_new_population, ctx.d_rng_states, pop_size, n_items,
		                                          config.mutation_rate);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		// Preservar élites: copiar los config.elitism_count mejores individuos de
		// d_population (generación actual) hacia d_new_population, ANTES de
		// sobrescribir d_population. Debe ir después de cruzamiento/mutación
		// (que escriben sobre d_new_population) y antes de la actualización.
		const int grid_elite = (config.elitism_count + block_size - 1) / block_size;
		kernel_preserve_elites<<<grid_elite, block_size>>>(ctx.d_population, ctx.d_new_population, d_elite_indices,
		                                                   config.elitism_count, n_items);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		// Actualización: copiar d_new_population -> d_population
		kernel_update_population<<<grid_pop, block_size>>>(ctx.d_population, ctx.d_new_population, pop_size, n_items);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
	}

	// Liberar memoria de device
	cudaFree(d_elite_indices);
	cudaFree(d_identity);
	cudaFree(d_sorted_indices);
	cudaFree(d_sorted_fitness);
	cudaFree(d_temp_storage);
	cudaFree(d_cum_probs);
	cudaFree(d_best_global);
	cudaFree(d_best_feas);
	free_cuda_context(ctx);
	free_gpu_instance(gi);

	return result;
}
