// genetic_algorithm_cuda.cu  —  Variante 2: CUDA básico
// Kernels: kernel_init_rng, kernel_init_population, kernel_evaluate_fitness
// Funciones host: upload_instance, alloc/free helpers, run_genetic_algorithm_cuda_basic

#include "fitness.hpp"
#include "genetic_algorithm_cuda.cuh"

#include <algorithm>
#include <numeric>
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

// kernel_init_population — Genera la población inicial (Bernoulli 0.5 por gen).
// 1 hilo = 1 individuo. Estado cuRAND cargado a registro local para evitar
// accesos repetidos a memoria global dentro del loop.
__global__ void kernel_init_population(uint8_t *d_population, int pop_size, int n_items, curandState *d_rng_states) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= pop_size)
		return;

	curandState local_state = d_rng_states[idx]; // registro local: evita accesos globales en el loop

	uint8_t *genes = d_population + static_cast<size_t>(idx) * n_items;

	for (int i = 0; i < n_items; ++i) {
		genes[i] = (curand_uniform(&local_state) < 0.5f) ? 1U : 0U;
	}

	d_rng_states[idx] = local_state; // devolver estado actualizado a memoria global
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
                                        double max_volume, GpuPenalties penalties, double *d_fitness,
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

	// Paso 6: fitness y factibilidad dura
	// Factibilidad dura: peso, volumen, incompatibilidades y dependencias.
	// Categoría es restricción blanda (penaliza pero no invalida).
	double penalty = penalties.alpha * excess_weight + penalties.beta * excess_volume +
	                 penalties.gamma * static_cast<double>(cat_violations) +
	                 penalties.delta * static_cast<double>(incomp_violations) +
	                 penalties.epsilon * static_cast<double>(dep_violations);

	d_fitness[idx] = total_value - penalty;

	d_feasible[idx] =
	    (excess_weight <= 0.0 && excess_volume <= 0.0 && incomp_violations == 0 && dep_violations == 0) ? 1U : 0U;
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

	// Resultados parciales (1 valor por bloque) de la reducción "mejor individuo" /
	// "mejor individuo factible".
	ctx.grid_pop = (pop_size + block_size - 1) / block_size;
	CUDA_CHECK(cudaMalloc(&ctx.d_block_best_idx, ctx.grid_pop * sizeof(int)));
	CUDA_CHECK(cudaMalloc(&ctx.d_block_best_fitness, ctx.grid_pop * sizeof(double)));
	CUDA_CHECK(cudaMalloc(&ctx.d_block_best_feasible_idx, ctx.grid_pop * sizeof(int)));
	CUDA_CHECK(cudaMalloc(&ctx.d_block_best_feasible_fitness, ctx.grid_pop * sizeof(double)));

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
	cudaFree(ctx.d_block_best_idx);
	cudaFree(ctx.d_block_best_fitness);
	cudaFree(ctx.d_block_best_feasible_idx);
	cudaFree(ctx.d_block_best_feasible_fitness);
	ctx = GACudaContext{};
}

// run_genetic_algorithm_cuda_basic — Loop principal, Variante 2.
//
// CPU/GPU:
//   GPU → init_rng, init_population, evaluate_fitness, operadores (Cristóbal)
//   CPU → tracking de estancamiento y mejor individuo (sobre h_fitness/h_feasible)
//
// Transferencias por generación:
//   device→host: d_fitness + d_feasible (≤ pop_size × 9 bytes)
//                cromosoma del mejor solo cuando mejora (n_items bytes)
//   host→device: d_elite_indices (elitism_count × 4 bytes)
//   La población completa NUNCA se transfiere durante el loop.
GARunResult run_genetic_algorithm_cuda_basic(const ProblemInstance &instance, const GAConfig &config, int seed,
                                             int block_size) {

	if (instance.items.empty()) {
		throw std::invalid_argument("La instancia no tiene ítems.");
	}

	const int pop_size = config.population_size;
	const int n_items = static_cast<int>(instance.items.size());

	// Subir instancia al device (única vez)
	GpuInstance gi = upload_instance(instance, config.penalties);
	GACudaContext ctx = alloc_cuda_context(pop_size, n_items, block_size);

	int *d_elite_indices = nullptr;
	CUDA_CHECK(cudaMalloc(&d_elite_indices, config.elitism_count * sizeof(int)));

	GpuPenalties penalties{config.penalties.alpha, config.penalties.beta, config.penalties.gamma,
	                       config.penalties.delta, config.penalties.epsilon};

	// 1 hilo por individuo; hilos sobrantes descartados con el guard idx>=pop_size
	const int grid_pop = (pop_size + block_size - 1) / block_size;

	// Inicializar cuRAND
	kernel_init_rng<<<grid_pop, block_size>>>(ctx.d_rng_states, pop_size, static_cast<unsigned long long>(seed));
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	// Generar población inicial aleatoria
	kernel_init_population<<<grid_pop, block_size>>>(ctx.d_population, pop_size, n_items, ctx.d_rng_states);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	// Buffers host: solo fitness, nunca la población completa.
	// (d_feasible ya no se transfiere completo: el kernel de reducción lo
	// resume en grid_pop valores).
	vector<double> h_fitness(pop_size);

	// Resultados de la reducción "mejor individuo" / "mejor factible" (1 valor por bloque)
	vector<int> h_block_best_idx(ctx.grid_pop);
	vector<double> h_block_best_fitness(ctx.grid_pop);
	vector<int> h_block_best_feasible_idx(ctx.grid_pop);
	vector<double> h_block_best_feasible_fitness(ctx.grid_pop);

	const size_t reduction_shared_bytes = static_cast<size_t>(block_size) * (2 * sizeof(double) + 2 * sizeof(int));

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
		    gi.max_volume, penalties, ctx.d_fitness, ctx.d_feasible, pop_size, n_items);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		// Transferir fitness completo al host (lo necesita partial_sort para elitismo)
		CUDA_CHECK(cudaMemcpy(h_fitness.data(), ctx.d_fitness, pop_size * sizeof(double), cudaMemcpyDeviceToHost));

		// Reducción en GPU: mejor individuo (global) y mejor individuo factible,
		// resumidos en grid_pop valores parciales (1 por bloque).
		kernel_find_best_feasible<<<grid_pop, block_size, reduction_shared_bytes>>>(
		    ctx.d_fitness, ctx.d_feasible, pop_size, ctx.d_block_best_idx, ctx.d_block_best_fitness,
		    ctx.d_block_best_feasible_idx, ctx.d_block_best_feasible_fitness);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		CUDA_CHECK(cudaMemcpy(h_block_best_idx.data(), ctx.d_block_best_idx, ctx.grid_pop * sizeof(int),
		                      cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(h_block_best_fitness.data(), ctx.d_block_best_fitness, ctx.grid_pop * sizeof(double),
		                      cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(h_block_best_feasible_idx.data(), ctx.d_block_best_feasible_idx,
		                      ctx.grid_pop * sizeof(int), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(h_block_best_feasible_fitness.data(), ctx.d_block_best_feasible_fitness,
		                      ctx.grid_pop * sizeof(double), cudaMemcpyDeviceToHost));

		// Reducción final (sobre grid_pop elementos) en CPU
		int best_idx = h_block_best_idx[0];
		double gen_best_fit = h_block_best_fitness[0];
		int best_feasible_idx = h_block_best_feasible_idx[0];
		double gen_best_feasible_fit = h_block_best_feasible_fitness[0];

		for (int b = 1; b < ctx.grid_pop; ++b) {
			if (h_block_best_fitness[b] > gen_best_fit) {
				gen_best_fit = h_block_best_fitness[b];
				best_idx = h_block_best_idx[b];
			}
			if (h_block_best_feasible_idx[b] >= 0 &&
			    (best_feasible_idx < 0 || h_block_best_feasible_fitness[b] > gen_best_feasible_fit)) {
				gen_best_feasible_fit = h_block_best_feasible_fitness[b];
				best_feasible_idx = h_block_best_feasible_idx[b];
			}
		}

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

		// Calcular índices de élite en CPU y subirlos al device
		vector<int> indices(pop_size);
		std::iota(indices.begin(), indices.end(), 0);
		std::partial_sort(indices.begin(), indices.begin() + config.elitism_count, indices.end(),
		                  [&h_fitness](int a, int b) { return h_fitness[a] > h_fitness[b]; });

		CUDA_CHECK(
		    cudaMemcpy(d_elite_indices, indices.data(), config.elitism_count * sizeof(int), cudaMemcpyHostToDevice));

		result.generations_executed = gen + 1;

		// Criterio de término por estancamiento
		if (stagnation_counter >= config.max_stagnation_generations) {
			break;
		}

		// Operadores genéticos

		// Selección por torneo
		kernel_tournament_selection<<<grid_pop, block_size>>>(
		    ctx.d_fitness, ctx.d_selected_a, ctx.d_selected_b, ctx.d_rng_states, pop_size,
		    3 /* tournament_size: constante para Variante 2 básica */);
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
	free_cuda_context(ctx);
	free_gpu_instance(gi);

	return result;
}
