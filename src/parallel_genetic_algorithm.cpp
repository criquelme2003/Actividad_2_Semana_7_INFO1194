#include "parallel_genetic_algorithm.hpp"

#include "operators.hpp"
#include "selection.hpp"
#include <algorithm>
#include <omp.h>
#include <stdexcept>

using std::invalid_argument;
using std::move;
using std::size_t;
using std::sort;
using std::vector;

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

ParallelGeneticAlgorithm::ParallelGeneticAlgorithm(const ProblemInstance &instance, GAConfig config, const int seed,
                                                   const int num_threads)
    : instance_(instance), config_(config), evaluator_(instance, config.penalties), num_threads_(num_threads),
      master_rng_(seed) {

	if (instance_.items.empty()) {
		throw invalid_argument("La instancia debe tener al menos un item");
	}
	if (config_.population_size <= 1) {
		throw invalid_argument("population_size debe ser mayor a 1");
	}
	if (config_.generations <= 0) {
		throw invalid_argument("generations debe ser mayor a 0");
	}
	if (config_.tournament_size <= 0 || config_.tournament_size > config_.population_size) {
		throw invalid_argument("tournament_size invalido");
	}
	if (config_.elitism_count < 0 || config_.elitism_count >= config_.population_size) {
		throw invalid_argument("elitism_count invalido");
	}
	if (config_.crossover_rate < 0.0 || config_.crossover_rate > 1.0) {
		throw invalid_argument("crossover_rate debe estar en [0,1]");
	}
	if (config_.mutation_rate < 0.0 || config_.mutation_rate > 1.0) {
		throw invalid_argument("mutation_rate debe estar en [0,1]");
	}
	if (num_threads_ <= 0) {
		throw invalid_argument("num_threads debe ser mayor a 0");
	}

	// Un RNG por hilo, sembrado deterministicamente para reproducibilidad.
	// La constante 1000003 (primo) separa las secuencias MT de distintos hilos.
	thread_rngs_.reserve(static_cast<size_t>(num_threads_));
	for (int t = 0; t < num_threads_; ++t) {
		thread_rngs_.emplace_back(static_cast<uint32_t>(seed + 1 + t * 1000003));
	}
}

// ---------------------------------------------------------------------------
// Inicialización de población
// ---------------------------------------------------------------------------

Chromosome ParallelGeneticAlgorithm::make_random_chromosome() {
	Chromosome chromosome;
	chromosome.genes.resize(instance_.items.size(), 0U);
	std::bernoulli_distribution bit(0.5);
	for (auto &gene : chromosome.genes) {
		gene = bit(master_rng_) ? 1U : 0U;
	}
	return chromosome;
}

vector<Chromosome> ParallelGeneticAlgorithm::make_initial_population() {
	vector<Chromosome> population;
	population.reserve(static_cast<size_t>(config_.population_size));
	for (int i = 0; i < config_.population_size; ++i) {
		population.push_back(make_random_chromosome());
	}
	return population;
}

// ---------------------------------------------------------------------------
// Paralelización 1: evaluación de aptitud
//
// Datos compartidos (solo lectura):  population_, evaluator_
// Datos privados por hilo:           FitnessBreakdown out, category_counts, índice i
// Escritura sin carrera:             evaluated[i], pre-alocado, índice exclusivo por hilo
// Schedule static: carga uniforme — cada evaluate() recorre los mismos ítems
// ---------------------------------------------------------------------------

vector<EvaluatedIndividual> ParallelGeneticAlgorithm::evaluate_population_parallel() const {
	const auto pop_size = static_cast<int>(population_.size());
	vector<EvaluatedIndividual> evaluated(static_cast<size_t>(pop_size));

#pragma omp parallel for schedule(static) num_threads(num_threads_)
	for (int i = 0; i < pop_size; ++i) {
		evaluated[static_cast<size_t>(i)] = {population_[static_cast<size_t>(i)],
		                                     evaluator_.evaluate(population_[static_cast<size_t>(i)])};
	}

	return evaluated;
}

// ---------------------------------------------------------------------------
// Paralelización 2: selección por torneo + cruce + mutación
//
// Cada iteración p genera un par de hijos de forma completamente independiente:
//   1. Selección: tournament_select_index lee evaluated (solo lectura) y usa
//      thread_rngs_[tid] (RNG privado al hilo) → sin carrera.
//   2. Cruce: crossover_one_point lee evaluated (solo lectura), escribe locales.
//   3. Mutación: mutate_bit_flip modifica locales con el RNG privado.
//   4. Escritura: buf[2p] y buf[2p+1] son índices exclusivos del par p → sin carrera.
//
// Caso impar: se genera num_pairs = ceil(offspring_needed/2) pares; el hijo
// sobrante se elimina con resize() al final.
// ---------------------------------------------------------------------------

vector<Chromosome> ParallelGeneticAlgorithm::generate_offspring_parallel(const vector<EvaluatedIndividual> &evaluated,
                                                                         const int offspring_needed) {

	const int num_pairs = (offspring_needed + 1) / 2;
	vector<Chromosome> buf(static_cast<size_t>(num_pairs) * 2);

#pragma omp parallel for schedule(static) num_threads(num_threads_)
	for (int p = 0; p < num_pairs; ++p) {
		// RNG privado al hilo — no compartido entre iteraciones
		auto &rng = thread_rngs_[static_cast<size_t>(omp_get_thread_num())];

		// Selección por torneo (paralela): cada hilo selecciona sus dos padres
		// usando su RNG privado; evaluated es solo lectura → sin carrera
		const size_t idx_a = tournament_select_index(evaluated, config_.tournament_size, rng);
		const size_t idx_b = tournament_select_index(evaluated, config_.tournament_size, rng);

		// Cruce one-point (paralelo): produce dos hijos locales al hilo
		auto [child_a, child_b] =
		    crossover_one_point(evaluated[idx_a].chromosome, evaluated[idx_b].chromosome, config_.crossover_rate, rng);

		// Mutación bit-flip (paralela): modifica los hijos locales con el RNG privado
		mutate_bit_flip(child_a, config_.mutation_rate, rng);
		mutate_bit_flip(child_b, config_.mutation_rate, rng);

		// Escritura en índices exclusivos del par p → sin carrera
		buf[static_cast<size_t>(p) * 2] = move(child_a);
		buf[static_cast<size_t>(p) * 2 + 1] = move(child_b);
	}

	// Descarta el hijo extra si offspring_needed era impar
	buf.resize(static_cast<size_t>(offspring_needed));
	return buf;
}

// ---------------------------------------------------------------------------
// Bucle evolutivo principal
// ---------------------------------------------------------------------------

// Ejecuta una generación completa sobre population_ y actualiza los campos
// best_by_fitness_, best_feasible_, has_feasible_.
// Retorna true si se alcanzó el criterio de estancamiento.
static bool run_one_generation_impl(
    ParallelGeneticAlgorithm &self, std::vector<EvaluatedIndividual> (ParallelGeneticAlgorithm::*eval_fn)() const,
    std::vector<Chromosome> (ParallelGeneticAlgorithm::*offspring_fn)(const std::vector<EvaluatedIndividual> &, int),
    std::vector<Chromosome> &population, EvaluatedIndividual &best_by_fitness, EvaluatedIndividual &best_feasible,
    bool &has_feasible, int &stagnation_counter, const GAConfig &config);

GARunResult ParallelGeneticAlgorithm::run() {
	omp_set_num_threads(num_threads_);

	if (population_.empty()) {
		population_ = make_initial_population();
	}

	GARunResult result{};
	result.has_feasible = false;
	int stagnation_counter = 0;

	for (int generation = 0; generation < config_.generations; ++generation) {

		// --- EVALUACIÓN PARALELA ---
		auto evaluated = evaluate_population_parallel();

		// --- ORDENAMIENTO (secuencial): O(N log N) sobre ~120 floats, negligible ---
		sort(evaluated.begin(), evaluated.end(), [](const EvaluatedIndividual &a, const EvaluatedIndividual &b) {
			return a.evaluation.fitness > b.evaluation.fitness;
		});

		// --- TRACKING DEL MEJOR (secuencial): escritura única sobre el resultado ---
		if (generation == 0 || evaluated.front().evaluation.fitness > result.best_by_fitness.evaluation.fitness) {
			result.best_by_fitness = evaluated.front();
			best_by_fitness_ = evaluated.front();
			stagnation_counter = 0;
		} else {
			stagnation_counter++;
		}

		for (const auto &ind : evaluated) {
			if (!ind.evaluation.feasible_hard) {
				continue;
			}
			if (!result.has_feasible || ind.evaluation.fitness > result.best_feasible.evaluation.fitness) {
				result.best_feasible = ind;
				best_feasible_ = ind;
				result.has_feasible = true;
				has_feasible_ = true;
			}
			break;
		}

		// --- CRITERIO DE PARADA POR ESTANCAMIENTO ---
		if (stagnation_counter >= config_.max_stagnation_generations) {
			result.generations_executed = generation + 1;
			return result;
		}

		// --- ELITISMO (secuencial): elitism_count típico = 2, overhead de OMP > trabajo ---
		vector<Chromosome> next_population;
		next_population.reserve(population_.size());
		for (int e = 0; e < config_.elitism_count; ++e) {
			next_population.push_back(evaluated[static_cast<size_t>(e)].chromosome);
		}

		// --- GENERACIÓN DE OFFSPRING PARALELA (selección + cruce + mutación) ---
		const int offspring_needed = static_cast<int>(population_.size()) - config_.elitism_count;
		auto offspring = generate_offspring_parallel(evaluated, offspring_needed);

		for (auto &ch : offspring) {
			next_population.push_back(move(ch));
		}

		population_ = move(next_population);
		result.generations_executed = generation + 1;
	}

	return result;
}

// ---------------------------------------------------------------------------
// Interfaz para Variante 3 (IslandModel)
// ---------------------------------------------------------------------------

void ParallelGeneticAlgorithm::run_n_generations(const int n) {
	omp_set_num_threads(num_threads_);

	if (population_.empty()) {
		population_ = make_initial_population();
	}

	for (int g = 0; g < n; ++g) {
		auto evaluated = evaluate_population_parallel();

		sort(evaluated.begin(), evaluated.end(), [](const EvaluatedIndividual &a, const EvaluatedIndividual &b) {
			return a.evaluation.fitness > b.evaluation.fitness;
		});

		best_by_fitness_ = evaluated.front();

		for (const auto &ind : evaluated) {
			if (!ind.evaluation.feasible_hard) {
				continue;
			}
			if (!has_feasible_ || ind.evaluation.fitness > best_feasible_.evaluation.fitness) {
				best_feasible_ = ind;
				has_feasible_ = true;
			}
			break;
		}

		vector<Chromosome> next_population;
		next_population.reserve(population_.size());

		for (int e = 0; e < config_.elitism_count; ++e) {
			next_population.push_back(evaluated[static_cast<size_t>(e)].chromosome);
		}

		const int offspring_needed = static_cast<int>(population_.size()) - config_.elitism_count;
		auto offspring = generate_offspring_parallel(evaluated, offspring_needed);

		for (auto &ch : offspring) {
			next_population.push_back(move(ch));
		}

		population_ = move(next_population);
	}
}

// Reemplaza los peores individuos de la población con los migrantes recibidos.
// Se llama desde IslandModel tras la barrera de sincronización entre batches.
void ParallelGeneticAlgorithm::inject_individuals(const vector<Chromosome> &migrants) {
	if (population_.empty() || migrants.empty()) {
		return;
	}

	// Evaluar y ordenar la población actual para identificar los peores
	auto evaluated = evaluate_population_parallel();
	sort(evaluated.begin(), evaluated.end(), [](const EvaluatedIndividual &a, const EvaluatedIndividual &b) {
		return a.evaluation.fitness > b.evaluation.fitness;
	});

	// Reemplazar los últimos (peores) individuos con los migrantes
	const size_t replace_count = std::min(migrants.size(), population_.size());
	const size_t pop_size = population_.size();

	// Reconstruir population_ a partir del orden evaluado
	for (size_t i = 0; i < pop_size; ++i) {
		population_[i] = evaluated[i].chromosome;
	}

	// Los peores están al final del vector ordenado
	for (size_t i = 0; i < replace_count; ++i) {
		population_[pop_size - 1 - i] = migrants[i];
	}
}

EvaluatedIndividual ParallelGeneticAlgorithm::get_best() const { return best_by_fitness_; }

EvaluatedIndividual ParallelGeneticAlgorithm::get_best_feasible() const { return best_feasible_; }

bool ParallelGeneticAlgorithm::has_feasible_solution() const { return has_feasible_; }

// Evalúa population_ (paralelo), ordena por fitness y extrae los n primeros cromosomas.
// Llamado desde IslandModel::perform_migration en el hilo principal (sin concurrencia).
vector<Chromosome> ParallelGeneticAlgorithm::get_top_n(const int n) const {
	if (population_.empty() || n <= 0) {
		return {};
	}

	auto evaluated = evaluate_population_parallel();
	sort(evaluated.begin(), evaluated.end(), [](const EvaluatedIndividual &a, const EvaluatedIndividual &b) {
		return a.evaluation.fitness > b.evaluation.fitness;
	});

	const int count = std::min(n, static_cast<int>(evaluated.size()));
	vector<Chromosome> top;
	top.reserve(static_cast<size_t>(count));
	for (int i = 0; i < count; ++i) {
		top.push_back(evaluated[static_cast<size_t>(i)].chromosome);
	}
	return top;
}
