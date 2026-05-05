#include "genetic_algorithm.hpp"

#include "operators.hpp"
#include "selection.hpp"
#include <algorithm>
#include <stdexcept>

using std::invalid_argument;
using std::move;
using std::size_t;
using std::sort;
using std::vector;

// Constructor: recibe el problema a resolver, la configuración del GA y una semilla
// para el generador de números aleatorios (reproducibilidad).
// Valida que los parámetros sean coherentes antes de ejecutar cualquier cosa.
GeneticAlgorithm::GeneticAlgorithm(const ProblemInstance &instance, GAConfig config, const int seed)
    : instance_(instance), config_(config), evaluator_(instance, config.penalties), rng_(seed) {

	// Sin ítems no hay problema que resolver
	if (instance_.items.empty()) {
		throw invalid_argument("La instancia debe tener al menos un item");
	}

	// Con 1 solo individuo no hay cruce posible
	if (config_.population_size <= 1) {
		throw invalid_argument("population_size debe ser mayor a 1");
	}

	if (config_.generations <= 0) {
		throw invalid_argument("generations debe ser mayor a 0");
	}

	// El torneo debe tener al menos 1 participante y no más que la población completa
	if (config_.tournament_size <= 0 || config_.tournament_size > config_.population_size) {
		throw invalid_argument("tournament_size invalido");
	}

	// El elitismo no puede vaciar la población (debe quedar espacio para hijos)
	if (config_.elitism_count < 0 || config_.elitism_count >= config_.population_size) {
		throw invalid_argument("elitism_count invalido");
	}

	// Tasas de probabilidad: deben estar en el rango [0, 1]
	if (config_.crossover_rate < 0.0 || config_.crossover_rate > 1.0) {
		throw invalid_argument("crossover_rate debe estar en [0,1]");
	}

	if (config_.mutation_rate < 0.0 || config_.mutation_rate > 1.0) {
		throw invalid_argument("mutation_rate debe estar en [0,1]");
	}
}

// Crea un individuo con genes aleatorios.
// Cada gen representa si el ítem i está incluido (1) o no (0) en la solución.
// Se usa distribución de Bernoulli con p=0.5: cada bit tiene igual probabilidad.
Chromosome GeneticAlgorithm::make_random_chromosome() {
	Chromosome chromosome;

	// Un gen por ítem del problema
	chromosome.genes.resize(instance_.items.size(), 0U);
	std::bernoulli_distribution bit(0.5);

	for (auto &gene : chromosome.genes) {
		gene = bit(rng_) ? 1U : 0U;
	}

	return chromosome;
}

// Genera la población inicial: un conjunto de individuos completamente aleatorios.
// Esta diversidad inicial es clave para que el GA explore bien el espacio de soluciones.
vector<Chromosome> GeneticAlgorithm::make_initial_population() {
	vector<Chromosome> population;
	population.reserve(static_cast<size_t>(config_.population_size));

	for (int i = 0; i < config_.population_size; ++i) {
		population.push_back(make_random_chromosome());
	}

	return population;
}

// Evalúa cada individuo de la población usando la función de fitness del evaluador.
// Devuelve pares (cromosoma, evaluación) para no perder el vínculo entre ambos.
vector<EvaluatedIndividual> GeneticAlgorithm::evaluate_population(const vector<Chromosome> &population) const {
	vector<EvaluatedIndividual> evaluated;
	evaluated.reserve(population.size());

	for (const auto &individual : population) {
		evaluated.push_back(EvaluatedIndividual{individual, evaluator_.evaluate(individual)});
	}

	return evaluated;
}

// Bucle principal del Algoritmo Genético.
// Cada iteración representa una generación: evaluar → seleccionar → cruzar → mutar.
// Retorna el mejor resultado encontrado (por fitness y el mejor factible).
GARunResult GeneticAlgorithm::run() {
	// --- INICIALIZACIÓN ---
	if (population_.empty()) {
		population_ = make_initial_population();
	}
	has_feasible_ = false;

	GARunResult result{};
	result.has_feasible = false;
	int stagnation_counter = 0;

	for (int generation = 0; generation < config_.generations; ++generation) {

		// --- EVALUACIÓN ---
		auto evaluated = evaluate_population(population_);

		sort(evaluated.begin(), evaluated.end(), [](const EvaluatedIndividual &a, const EvaluatedIndividual &b) {
			return a.evaluation.fitness > b.evaluation.fitness;
		});

		// --- ACTUALIZACIÓN DEL MEJOR GLOBAL ---
		if (generation == 0 || evaluated.front().evaluation.fitness > result.best_by_fitness.evaluation.fitness) {
			result.best_by_fitness = evaluated.front();
			best_by_fitness_ = evaluated.front();
			stagnation_counter = 0;
		} else {
			stagnation_counter++;
		}

		// --- MEJOR SOLUCIÓN FACTIBLE ---
		for (const auto &individual : evaluated) {
			if (!individual.evaluation.feasible_hard) {
				continue;
			}
			if (!result.has_feasible || individual.evaluation.fitness > result.best_feasible.evaluation.fitness) {
				result.best_feasible = individual;
				best_feasible_ = individual;
				result.has_feasible = true;
				has_feasible_ = true;
			}
			break;
		}

		// --- CRITERIO DE PARADA TEMPRANA (estancamiento) ---
		if (stagnation_counter >= config_.max_stagnation_generations) {
			result.generations_executed = generation + 1;
			return result;
		}

		// --- CONSTRUCCIÓN DE LA NUEVA GENERACIÓN ---
		vector<Chromosome> next_population;
		next_population.reserve(population_.size());

		for (int elite = 0; elite < config_.elitism_count; ++elite) {
			next_population.push_back(evaluated[static_cast<size_t>(elite)].chromosome);
		}

		while (next_population.size() < population_.size()) {
			const size_t idx_a = tournament_select_index(evaluated, config_.tournament_size, rng_);
			const size_t idx_b = tournament_select_index(evaluated, config_.tournament_size, rng_);

			auto [child_a, child_b] = crossover_one_point(evaluated[idx_a].chromosome, evaluated[idx_b].chromosome,
			                                              config_.crossover_rate, rng_);

			mutate_bit_flip(child_a, config_.mutation_rate, rng_);
			mutate_bit_flip(child_b, config_.mutation_rate, rng_);

			next_population.push_back(move(child_a));
			if (next_population.size() < population_.size()) {
				next_population.push_back(move(child_b));
			}
		}

		population_ = move(next_population);
		result.generations_executed = generation + 1;
	}

	return result;
}

// ---------------------------------------------------------------------------
// Interfaz para IslandModel (Variante 3)
// ---------------------------------------------------------------------------

// Evoluciona exactamente n generaciones sin criterio de parada temprana.
// Si la población está vacía, la inicializa primero.
// Actualiza best_by_fitness_ y best_feasible_ en cada generación.
void GeneticAlgorithm::run_n_generations(const int n) {
	if (population_.empty()) {
		population_ = make_initial_population();
	}

	for (int g = 0; g < n; ++g) {
		auto evaluated = evaluate_population(population_);

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

		while (next_population.size() < population_.size()) {
			const size_t idx_a = tournament_select_index(evaluated, config_.tournament_size, rng_);
			const size_t idx_b = tournament_select_index(evaluated, config_.tournament_size, rng_);

			auto [child_a, child_b] = crossover_one_point(evaluated[idx_a].chromosome, evaluated[idx_b].chromosome,
			                                              config_.crossover_rate, rng_);

			mutate_bit_flip(child_a, config_.mutation_rate, rng_);
			mutate_bit_flip(child_b, config_.mutation_rate, rng_);

			next_population.push_back(move(child_a));
			if (next_population.size() < population_.size()) {
				next_population.push_back(move(child_b));
			}
		}

		population_ = move(next_population);
	}
}

// Reemplaza los peores individuos de la población con los migrantes recibidos.
void GeneticAlgorithm::inject_individuals(const vector<Chromosome> &migrants) {
	if (population_.empty() || migrants.empty()) {
		return;
	}

	auto evaluated = evaluate_population(population_);
	sort(evaluated.begin(), evaluated.end(), [](const EvaluatedIndividual &a, const EvaluatedIndividual &b) {
		return a.evaluation.fitness > b.evaluation.fitness;
	});

	const size_t replace_count = std::min(migrants.size(), population_.size());
	const size_t pop_size = population_.size();

	for (size_t i = 0; i < pop_size; ++i) {
		population_[i] = evaluated[i].chromosome;
	}

	for (size_t i = 0; i < replace_count; ++i) {
		population_[pop_size - 1 - i] = migrants[i];
	}
}

EvaluatedIndividual GeneticAlgorithm::get_best() const { return best_by_fitness_; }

EvaluatedIndividual GeneticAlgorithm::get_best_feasible() const { return best_feasible_; }

bool GeneticAlgorithm::has_feasible_solution() const { return has_feasible_; }

// Evalúa la población actual, la ordena y devuelve los n mejores cromosomas.
vector<Chromosome> GeneticAlgorithm::get_top_n(const int n) const {
	if (population_.empty() || n <= 0) {
		return {};
	}

	auto evaluated = evaluate_population(population_);
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
