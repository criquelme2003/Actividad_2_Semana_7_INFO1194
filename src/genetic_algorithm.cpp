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

GeneticAlgorithm::GeneticAlgorithm(const ProblemInstance &instance, GAConfig config, const int seed)
    : instance_(instance), config_(config), evaluator_(instance, config.penalties), rng_(seed) {

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
}

Chromosome GeneticAlgorithm::make_random_chromosome() {
	Chromosome chromosome;

	chromosome.genes.resize(instance_.items.size(), 0U);
	std::bernoulli_distribution bit(0.5);

	for (auto &gene : chromosome.genes) {
		gene = bit(rng_) ? 1U : 0U;
	}

	return chromosome;
}

vector<Chromosome> GeneticAlgorithm::make_initial_population() {
	vector<Chromosome> population;
	population.reserve(static_cast<size_t>(config_.population_size));

	for (int i = 0; i < config_.population_size; ++i) {
		population.push_back(make_random_chromosome());
	}

	return population;
}

vector<EvaluatedIndividual> GeneticAlgorithm::evaluate_population(const vector<Chromosome> &population) const {
	vector<EvaluatedIndividual> evaluated;
	evaluated.reserve(population.size());

	for (const auto &individual : population) {
		evaluated.push_back(EvaluatedIndividual{individual, evaluator_.evaluate(individual)});
	}

	return evaluated;
}

GARunResult GeneticAlgorithm::run() {
	vector<Chromosome> population = make_initial_population();

	GARunResult result{};
	result.has_feasible = false;
	int stagnation_counter = 0;

	for (int generation = 0; generation < config_.generations; ++generation) {
		auto evaluated = evaluate_population(population);
		sort(evaluated.begin(), evaluated.end(), [](const EvaluatedIndividual &a, const EvaluatedIndividual &b) {
			return a.evaluation.fitness > b.evaluation.fitness;
		});

		if (generation == 0 || evaluated.front().evaluation.fitness > result.best_by_fitness.evaluation.fitness) {
			result.best_by_fitness = evaluated.front();
			stagnation_counter = 0;
		} else {
			stagnation_counter++;
		}

		for (const auto &individual : evaluated) {
			if (!individual.evaluation.feasible_hard) {
				continue;
			}
			if (!result.has_feasible || individual.evaluation.fitness > result.best_feasible.evaluation.fitness) {
				result.best_feasible = individual;
				result.has_feasible = true;
			}
			break;
		}

		if (stagnation_counter >= config_.max_stagnation_generations) {
			result.generations_executed = generation + 1;
			return result;
		}

		vector<Chromosome> next_population;
		next_population.reserve(population.size());

		for (int elite = 0; elite < config_.elitism_count; ++elite) {
			next_population.push_back(evaluated[static_cast<size_t>(elite)].chromosome);
		}

		while (next_population.size() < population.size()) {
			const size_t idx_a = tournament_select_index(evaluated, config_.tournament_size, rng_);
			const size_t idx_b = tournament_select_index(evaluated, config_.tournament_size, rng_);

			auto [child_a, child_b] = crossover_one_point(evaluated[idx_a].chromosome, evaluated[idx_b].chromosome,
			                                              config_.crossover_rate, rng_);

			mutate_bit_flip(child_a, config_.mutation_rate, rng_);
			mutate_bit_flip(child_b, config_.mutation_rate, rng_);

			next_population.push_back(move(child_a));
			if (next_population.size() < population.size()) {
				next_population.push_back(move(child_b));
			}
		}

		population = move(next_population);
		result.generations_executed = generation + 1;
	}

	return result;
}
