#pragma once

#include "fitness.hpp"
#include "models.hpp"
#include <random>

class GeneticAlgorithm {
public:
	GeneticAlgorithm(const ProblemInstance &instance, GAConfig config, int seed);

	GARunResult run();

	// Interfaz para IslandModel (Variante 3): cada isla es un GA secuencial
	// independiente que el modelo de islas evoluciona batch a batch.
	void run_n_generations(int n);
	void inject_individuals(const std::vector<Chromosome> &migrants);
	EvaluatedIndividual get_best() const;
	EvaluatedIndividual get_best_feasible() const;
	bool has_feasible_solution() const;
	std::vector<Chromosome> get_top_n(int n) const;

private:
	Chromosome make_random_chromosome();
	std::vector<Chromosome> make_initial_population();
	std::vector<EvaluatedIndividual> evaluate_population(const std::vector<Chromosome> &population) const;

	const ProblemInstance &instance_;
	GAConfig config_;
	FitnessEvaluator evaluator_;
	std::mt19937 rng_;

	// Estado persistente: permite que run_n_generations mantenga la población
	// entre llamadas sucesivas (necesario para el modelo de islas).
	std::vector<Chromosome> population_;
	EvaluatedIndividual best_by_fitness_;
	EvaluatedIndividual best_feasible_;
	bool has_feasible_ = false;
};
