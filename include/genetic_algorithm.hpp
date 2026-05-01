#pragma once

#include "fitness.hpp"
#include "models.hpp"
#include <random>

class GeneticAlgorithm {
public:
	GeneticAlgorithm(const ProblemInstance &instance, GAConfig config,
	                 int seed);

	GARunResult run();

private:
	Chromosome make_random_chromosome();
	std::vector<Chromosome> make_initial_population();
	std::vector<EvaluatedIndividual>
	evaluate_population(const std::vector<Chromosome> &population) const;

	const ProblemInstance &instance_;
	GAConfig config_;
	FitnessEvaluator evaluator_;
	std::mt19937 rng_;
};
