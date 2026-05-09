#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct Item {
	int id;
	double value;
	double weight;
	double volume;
	std::string category;
};

struct CategoryRule {
	std::string category;
	int minimum;
	int maximum;
};

struct PenaltyConfig {
	double alpha;      // α: peso del valor normalizado      (α + β = 1)
	double beta;       // β: peso de la violación normalizada
	double w_weight;   // w1: exceso de peso                 (Σwi = 1)
	double w_volume;   // w2: exceso de volumen
	double w_category; // w3: violaciones de categoría
	double w_incompat; // w4: incompatibilidades
	double w_dep;      // w5: dependencias incumplidas
};

struct ProblemInstance {
	std::vector<Item> items;
	std::vector<CategoryRule> category_rules;
	std::vector<std::pair<int, int>> incompatibilities;
	std::vector<std::pair<int, int>> dependencies;
	std::vector<std::pair<std::size_t, std::size_t>> incompatibility_indices;
	std::vector<std::pair<std::size_t, std::size_t>> dependency_indices;
	std::unordered_map<int, std::size_t> id_to_index;
	double max_weight;
	double max_volume;
	double total_weight;
	double total_volume;
	double max_possible_value;
};

struct Chromosome {
	std::vector<std::uint8_t> genes;
};

struct FitnessBreakdown {
	double total_value;
	double total_weight;
	double total_volume;
	int category_violations;
	int incompatibility_violations;
	int dependency_violations;
	double penalty;
	double fitness;
	bool feasible_hard;
};

struct EvaluatedIndividual {
	Chromosome chromosome;
	FitnessBreakdown evaluation;
};

struct GAConfig {
	int population_size;
	int generations;
	int tournament_size;
	int elitism_count;
	double crossover_rate;
	double mutation_rate;
	int max_stagnation_generations;
	PenaltyConfig penalties;
};

struct GARunResult {
	EvaluatedIndividual best_by_fitness;
	EvaluatedIndividual best_feasible;
	bool has_feasible;
	int generations_executed;
};
