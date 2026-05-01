#include "selection.hpp"

#include <random>
#include <stdexcept>

using std::invalid_argument;
using std::size_t;
using std::uniform_int_distribution;
using std::vector;

size_t tournament_select_index(const vector<EvaluatedIndividual> &population, const int tournament_size,
                               std::mt19937 &rng) {
	if (population.empty()) {
		throw invalid_argument("La poblacion no puede estar vacia");
	}

	if (tournament_size <= 0) {
		throw invalid_argument("tournament_size debe ser mayor a cero");
	}

	uniform_int_distribution<size_t> pick_index(0, population.size() - 1);
	size_t best_index = pick_index(rng);

	for (int i = 1; i < tournament_size; ++i) {
		const size_t candidate_index = pick_index(rng);

		if (population[candidate_index].evaluation.fitness > population[best_index].evaluation.fitness) {
			best_index = candidate_index;
		}
	}

	return best_index;
}
