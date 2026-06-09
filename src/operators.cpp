#include "operators.hpp"

#include <algorithm>

using std::move;
using std::size_t;
using std::swap;

std::pair<Chromosome, Chromosome> crossover_one_point(const Chromosome &parent_a, const Chromosome &parent_b,
                                                      const double crossover_rate, std::mt19937 &rng) {
	Chromosome child_a = parent_a;
	Chromosome child_b = parent_b;

	std::uniform_real_distribution<double> coin(0.0, 1.0);

	if (coin(rng) > crossover_rate || parent_a.genes.size() < 2) {
		return {move(child_a), move(child_b)};
	}

	std::uniform_int_distribution<size_t> cross_point(1, parent_a.genes.size() - 1);
	const size_t point = cross_point(rng);
	for (size_t i = point; i < parent_a.genes.size(); ++i) {
		swap(child_a.genes[i], child_b.genes[i]);
	}

	return {move(child_a), move(child_b)};
}

void mutate_bit_flip(Chromosome &chromosome, const double mutation_rate, std::mt19937 &rng) {
	if (chromosome.genes.empty()) {
		return;
	}

	std::uniform_real_distribution<double> coin(0.0, 1.0);
	if (coin(rng) > mutation_rate) {
		return;
	}

	std::uniform_int_distribution<size_t> pick(0, chromosome.genes.size() - 1);
	const size_t idx = pick(rng);
	chromosome.genes[idx] = (chromosome.genes[idx] == 0U) ? 1U : 0U;
}
