#include "fitness.hpp"

#include <algorithm>
#include <unordered_map>

using std::max;
using std::size_t;
using std::string;
using std::unordered_map;

FitnessEvaluator::FitnessEvaluator(const ProblemInstance &instance, PenaltyConfig penalties)
    : instance_(instance), penalties_(penalties) {}

FitnessBreakdown FitnessEvaluator::evaluate(const Chromosome &chromosome) const {
	FitnessBreakdown out{};
	if (chromosome.genes.size() != instance_.items.size()) {
		return out;
	}

	unordered_map<string, int> category_counts;

	for (size_t i = 0; i < chromosome.genes.size(); ++i) {
		if (chromosome.genes[i] == 0U) {
			continue;
		}

		const Item &item = instance_.items[i];

		out.total_value += item.value;
		out.total_weight += item.weight;
		out.total_volume += item.volume;

		category_counts[item.category]++;
	}

	const double excess_weight = max(0.0, out.total_weight - instance_.max_weight);

	const double excess_volume = max(0.0, out.total_volume - instance_.max_volume);

	for (const auto &rule : instance_.category_rules) {
		const int count = category_counts[rule.category];

		if (count < rule.minimum) {
			out.category_violations += (rule.minimum - count);
		}

		if (rule.maximum >= 0 && count > rule.maximum) {
			out.category_violations += (count - rule.maximum);
		}
	}

	for (const auto &[index_a, index_b] : instance_.incompatibility_indices) {
		if (chromosome.genes[index_a] == 1U && chromosome.genes[index_b] == 1U) {
			out.incompatibility_violations++;
		}
	}

	for (const auto &[item_index, required_index] : instance_.dependency_indices) {
		if (chromosome.genes[item_index] == 1U && chromosome.genes[required_index] == 0U) {
			out.dependency_violations++;
		}
	}

	out.penalty = penalties_.alpha * excess_weight + penalties_.beta * excess_volume +
	              penalties_.gamma * static_cast<double>(out.category_violations) +
	              penalties_.delta * static_cast<double>(out.incompatibility_violations) +
	              penalties_.epsilon * static_cast<double>(out.dependency_violations);

	out.fitness = out.total_value - out.penalty;

	out.feasible_hard = (excess_weight <= 0.0) && (excess_volume <= 0.0) && out.incompatibility_violations == 0 &&
	                    out.dependency_violations == 0;

	return out;
}
