#include "fitness.hpp"

#include "constants.hpp"
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

	// Valor normalizado: fracción del máximo valor posible de la instancia
	const double valor_norm = instance_.max_possible_value > 0.0
	                              ? out.total_value / instance_.max_possible_value
	                              : 0.0;

	// Máximos de cada tipo de violación (denominadores Pi_max)
	const double max_excess_w = instance_.total_weight - instance_.max_weight;
	const double max_excess_v = instance_.total_volume - instance_.max_volume;
	const double max_cat      = static_cast<double>(instance_.items.size());
	const double max_incompat = static_cast<double>(std::max(size_t{1}, instance_.incompatibility_indices.size()));
	const double max_dep      = static_cast<double>(std::max(size_t{1}, instance_.dependency_indices.size()));

	// Violaciones normalizadas Pi(x) / Pi_max, acotadas en [0, 1]
	const double p1 = std::min(1.0, excess_weight / max_excess_w);
	const double p2 = std::min(1.0, excess_volume / max_excess_v);
	const double p3 = std::min(1.0, static_cast<double>(out.category_violations)        / max_cat);
	const double p4 = instance_.incompatibility_indices.empty() ? 0.0
	                  : std::min(1.0, static_cast<double>(out.incompatibility_violations) / max_incompat);
	const double p5 = instance_.dependency_indices.empty() ? 0.0
	                  : std::min(1.0, static_cast<double>(out.dependency_violations)      / max_dep);

	const double violacion_norm = penalties_.w_weight   * p1
	                            + penalties_.w_volume    * p2
	                            + penalties_.w_category  * p3
	                            + penalties_.w_incompat  * p4
	                            + penalties_.w_dep       * p5;

	out.penalty = penalties_.beta * violacion_norm;
	out.fitness = penalties_.alpha * valor_norm - out.penalty;

	out.feasible_hard = (out.total_weight <= instance_.max_weight + constants::FEASIBILITY_EPSILON) &&
	                    (out.total_volume <= instance_.max_volume + constants::FEASIBILITY_EPSILON) &&
	                    out.incompatibility_violations == 0 && out.dependency_violations == 0;

	return out;
}
