#include "selection.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

using std::size_t;
using std::uniform_real_distribution;
using std::vector;

static vector<double> normalize(const vector<double> &values) {
	double sum = 0.0;
	for (auto v : values) {
		sum += v;
	}

	vector<double> normalized;
	normalized.reserve(values.size());
	for (auto v : values) {
		normalized.push_back(v / sum);
	}
	return normalized;
}

static vector<double> cumulative(const vector<double> &values) {
	vector<double> acc;
	acc.reserve(values.size());
	double running = 0.0;
	for (auto v : values) {
		running += v;
		acc.push_back(running);
	}
	return acc;
}

vector<double> compute_rank_cumulative_probs(const int rank_count, const double p) {
	if (rank_count <= 0) {
		throw std::invalid_argument("rank_count debe ser mayor a cero");
	}
	if (p <= 0.0 || p >= 1.0) {
		throw std::invalid_argument("p debe estar en (0, 1)");
	}

	vector<double> probs;
	probs.reserve(static_cast<size_t>(rank_count));

	for (int i = 0; i < rank_count; ++i) {
		probs.push_back(p * std::pow(1.0 - p, static_cast<double>(i)));
	}

	return cumulative(normalize(probs));
}

size_t rank_select_index(const vector<double> &cumulative_probs, std::mt19937 &rng) {
	if (cumulative_probs.empty()) {
		throw std::invalid_argument("cumulative_probs no puede estar vacio");
	}

	uniform_real_distribution<double> dist(0.0, 1.0);
	const double rnd = dist(rng);

	for (size_t i = 0; i < cumulative_probs.size(); ++i) {
		if (rnd < cumulative_probs[i]) {
			return i;
		}
	}

	return cumulative_probs.size() - 1;
}
