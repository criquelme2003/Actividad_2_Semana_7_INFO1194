#pragma once

#include <random>
#include <vector>

std::vector<double>
compute_rank_cumulative_probs(int rank_count, double p);

std::size_t
rank_select_index(const std::vector<double> &cumulative_probs,
                  std::mt19937 &rng);
