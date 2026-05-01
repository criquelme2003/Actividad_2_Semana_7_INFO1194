#pragma once

#include "models.hpp"
#include <random>
#include <vector>

std::size_t
tournament_select_index(const std::vector<EvaluatedIndividual> &population,
                        int tournament_size, std::mt19937 &rng);
