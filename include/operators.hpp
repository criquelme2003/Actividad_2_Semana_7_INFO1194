#pragma once

#include "models.hpp"
#include <random>
#include <utility>

std::pair<Chromosome, Chromosome>
crossover_one_point(const Chromosome &parent_a, const Chromosome &parent_b,
                    double crossover_rate, std::mt19937 &rng);

void mutate_bit_flip(Chromosome &chromosome, double mutation_rate,
                     std::mt19937 &rng);
