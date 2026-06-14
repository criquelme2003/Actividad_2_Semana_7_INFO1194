#pragma once

#include "models.hpp"
#include <random>
#include <utility>

// Genera un cromosoma aleatorio y lo repara para que cumpla las restricciones
// duras (incompatibilidades, dependencias y capacidad), de modo que la
// poblacion inicial nazca 100% factible.
Chromosome make_feasible_random_chromosome(const ProblemInstance &instance, std::mt19937 &rng);

std::pair<Chromosome, Chromosome>
crossover_one_point(const Chromosome &parent_a, const Chromosome &parent_b,
                    double crossover_rate, std::mt19937 &rng);

void mutate_bit_flip(Chromosome &chromosome, double mutation_rate,
                     std::mt19937 &rng);
