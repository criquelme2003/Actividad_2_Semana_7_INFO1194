#include "operators.hpp"

#include <algorithm>

using std::move;
using std::size_t;
using std::swap;

// Desactiva el ítem idx y en cascada todos los que lo requerían (transitivamente).
static void cascade_deactivate(std::vector<std::uint8_t> &genes,
                               const std::vector<std::pair<std::size_t, std::size_t>> &dependency_indices,
                               std::size_t idx) {
	genes[idx] = 0U;
	bool changed = true;
	while (changed) {
		changed = false;
		for (const auto &[item_idx, required_idx] : dependency_indices) {
			if (genes[item_idx] == 1U && genes[required_idx] == 0U) {
				genes[item_idx] = 0U;
				changed = true;
			}
		}
	}
}

Chromosome make_feasible_random_chromosome(const ProblemInstance &instance, std::mt19937 &rng) {
	const size_t n = instance.items.size();
	Chromosome chromosome;
	chromosome.genes.resize(n, 0U);
	std::bernoulli_distribution bit(0.5);

	for (auto &gene : chromosome.genes) {
		gene = bit(rng) ? 1U : 0U;
	}

	// Paso 1 — incompatibilidades: desactivar uno del par y eliminar en cascada sus dependientes.
	// Hacerlo primero evita activar ítems requeridos que luego serían removidos aquí.
	for (const auto &[idx_a, idx_b] : instance.incompatibility_indices) {
		if (chromosome.genes[idx_a] == 1U && chromosome.genes[idx_b] == 1U) {
			std::bernoulli_distribution coin(0.5);
			cascade_deactivate(chromosome.genes, instance.dependency_indices,
			                   coin(rng) ? idx_a : idx_b);
		}
	}

	// Paso 2 — dependencias: eliminar en cascada ítems cuyos requeridos no están activos.
	// Solo se eliminan ítems (nunca se añaden) para no crear nuevas incompatibilidades.
	bool changed = true;
	while (changed) {
		changed = false;
		for (const auto &[item_idx, required_idx] : instance.dependency_indices) {
			if (chromosome.genes[item_idx] == 1U && chromosome.genes[required_idx] == 0U) {
				chromosome.genes[item_idx] = 0U;
				changed = true;
			}
		}
	}

	// Paso 3 — capacidad: cascade_deactivate en orden aleatorio hasta cumplir peso y volumen.
	// Se usa cascade_deactivate para evitar dejar dependientes huérfanos cuando se elimina un ítem.
	std::vector<size_t> selected;
	selected.reserve(n);
	for (size_t i = 0; i < n; ++i) {
		if (chromosome.genes[i] == 1U) {
			selected.push_back(i);
		}
	}
	std::shuffle(selected.begin(), selected.end(), rng);

	auto compute_totals = [&]() {
		double w = 0.0, v = 0.0;
		for (size_t i = 0; i < n; ++i) {
			if (chromosome.genes[i] == 1U) {
				w += instance.items[i].weight;
				v += instance.items[i].volume;
			}
		}
		return std::pair{w, v};
	};

	for (size_t i : selected) {
		auto [tw, tv] = compute_totals();
		if (tw <= instance.max_weight && tv <= instance.max_volume) {
			break;
		}
		if (chromosome.genes[i] == 1U) {
			cascade_deactivate(chromosome.genes, instance.dependency_indices, i);
		}
	}

	return chromosome;
}

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
	std::uniform_real_distribution<double> coin(0.0, 1.0);
	for (auto &gene : chromosome.genes) {
		if (coin(rng) <= mutation_rate) {
			gene = (gene == 0U) ? 1U : 0U;
		}
	}
}
