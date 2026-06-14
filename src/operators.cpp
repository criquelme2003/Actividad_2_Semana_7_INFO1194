#include "operators.hpp"

#include <algorithm>

using std::move;
using std::size_t;
using std::swap;

// Desactiva el item idx y, en cascada, todos los que lo requerian
// (transitivamente), manteniendo current_weight/current_volume al dia para
// evitar recalcular los totales sobre toda la instancia en cada paso.
static void cascade_deactivate(std::vector<std::uint8_t> &genes, const ProblemInstance &instance, size_t idx,
                               double &current_weight, double &current_volume) {
	if (genes[idx] == 0U) {
		return;
	}

	genes[idx] = 0U;
	current_weight -= instance.items[idx].weight;
	current_volume -= instance.items[idx].volume;

	bool changed = true;
	while (changed) {
		changed = false;
		for (const auto &[item_idx, required_idx] : instance.dependency_indices) {
			if (genes[item_idx] == 1U && genes[required_idx] == 0U) {
				genes[item_idx] = 0U;
				current_weight -= instance.items[item_idx].weight;
				current_volume -= instance.items[item_idx].volume;
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

	double current_weight = 0.0;
	double current_volume = 0.0;
	for (size_t i = 0; i < n; ++i) {
		chromosome.genes[i] = bit(rng) ? 1U : 0U;
		if (chromosome.genes[i] == 1U) {
			current_weight += instance.items[i].weight;
			current_volume += instance.items[i].volume;
		}
	}

	// Paso 1 - incompatibilidades: desactivar uno del par y eliminar en cascada sus dependientes.
	// Hacerlo primero evita activar items requeridos que luego serian removidos aqui.
	std::bernoulli_distribution coin(0.5);
	for (const auto &[idx_a, idx_b] : instance.incompatibility_indices) {
		if (chromosome.genes[idx_a] == 1U && chromosome.genes[idx_b] == 1U) {
			cascade_deactivate(chromosome.genes, instance, coin(rng) ? idx_a : idx_b, current_weight, current_volume);
		}
	}

	// Paso 2 - dependencias: eliminar en cascada items cuyos requeridos no estan activos.
	// Solo se eliminan items (nunca se anaden) para no crear nuevas incompatibilidades.
	bool changed = true;
	while (changed) {
		changed = false;
		for (const auto &[item_idx, required_idx] : instance.dependency_indices) {
			if (chromosome.genes[item_idx] == 1U && chromosome.genes[required_idx] == 0U) {
				chromosome.genes[item_idx] = 0U;
				current_weight -= instance.items[item_idx].weight;
				current_volume -= instance.items[item_idx].volume;
				changed = true;
			}
		}
	}

	// Paso 3 - capacidad: eliminar items en orden aleatorio (con cascada de
	// dependientes) hasta cumplir peso y volumen.
	std::vector<size_t> selected;
	selected.reserve(n);
	for (size_t i = 0; i < n; ++i) {
		if (chromosome.genes[i] == 1U) {
			selected.push_back(i);
		}
	}
	std::shuffle(selected.begin(), selected.end(), rng);

	for (size_t i : selected) {
		if (current_weight <= instance.max_weight && current_volume <= instance.max_volume) {
			break;
		}
		cascade_deactivate(chromosome.genes, instance, i, current_weight, current_volume);
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
