#pragma once

#include "models.hpp"
#include "parallel_genetic_algorithm.hpp"
#include <vector>

// Variante 3: Modelo de islas.
//
// K subpoblaciones evolucionan en paralelo (un hilo OpenMP por isla).
// Cada isla usa ParallelGeneticAlgorithm internamente (paralelismo anidado
// sobre los individuos). Cada migration_interval generaciones las islas
// intercambian sus mejores individuos usando topología de anillo.
class IslandModel {
public:
	IslandModel(const ProblemInstance &instance,
	            GAConfig island_config, // config compartida por todas las islas
	            int num_islands, int migration_interval, int migrants_per_island, int base_seed,
	            int threads_per_island); // hilos internos por isla

	GARunResult run();

private:
	// Topología de anillo: isla k envía sus mejores individuos a isla (k+1) % K.
	// Llamado secuencialmente tras la barrera implícita del parallel for.
	void perform_migration(std::vector<ParallelGeneticAlgorithm> &islands) const;

	const ProblemInstance &instance_;
	GAConfig island_config_;
	int num_islands_;
	int migration_interval_;
	int migrants_per_island_;
	int base_seed_;
	int threads_per_island_;
};
