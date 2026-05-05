#pragma once

#include "genetic_algorithm.hpp"
#include "models.hpp"
#include <vector>

// Variante 3: Modelo de islas.
//
// num_islands GAs secuenciales independientes evolucionan en paralelo
// (un hilo OpenMP por isla). Cada migration_interval generaciones las islas
// intercambian sus mejores individuos usando topología de anillo.
// Cada isla es un GeneticAlgorithm secuencial: el paralelismo es grueso
// (entre islas), no fino (dentro de cada isla).
class IslandModel {
public:
	IslandModel(const ProblemInstance &instance,
	            GAConfig island_config, // config compartida por todas las islas
	            int num_islands, int migration_interval, int migrants_per_island, int base_seed);

	GARunResult run();

private:
	// Topología de anillo: isla k envía sus mejores individuos a isla (k+1) % K.
	// Llamado secuencialmente tras la barrera implícita del parallel for.
	void perform_migration(std::vector<GeneticAlgorithm> &islands) const;

	const ProblemInstance &instance_;
	GAConfig island_config_;
	int num_islands_;
	int migration_interval_;
	int migrants_per_island_;
	int base_seed_;
};
