#include "island_model.hpp"

#include <algorithm>
#include <omp.h>
#include <utility>

using std::size_t;
using std::vector;

IslandModel::IslandModel(const ProblemInstance &instance, GAConfig island_config, const int num_islands,
                         const int migration_interval, const int migrants_per_island, const int base_seed,
                         const int threads_per_island)
    : instance_(instance), island_config_(std::move(island_config)), num_islands_(num_islands),
      migration_interval_(migration_interval), migrants_per_island_(migrants_per_island), base_seed_(base_seed),
      threads_per_island_(threads_per_island) {}

// ---------------------------------------------------------------------------
// Bucle principal del modelo de islas
// ---------------------------------------------------------------------------

GARunResult IslandModel::run() {
	// Cada isla recibe una semilla derivada de base_seed_ con separación prima
	// para garantizar secuencias Mersenne Twister ortogonales entre islas.
	vector<ParallelGeneticAlgorithm> islands;
	islands.reserve(static_cast<size_t>(num_islands_));
	for (int k = 0; k < num_islands_; ++k) {
		islands.emplace_back(instance_, island_config_, base_seed_ + k * 1000003, threads_per_island_);
	}

	// Nivel 1 (externo): un hilo OpenMP por isla.
	// Nivel 2 (interno): ParallelGeneticAlgorithm paraleliza individuos dentro de cada isla.
	omp_set_max_active_levels(2);

	const int total_generations = island_config_.generations;
	const int batches = total_generations / migration_interval_;
	const int remainder = total_generations % migration_interval_;

	for (int batch = 0; batch < batches; ++batch) {
		// Datos compartidos (solo lectura): instance_, island_config_
		// Datos exclusivos por hilo: islands[k] — índice único por iteración → sin carrera.
		// Paralelismo anidado: run_n_generations lanza su propio parallel for interno.
#pragma omp parallel for schedule(static) num_threads(num_islands_)
		for (int k = 0; k < num_islands_; ++k) {
			islands[static_cast<size_t>(k)].run_n_generations(migration_interval_);
		}
		// Barrera implícita al salir del parallel for: todas las islas terminaron
		// su batch antes de ejecutar la migración (sin condición de carrera).
		perform_migration(islands);
	}

	// Generaciones restantes cuando total_generations no es múltiplo de migration_interval_.
	if (remainder > 0) {
#pragma omp parallel for schedule(static) num_threads(num_islands_)
		for (int k = 0; k < num_islands_; ++k) {
			islands[static_cast<size_t>(k)].run_n_generations(remainder);
		}
	}

	// Agregar la mejor solución global entre todas las islas.
	GARunResult result{};
	result.has_feasible = false;
	result.generations_executed = total_generations;

	for (int k = 0; k < num_islands_; ++k) {
		const EvaluatedIndividual best = islands[static_cast<size_t>(k)].get_best();

		// Mejor por fitness global (puede no ser factible).
		if (result.best_by_fitness.chromosome.genes.empty() ||
		    best.evaluation.fitness > result.best_by_fitness.evaluation.fitness) {
			result.best_by_fitness = best;
		}

		// Mejor solución factible: primero verificar el mejor_por_fitness de esta isla.
		if (best.evaluation.feasible_hard) {
			if (!result.has_feasible || best.evaluation.fitness > result.best_feasible.evaluation.fitness) {
				result.best_feasible = best;
				result.has_feasible = true;
			}
		}

		// También consultar la mejor solución factible rastreada a lo largo de toda la
		// evolución de esta isla (puede diferir del individuo final best_by_fitness_).
		if (islands[static_cast<size_t>(k)].has_feasible_solution()) {
			const EvaluatedIndividual island_best_feas = islands[static_cast<size_t>(k)].get_best_feasible();
			if (!result.has_feasible || island_best_feas.evaluation.fitness > result.best_feasible.evaluation.fitness) {
				result.best_feasible = island_best_feas;
				result.has_feasible = true;
			}
		}
	}

	return result;
}

// ---------------------------------------------------------------------------
// Migración con topología de anillo
// ---------------------------------------------------------------------------

void IslandModel::perform_migration(vector<ParallelGeneticAlgorithm> &islands) const {
	const auto n = static_cast<size_t>(num_islands_);

	// Paso 1: recolectar los mejores migrantes de cada isla en el hilo principal.
	// get_top_n re-evalúa population_ (paralelo interno) y devuelve los n mejores cromosomas.
	vector<vector<Chromosome>> outbox(n);
	for (size_t k = 0; k < n; ++k) {
		outbox[k] = islands[k].get_top_n(migrants_per_island_);
	}

	// Paso 2: topología de anillo — isla k envía a isla (k+1) % n.
	// inject_individuals reemplaza los peores individuos con los migrantes recibidos.
	for (size_t k = 0; k < n; ++k) {
		const size_t dest = (k + 1) % n;
		islands[dest].inject_individuals(outbox[k]);
	}
}
