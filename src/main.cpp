#include "cli.hpp"
#include "constants.hpp"
#include "genetic_algorithm.hpp"
#include "instance_loader.hpp"
#include <fmt/core.h>
#include <stdexcept>
#include <utility>

using std::runtime_error;

using namespace constants;

namespace {
GAConfig build_config() {
	return GAConfig{.population_size = DEFAULT_POPULATION_SIZE,
	                .generations = DEFAULT_GENERATIONS,
	                .elitism_count = DEFAULT_ELITISM_COUNT,
	                .crossover_rate = DEFAULT_CROSSOVER_RATE,
	                .mutation_rate = DEFAULT_MUTATION_RATE,
	                .max_stagnation_generations = MAX_STAGNATION_GENERATIONS,
	                .penalties = PenaltyConfig{.alpha = PENALTY_ALPHA,
	                                           .beta = PENALTY_BETA,
	                                           .gamma = PENALTY_GAMMA,
	                                           .delta = PENALTY_DELTA,
	                                           .epsilon = PENALTY_EPSILON}};
}
} // namespace

int main(int argc, char *argv[]) {
	try {
		CliArguments args(argc, argv);

		if (args.verbose) {
			args.display();
		}

		auto instance_result = load_problem_instance(args.input_file, DEFAULT_CAPACITY_RATIO);

		if (!instance_result) {
			throw runtime_error(instance_result.error());
		}

		const ProblemInstance instance = std::move(instance_result.value());

		if (args.variant != "sequential") {
			throw runtime_error("En esta etapa solo esta implementada la variante sequential");
		}

		GeneticAlgorithm ga(instance, build_config(), args.seed);
		const GARunResult run_result = ga.run();

		fmt::print("Generaciones ejecutadas: {}\n", run_result.generations_executed);
		fmt::print("Mejor fitness: {:.3f}\n", run_result.best_by_fitness.evaluation.fitness);
		fmt::print("Mejor valor total: {:.3f}\n", run_result.best_by_fitness.evaluation.total_value);
		fmt::print("Factible (restricciones duras): {}\n",
		           run_result.best_by_fitness.evaluation.feasible_hard ? "si" : "no");
		fmt::print("Penalizacion total: {:.3f}\n", run_result.best_by_fitness.evaluation.penalty);

		if (run_result.has_feasible) {
			fmt::print("Mejor fitness factible: {:.3f}\n", run_result.best_feasible.evaluation.fitness);
		} else {
			fmt::print("No se encontro solucion factible en restricciones duras\n");
		}

		return 0;
	} catch (const std::exception &err) {
		fmt::print(stderr, "Error: {}\n", err.what());
		return 1;
	}
}
