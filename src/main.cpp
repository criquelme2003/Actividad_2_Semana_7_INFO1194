#include "cli.hpp"
#include "constants.hpp"
#include "genetic_algorithm.hpp"
#include "instance_loader.hpp"
#include "island_model.hpp"
#include "parallel_genetic_algorithm.hpp"
#include <chrono>
#include <fmt/core.h>
#include <fstream>
#include <stdexcept>
#include <utility>

using std::runtime_error;

using namespace constants;

namespace {
GAConfig build_config() {
	return GAConfig{.population_size = DEFAULT_POPULATION_SIZE,
	                .generations = DEFAULT_GENERATIONS,
	                .tournament_size = DEFAULT_TOURNAMENT_SIZE,
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

static int count_selected(const Chromosome &c) {
	int n = 0;
	for (auto g : c.genes) {
		n += g;
	}
	return n;
}

// Escribe la cabecera CSV solo si el archivo no existe o está vacío.
void ensure_csv_header(const std::string &path) {
	std::ifstream check(path);
	const bool needs_header = !check.is_open() || check.peek() == std::ifstream::traits_type::eof();
	check.close();
	if (!needs_header) {
		return;
	}
	std::ofstream f(path);
	if (f.is_open()) {
		f << "instance,variant,threads,seed,generations,"
		     "best_fitness,best_value,"
		     "best_feasible_fitness,best_feasible_value,"
		     "feasible,time_ms,"
		     "best_feasible_items,best_feasible_weight,best_feasible_volume\n";
	}
}

void write_csv_row(const std::string &path, const std::string &instance, const std::string &variant, int threads,
                   int seed, const GARunResult &result, double elapsed_ms) {
	ensure_csv_header(path);
	std::ofstream f(path, std::ios::app);
	if (!f.is_open()) {
		return;
	}
	const double feasible_fitness = result.has_feasible ? result.best_feasible.evaluation.fitness : 0.0;
	const double feasible_value   = result.has_feasible ? result.best_feasible.evaluation.total_value : 0.0;
	const int    feasible_items   = result.has_feasible ? count_selected(result.best_feasible.chromosome) : 0;
	const double feasible_weight  = result.has_feasible ? result.best_feasible.evaluation.total_weight : 0.0;
	const double feasible_volume  = result.has_feasible ? result.best_feasible.evaluation.total_volume : 0.0;

	f << instance << ',' << variant << ',' << threads << ',' << seed << ',' << result.generations_executed << ','
	  << result.best_by_fitness.evaluation.fitness << ',' << result.best_by_fitness.evaluation.total_value << ','
	  << feasible_fitness << ',' << feasible_value << ',' << (result.has_feasible ? 1 : 0) << ',' << elapsed_ms << ','
	  << feasible_items << ',' << feasible_weight << ',' << feasible_volume << '\n';
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

		const auto t_start = std::chrono::steady_clock::now();
		GARunResult run_result;

		if (args.variant == "sequential") {
			GeneticAlgorithm ga(instance, build_config(), args.seed);
			run_result = ga.run();
		} else if (args.variant == "parallel") {
			ParallelGeneticAlgorithm ga(instance, build_config(), args.seed, args.num_threads);
			run_result = ga.run();
		} else if (args.variant == "islands") {
			// Cada hilo OpenMP corre una isla secuencial independiente.
			// --threads N = N islas corriendo en paralelo (paralelismo grueso entre islas).
			IslandModel islands_model(instance, build_config(), args.num_threads, DEFAULT_MIGRATION_INTERVAL,
			                          DEFAULT_MIGRANTS_PER_ISLAND, args.seed);
			run_result = islands_model.run();
		} else {
			throw runtime_error("Variante no reconocida: '" + args.variant +
			                    "'. Opciones disponibles: sequential, parallel, islands");
		}

		const double elapsed_ms =
		    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();

		fmt::print("Variante: {}\n", args.variant);
		fmt::print("Hilos: {}\n", args.num_threads);
		fmt::print("Semilla: {}\n", args.seed);
		fmt::print("Generaciones ejecutadas: {}\n", run_result.generations_executed);
		fmt::print("Mejor fitness: {:.3f}\n", run_result.best_by_fitness.evaluation.fitness);
		fmt::print("Mejor valor total: {:.3f}\n", run_result.best_by_fitness.evaluation.total_value);
		fmt::print("Factible (restricciones duras): {}\n",
		           run_result.best_by_fitness.evaluation.feasible_hard ? "si" : "no");
		fmt::print("Penalizacion total: {:.3f}\n", run_result.best_by_fitness.evaluation.penalty);
		fmt::print("Tiempo de ejecucion: {:.3f} ms\n", elapsed_ms);

		if (run_result.has_feasible) {
			fmt::print("Mejor fitness factible: {:.6f}\n", run_result.best_feasible.evaluation.fitness);
			fmt::print("Mejor valor factible:   {:.6f}\n", run_result.best_feasible.evaluation.total_value);
			if (args.verbose) {
				fmt::print("  Items seleccionados:   {}\n", count_selected(run_result.best_feasible.chromosome));
				fmt::print("  Peso total:            {:.6f}\n", run_result.best_feasible.evaluation.total_weight);
				fmt::print("  Volumen total:         {:.6f}\n", run_result.best_feasible.evaluation.total_volume);
				fmt::print("  Penalizacion:          {:.6f}\n", run_result.best_feasible.evaluation.penalty);
				fmt::print("  Violaciones categoria: {}\n", run_result.best_feasible.evaluation.category_violations);
				fmt::print("  Violaciones incompat:  {}\n", run_result.best_feasible.evaluation.incompatibility_violations);
				fmt::print("  Violaciones dep:       {}\n", run_result.best_feasible.evaluation.dependency_violations);
			}
		} else {
			fmt::print("No se encontro solucion factible en restricciones duras\n");
		}

		if (!args.output_csv.empty()) {
			write_csv_row(args.output_csv, args.input_file, args.variant, args.num_threads, args.seed, run_result,
			              elapsed_ms);
		}

		return 0;
	} catch (const std::exception &err) {
		fmt::print(stderr, "Error: {}\n", err.what());
		return 1;
	}
}
