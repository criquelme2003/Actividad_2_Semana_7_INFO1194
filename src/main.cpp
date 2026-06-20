#include "cli.hpp"
#include "constants.hpp"
#include "genetic_algorithm.hpp"
#include "genetic_algorithm_cuda_interface.hpp"
#include "instance_loader.hpp"
#include <chrono>
#include <filesystem>
#include <fmt/core.h>
#include <fstream>
#include <stdexcept>
#include <utility>

using std::runtime_error;
using namespace constants;

namespace {
GAConfig build_config(const CliArguments &args) {
	return GAConfig{.population_size            = args.population_size,
	                .generations                = args.generations,
	                .elitism_count              = DEFAULT_ELITISM_COUNT,
	                .crossover_rate             = DEFAULT_CROSSOVER_RATE,
	                .mutation_rate              = DEFAULT_MUTATION_RATE,
	                .max_stagnation_generations = MAX_STAGNATION_GENERATIONS,
	                .penalties                  = PenaltyConfig{.alpha      = PENALTY_ALPHA,
	                                                            .beta       = PENALTY_BETA,
	                                                            .w_weight   = PENALTY_W_WEIGHT,
	                                                            .w_volume   = PENALTY_W_VOLUME,
	                                                            .w_category = PENALTY_W_CATEGORY,
	                                                            .w_incompat = PENALTY_W_INCOMPAT,
	                                                            .w_dep      = PENALTY_W_DEP}};
}

// Extrae el nombre base de la ruta de la instancia (último componente del path)
std::string instance_name(const std::string &path) {
	return std::filesystem::path(path).filename().string();
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
		const GAConfig config          = build_config(args);

		GARunResult run_result{};

		// ── Medición de tiempo total (wall-clock) ───────────────────────────
		auto t_start = std::chrono::high_resolution_clock::now();

		if (args.variant == "sequential") {
			GeneticAlgorithm ga(instance, config, args.seed);
			run_result = ga.run();
		} else if (args.variant == "cuda_basic") {
			run_result = run_genetic_algorithm_cuda_basic(instance, config, args.seed, args.block_size);
		} else if (args.variant == "cuda_optimized") {
			run_result = run_genetic_algorithm_cuda_optimized(instance, config, args.seed, args.block_size);
		} else {
			throw runtime_error("Variante desconocida: '" + args.variant +
			                    "'. Variantes disponibles: sequential, cuda_basic, cuda_optimized");
		}

		auto t_end = std::chrono::high_resolution_clock::now();
		double elapsed_ms =
		    std::chrono::duration<double, std::milli>(t_end - t_start).count();
		// ────────────────────────────────────────────────────────────────────

		// ── Salida estándar ─────────────────────────────────────────────────
		fmt::print("Generaciones ejecutadas: {}\n", run_result.generations_executed);
		fmt::print("Tiempo total: {:.2f} ms\n", elapsed_ms);
		fmt::print("Tiempo kernels GPU: {:.2f} ms\n", run_result.kernel_time_ms);
		fmt::print("Tiempo transferencias GPU: {:.2f} ms (H2D: {:.2f}, D2H: {:.2f})\n",
		           run_result.transfer_h2d_ms + run_result.transfer_d2h_ms,
		           run_result.transfer_h2d_ms, run_result.transfer_d2h_ms);
		fmt::print("Mejor fitness: {:.6f}\n", run_result.best_by_fitness.evaluation.fitness);
		fmt::print("Mejor valor total: {:.3f}\n", run_result.best_by_fitness.evaluation.total_value);
		fmt::print("Factible (restricciones duras): {}\n",
		           run_result.best_by_fitness.evaluation.feasible_hard ? "si" : "no");
		fmt::print("Penalizacion total: {:.6f}\n", run_result.best_by_fitness.evaluation.penalty);

		if (run_result.has_feasible) {
			fmt::print("Mejor fitness factible: {:.6f}\n", run_result.best_feasible.evaluation.fitness);
			fmt::print("Mejor valor factible: {:.3f}\n",   run_result.best_feasible.evaluation.total_value);
		} else {
			fmt::print("No se encontro solucion factible en restricciones duras\n");
		}
		// ────────────────────────────────────────────────────────────────────

		// ── Salida CSV experimental (si se especificó --bench-out) ──────────
		// Formato: variant,instance,pop_size,n_items,seed,block_size,
		//          gen_exec,time_ms,kernel_time_ms,transfer_d2h_ms,transfer_h2d_ms,
		//          best_fitness,best_value,has_feasible,
		//          best_feasible_fitness,best_feasible_value
		if (!args.bench_out.empty()) {
			std::ofstream csv(args.bench_out, std::ios::app);
			const auto &bf = run_result.best_by_fitness.evaluation;
			double feas_fit = run_result.has_feasible ? run_result.best_feasible.evaluation.fitness   : -1.0;
			double feas_val = run_result.has_feasible ? run_result.best_feasible.evaluation.total_value : -1.0;
			csv << args.variant                        << ","
			    << instance_name(args.input_file)      << ","
			    << args.population_size                << ","
			    << instance.items.size()               << ","
			    << args.seed                           << ","
			    << args.block_size                     << ","
			    << run_result.generations_executed     << ","
			    << elapsed_ms                          << ","
			    << run_result.kernel_time_ms           << ","
			    << run_result.transfer_d2h_ms          << ","
			    << run_result.transfer_h2d_ms          << ","
			    << bf.fitness                          << ","
			    << bf.total_value                      << ","
			    << (run_result.has_feasible ? 1 : 0)  << ","
			    << feas_fit                            << ","
			    << feas_val                            << "\n";
		}
		// ────────────────────────────────────────────────────────────────────

		return 0;
	} catch (const std::exception &err) {
		fmt::print(stderr, "Error: {}\n", err.what());
		return 1;
	}
}
