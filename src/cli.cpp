#include "cli.hpp"
#include "argparse/argparse.hpp"
#include "constants.hpp"
#include <cstdlib>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <stdexcept>

using std::runtime_error;
using std::string;

CliArguments::CliArguments(int argc, char *argv[]) {
	argparse::ArgumentParser program("mochila_ga");

	program.add_argument("--instance").help("Ruta de la instancia/archivo de entrada").required();

	program.add_argument("--variant").help("Variante del algoritmo").default_value(string("sequential"));

	program.add_argument("-t", "--threads").help("Número de hilos para OpenMP").default_value(1).scan<'i', int>();

	program.add_argument("--seed")
	    .help("Semilla para reproducibilidad")
	    .default_value(constants::DEFAULT_SEED)
	    .scan<'i', int>();

	program.add_argument("-v", "--verbose").help("Modo verbose").default_value(false).implicit_value(true);

	program.add_argument("--output-csv")
	    .help("Ruta del archivo CSV donde se acumulan los resultados experimentales")
	    .default_value(string(""));

	try {
		program.parse_args(argc, argv);
	} catch (const runtime_error &err) {
		fmt::print(stderr, "{}\n{}", err.what(), fmt::streamed(program));
		std::exit(1);
	}

	input_file = program.get<string>("--instance");
	variant = program.get<string>("--variant");
	seed = program.get<int>("--seed");
	num_threads = program.get<int>("--threads");
	verbose = program.get<bool>("--verbose");
	output_csv = program.get<string>("--output-csv");
}

void CliArguments::display() {
	fmt::print("Input file: {}\n", input_file);
	fmt::print("Variant: {}\n", variant);
	fmt::print("Seed: {}\n", seed);
	fmt::print("Threads: {}\n", num_threads);
	fmt::print("Verbose: {}\n", verbose ? "Yes" : "No");
	if (!output_csv.empty()) {
		fmt::print("Output CSV: {}\n", output_csv);
	}
}
