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

	program.add_argument("--block-size")
	    .help("Tamaño de bloque CUDA (hilos por bloque)")
	    .default_value(constants::DEFAULT_BLOCK_SIZE)
	    .scan<'i', int>();

	program.add_argument("--seed")
	    .help("Semilla para reproducibilidad")
	    .default_value(constants::DEFAULT_SEED)
	    .scan<'i', int>();

	program.add_argument("--pop-size")
	    .help("Tamaño de la población")
	    .default_value(constants::DEFAULT_POPULATION_SIZE)
	    .scan<'i', int>();

	program.add_argument("--generations")
	    .help("Número máximo de generaciones")
	    .default_value(constants::DEFAULT_GENERATIONS)
	    .scan<'i', int>();

	program.add_argument("--bench-out")
	    .help("Archivo CSV donde se añade una fila con los resultados experimentales")
	    .default_value(string(""));

	program.add_argument("-v", "--verbose").help("Modo verbose").default_value(false).implicit_value(true);

	try {
		program.parse_args(argc, argv);
	} catch (const runtime_error &err) {
		fmt::print(stderr, "{}\n{}", err.what(), fmt::streamed(program));
		std::exit(1);
	}

	input_file      = program.get<string>("--instance");
	variant         = program.get<string>("--variant");
	seed            = program.get<int>("--seed");
	num_threads     = program.get<int>("--threads");
	block_size      = program.get<int>("--block-size");
	population_size = program.get<int>("--pop-size");
	generations     = program.get<int>("--generations");
	bench_out       = program.get<string>("--bench-out");
	verbose         = program.get<bool>("--verbose");
}

void CliArguments::display() {
	fmt::print("Input file: {}\n", input_file);
	fmt::print("Variant: {}\n", variant);
	fmt::print("Seed: {}\n", seed);
	fmt::print("Threads: {}\n", num_threads);
	fmt::print("Verbose: {}\n", verbose ? "Yes" : "No");
}
