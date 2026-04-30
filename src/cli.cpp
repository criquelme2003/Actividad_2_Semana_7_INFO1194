#include "cli.hpp"
#include "argparse/argparse.hpp"
#include <cstdlib>
#include <iostream>
#include <stdexcept>

CliArguments::CliArguments(int argc, char *argv[]) {
	argparse::ArgumentParser program("mochila_ga");

	program.add_argument("--instance")
	    .help("Ruta de la instancia/archivo de entrada")
	    .required();

	program.add_argument("--variant")
	    .help("Variante del algoritmo")
	    .default_value(std::string("islands"));

	program.add_argument("-t", "--threads")
	    .help("Número de hilos para OpenMP")
	    .default_value(1)
	    .scan<'i', int>();

	program.add_argument("--seed")
	    .help("Semilla para reproducibilidad")
	    .default_value(123)
	    .scan<'i', int>();

	program.add_argument("-v", "--verbose")
	    .help("Modo verbose")
	    .default_value(false)
	    .implicit_value(true);

	try {
		program.parse_args(argc, argv);
	} catch (const std::runtime_error &err) {
		std::cerr << err.what() << std::endl;
		std::cerr << program;
		std::exit(1);
	}

	input_file = program.get<std::string>("--instance");
	variant = program.get<std::string>("--variant");
	seed = program.get<int>("--seed");
	num_threads = program.get<int>("--threads");
	verbose = program.get<bool>("--verbose");
}

void CliArguments::display() {
	std::cout << "Input file: " << input_file << std::endl;
	std::cout << "Variant: " << variant << std::endl;
	std::cout << "Seed: " << seed << std::endl;
	std::cout << "Threads: " << num_threads << std::endl;
	std::cout << "Verbose: " << (verbose ? "Yes" : "No") << std::endl;
}
