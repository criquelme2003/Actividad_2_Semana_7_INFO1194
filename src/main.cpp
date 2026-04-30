#include "cli.hpp"
#include "record_csv.hpp"
#include <expected>
#include <iostream>
#include <omp.h>
#include <stdexcept>
#include <vector>

int main(int argc, char *argv[]) {
	CliArguments args(argc, argv);
	auto [num_threads, seed, verbose, input_file, variant] = args;

	omp_set_num_threads(num_threads);

	std::vector<Record> records;
	auto read_result = read_csv<Record>(input_file);

	if (!read_result) {
		throw std::runtime_error(read_result.error());
	}

	records = std::move(read_result.value());

	if (verbose) {
		std::cout << "Leidos " << records.size() << " registros" << std::endl;
	}

	double suma = 0.0;

#pragma omp parallel for reduction(+ : suma)
	for (size_t i = 0; i < records.size(); ++i) {
		suma += records[i].campo2;
	}

	if (verbose) {
		std::cout << "Suma de campo2: " << suma << std::endl;
	}

	return 0;
}
