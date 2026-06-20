#pragma once
#include <string>

class CliArguments {
public:
	int num_threads;
	int seed;
	int block_size;
	int population_size;
	int generations;
	bool verbose;
	std::string input_file;
	std::string variant;
	std::string bench_out; // archivo CSV para resultados experimentales (vacío = no guardar)

	CliArguments(int argc, char *argv[]);
	void display();
};
