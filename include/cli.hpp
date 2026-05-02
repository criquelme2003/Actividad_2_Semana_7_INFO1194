#pragma once
#include <string>

class CliArguments {
public:
	int num_threads;
	int seed;
	bool verbose;
	std::string input_file;
	std::string variant;
	std::string output_csv;

	CliArguments(int argc, char *argv[]);
	void display();
};
