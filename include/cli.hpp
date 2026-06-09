#pragma once
#include <string>

class CliArguments {
public:
	int num_threads;
	int seed;
	int block_size;
	bool verbose;
	std::string input_file;
	std::string variant;

	CliArguments(int argc, char *argv[]);
	void display();
};
