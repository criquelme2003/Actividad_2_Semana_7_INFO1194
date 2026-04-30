#pragma once
#include <string>

class CliArguments {
public:
	int num_threads;
	int seed;
	bool verbose;
	std::string input_file;
	std::string variant;

	CliArguments(int argc, char *argv[]);
	void display();
};
