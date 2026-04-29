#include "argparse/argparse.hpp" // argparse
#include "csv.h"                 // fast-cpp-csv-parser
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

// Estructura para almacenar datos CSV (ejemplo)
struct Record {
    std::string campo1;
    double campo2;
    int campo3;
};

int main(int argc, char *argv[]) {
    // Configurar parser de argumentos
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
        return 1;
    }

    const std::string input_file = program.get<std::string>("--instance");
    const std::string variant = program.get<std::string>("--variant");
    const int num_threads = program.get<int>("--threads");
    const int seed = program.get<int>("--seed");
    const bool verbose = program.get<bool>("--verbose");

    // Configurar OpenMP
    omp_set_num_threads(num_threads);
    if (verbose) {
        std::cout << "Usando " << num_threads << " hilos" << std::endl;
        std::cout << "Variant: " << variant << std::endl;
        std::cout << "Seed: " << seed << std::endl;
    }

    // Leer CSV
    std::vector<Record> records;
    try {
        io::CSVReader<3> in(input_file);
        in.read_header(io::ignore_extra_column, "col1", "col2", "col3");
        std::string col1;
        double col2;
        int col3;
        while (in.read_row(col1, col2, col3)) {
            records.push_back({col1, col2, col3});
        }
    } catch (const std::exception &e) {
        std::cerr << "Error al leer CSV: " << e.what() << std::endl;
        return 1;
    }

    if (verbose) {
        std::cout << "Leídos " << records.size() << " registros" << std::endl;
    }

    // Procesamiento paralelo con OpenMP (ejemplo: cálculo simple)
    double suma = 0.0;
#pragma omp parallel for reduction(+ : suma)
    for (size_t i = 0; i < records.size(); ++i) {
        suma += records[i].campo2;
        if (verbose && omp_get_thread_num() == 0 && i % 1000 == 0) {
            std::cout << "Progreso: " << i << "/" << records.size() << std::endl;
        }
    }

    if (verbose) {
        std::cout << "Suma de campo2: " << suma << std::endl;
    }

    return 0;
}
