#pragma once

#include "fitness.hpp"
#include "models.hpp"
#include <random>
#include <vector>

// GA paralelo con OpenMP. Paraleliza evaluación de aptitud y la cadena
// selección-cruce-mutación usando un RNG independiente por hilo.
//
// También expone run_n_generations / inject_individuals / get_best
// para ser usado como bloque base del modelo de islas (Variante 3).
class ParallelGeneticAlgorithm {
public:
    ParallelGeneticAlgorithm(const ProblemInstance &instance, GAConfig config, int seed,
                             int num_threads);

    // Ejecuta el bucle completo hasta convergencia o max generaciones.
    GARunResult run();

    // --- Interfaz para Variante 3 (IslandModel) ---

    // Inicializa la población si aún no existe, luego evoluciona exactamente n generaciones.
    // Permite que IslandModel controle el ciclo de migración externamente.
    void run_n_generations(int n);

    // Reemplaza los peores individuos de la población actual con los migrantes recibidos.
    void inject_individuals(const std::vector<Chromosome> &migrants);

    // Devuelve el mejor individuo evaluado de la generación actual (sin evolucionar).
    EvaluatedIndividual get_best() const;

private:
    Chromosome              make_random_chromosome();
    std::vector<Chromosome> make_initial_population();

    // Evalúa todos los individuos de population_ en paralelo.
    std::vector<EvaluatedIndividual> evaluate_population_parallel() const;

    // Selección por torneo + cruce + mutación en paralelo para generar offspring_needed hijos.
    std::vector<Chromosome> generate_offspring_parallel(
        const std::vector<EvaluatedIndividual> &evaluated, int offspring_needed);

    const ProblemInstance    &instance_;
    GAConfig                  config_;
    FitnessEvaluator          evaluator_;    // const evaluate() → thread-safe
    int                       num_threads_;
    std::mt19937              master_rng_;   // solo para make_initial_population
    std::vector<std::mt19937> thread_rngs_;  // thread_rngs_[t] es privado al hilo t

    // Población como campo para que run_n_generations retome el estado entre llamadas.
    std::vector<Chromosome> population_;

    // Mejor individuo cacheado (actualizado cada generación).
    EvaluatedIndividual best_by_fitness_;
    EvaluatedIndividual best_feasible_;
    bool                has_feasible_{false};
};
