# Diseno del GA secuencial

El motor principal esta en `src/genetic_algorithm.cpp`.

Flujo por generacion:

1. Evaluar poblacion completa.
2. Ordenar por fitness descendente.
3. Actualizar mejor global por fitness.
4. Actualizar mejor factible (restricciones duras).
5. Aplicar elitismo.
6. Completar nueva poblacion con torneo + crossover + mutacion.
7. Terminar por tope de generaciones o estancamiento.

Parametros relevantes (`GAConfig`):

- `population_size`
- `generations`
- `tournament_size`
- `elitism_count`
- `crossover_rate`
- `mutation_rate`
- `max_stagnation_generations`

La clase `GeneticAlgorithm` conserva `rng` interno inicializado por semilla para reproducibilidad.

Valores actuales usados en esta etapa (definidos en `include/constants.hpp`):

- `DEFAULT_POPULATION_SIZE = 120`
- `DEFAULT_GENERATIONS = 200`
- `DEFAULT_TOURNAMENT_SIZE = 3`
- `DEFAULT_ELITISM_COUNT = 2`
- `DEFAULT_CROSSOVER_RATE = 0.85`
- `DEFAULT_MUTATION_RATE = 0.02`
- `DEFAULT_SEED = 123`
- `MAX_STAGNATION_GENERATIONS = 50`

Fragmento del ciclo principal:

```cpp
for (int generation = 0; generation < config_.generations; ++generation) {
    auto evaluated = evaluate_population(population);
    std::sort(evaluated.begin(), evaluated.end(),
              [](const EvaluatedIndividual &a, const EvaluatedIndividual &b) {
                  return a.evaluation.fitness > b.evaluation.fitness;
              });
    // elitismo + seleccion + operadores
}
```

Limitaciones actuales de esta fase:

- no se ejecuta paralelizacion OpenMP del GA;
- no se implementa modelo de islas;
- las penalizaciones usan valores por defecto configurables.

---

Siguiente: [04-fitness-penalizaciones-y-factibilidad.md](./04-fitness-penalizaciones-y-factibilidad.md)
