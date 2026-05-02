# Guia de implementacion: Variante 3 — Modelo de islas

Esta guia asume que Variante 1 (secuencial) y Variante 2 (paralela) estan completas.
El codigo base de Variante 2 ya expone la interfaz necesaria para construir el modelo de islas.

---

## Concepto del modelo de islas

En lugar de evolucionar una sola poblacion grande, se crean `K` subpoblaciones independientes
(islas). Cada isla evoluciona por su cuenta durante `migration_interval` generaciones. Al
terminar ese bloque, las islas intercambian sus mejores individuos (migracion). El ciclo se
repite hasta alcanzar el total de generaciones.

```
[Isla 0] → evoluciona 25 gen → envia 2 mejores → recibe 2 de Isla K-1
[Isla 1] → evoluciona 25 gen → envia 2 mejores → recibe 2 de Isla 0
...
[Isla K-1] → evoluciona 25 gen → envia 2 mejores → recibe 2 de Isla K-2
```

Las islas evolucionan en paralelo (un hilo por isla). La migracion ocurre de forma secuencial
tras una barrera de sincronizacion.

---

## Interfaz ya disponible en `ParallelGeneticAlgorithm`

`include/parallel_genetic_algorithm.hpp` ya expone los tres metodos que necesita `IslandModel`:

```cpp
// Evoluciona exactamente n generaciones a partir del estado interno actual.
// Si la poblacion esta vacia, la inicializa primero.
void run_n_generations(int n);

// Reemplaza los peores individuos de la poblacion con los cromosomas recibidos.
void inject_individuals(const std::vector<Chromosome> &migrants);

// Devuelve el mejor individuo evaluado hasta el momento (sin evolucionar).
EvaluatedIndividual get_best() const;
```

No es necesario modificar `ParallelGeneticAlgorithm` ni ningun archivo de Variante 2.

---

## Archivos a crear

| Archivo | Contenido |
|---|---|
| `include/island_model.hpp` | Declaracion de `IslandModel` |
| `src/island_model.cpp` | Implementacion: bucle de migracion + paralelismo |

## Archivos a modificar

| Archivo | Cambio |
|---|---|
| `src/main.cpp` | Agregar rama `else if (args.variant == "islands")` |
| `CMakeLists.txt` | Agregar `src/island_model.cpp` al `add_executable` |
| `include/constants.hpp` | Agregar constantes del modelo de islas |

---

## Estructura de `IslandModel`

```cpp
// include/island_model.hpp
#pragma once
#include "models.hpp"
#include <vector>

class IslandModel {
public:
    IslandModel(const ProblemInstance &instance,
                GAConfig island_config,      // config compartida por todas las islas
                int num_islands,
                int migration_interval,      // cada cuantas generaciones migrar
                int migrants_per_island,     // cuantos individuos enviar/recibir
                int base_seed,               // isla i se siembra con base_seed + i * PRIME
                int threads_per_island);     // hilos que usa cada isla internamente

    GARunResult run();

private:
    void perform_migration(std::vector<ParallelGeneticAlgorithm> &islands) const;

    const ProblemInstance &instance_;
    GAConfig               island_config_;
    int                    num_islands_;
    int                    migration_interval_;
    int                    migrants_per_island_;
    int                    base_seed_;
    int                    threads_per_island_;
};
```

---

## Implementacion del bucle principal

```cpp
// src/island_model.cpp
#include "island_model.hpp"
#include "parallel_genetic_algorithm.hpp"
#include <omp.h>
#include <algorithm>

GARunResult IslandModel::run() {
    // Crear una isla por indice, cada una con semilla distinta
    std::vector<ParallelGeneticAlgorithm> islands;
    islands.reserve(static_cast<std::size_t>(num_islands_));
    for (int i = 0; i < num_islands_; ++i) {
        islands.emplace_back(instance_, island_config_,
                             base_seed_ + i * 999983,   // primo para separar streams RNG
                             threads_per_island_);
    }

    const int total_gen = island_config_.generations;
    int done = 0;

    while (done < total_gen) {
        const int batch = std::min(migration_interval_, total_gen - done);

        // Las islas evolucionan en paralelo — una isla por hilo
        #pragma omp parallel for schedule(static) num_threads(num_islands_)
        for (int i = 0; i < num_islands_; ++i) {
            islands[static_cast<std::size_t>(i)].run_n_generations(batch);
        }
        // Barrera implicita aqui: todas las islas terminaron el batch

        done += batch;

        if (done < total_gen) {
            perform_migration(islands);  // secuencial, despues de la barrera
        }
    }

    // Recolectar el mejor global entre todas las islas
    GARunResult result{};
    for (auto &island : islands) {
        const auto best = island.get_best();
        if (!result.has_feasible ||
            best.evaluation.fitness > result.best_by_fitness.evaluation.fitness) {
            result.best_by_fitness = best;
        }
        // TODO: tambien recolectar best_feasible si se expone en get_best_feasible()
    }
    result.generations_executed = total_gen;
    return result;
}
```

---

## Implementacion de la migracion (topologia anillo)

```cpp
void IslandModel::perform_migration(
    std::vector<ParallelGeneticAlgorithm> &islands) const {

    const auto k = static_cast<std::size_t>(num_islands_);

    // Paso 1: cada isla extrae sus mejores individuos
    // get_best() devuelve solo 1 individuo; para migrants_per_island_ > 1
    // necesitas exponer get_top_n() o evaluar + ordenar la poblacion aqui.
    // Por ahora se muestra el caso migrants_per_island_ = 1.
    std::vector<std::vector<Chromosome>> outbox(k);
    for (std::size_t i = 0; i < k; ++i) {
        outbox[i].push_back(islands[i].get_best().chromosome);
    }

    // Paso 2: anillo — isla i recibe de isla (i-1+k)%k
    for (std::size_t i = 0; i < k; ++i) {
        const std::size_t sender = (i + k - 1) % k;
        islands[i].inject_individuals(outbox[sender]);
    }
}
```

> **Nota sobre migrants_per_island_ > 1:** `get_best()` retorna solo el mejor.
> Para enviar varios migrantes hay dos opciones:
> - Agregar `get_top_n(int n)` a `ParallelGeneticAlgorithm` que retorne los `n` mejores
>   del vector `population_` tras evaluarlo.
> - Hacer `outbox` de tipo `std::vector<EvaluatedIndividual>` y que la isla exponga su
>   poblacion evaluada ordenada (requiere un getter adicional).
> La opcion recomendada es `get_top_n(int n)` — minima superficie de API.

---

## Constantes sugeridas en `constants.hpp`

```cpp
// Modelo de islas
inline constexpr int DEFAULT_NUM_ISLANDS        = 4;
inline constexpr int DEFAULT_MIGRATION_INTERVAL = 25;   // generaciones entre migraciones
inline constexpr int DEFAULT_MIGRANTS_PER_ISLAND = 2;   // individuos que migran por isla
```

Con `DEFAULT_GENERATIONS = 200` y `migration_interval = 25`, se realizan 8 rondas de
migracion por ejecucion.

---

## Rama en `main.cpp`

```cpp
} else if (args.variant == "islands") {
    IslandModel islands(instance, build_config(),
                        DEFAULT_NUM_ISLANDS,
                        DEFAULT_MIGRATION_INTERVAL,
                        DEFAULT_MIGRANTS_PER_ISLAND,
                        args.seed,
                        args.num_threads / DEFAULT_NUM_ISLANDS);
    run_result = islands.run();
```

El calculo `args.num_threads / DEFAULT_NUM_ISLANDS` distribuye los hilos disponibles entre
islas. Si el usuario pasa `--threads 8` con 4 islas, cada isla usa 2 hilos internamente.

---

## Consideraciones de concurrencia

| Zona | Tipo | Mecanismo |
|---|---|---|
| `run_n_generations` por isla | Paralela (entre islas) | `#pragma omp parallel for` sobre islas |
| Internos de cada isla | Paralelos (entre individuos) | `#pragma omp parallel for` de Variante 2 |
| `perform_migration` | Secuencial | Barrera implicita del `parallel for` |
| `outbox` | Local en hilo principal | Sin concurrencia |

No hay condiciones de carrera: la migracion ocurre siempre despues de que todas las islas
terminaron su batch (barrera implicita de OpenMP al salir del `parallel for`).

---

## Checklist de implementacion

- [ ] Agregar constantes de islas en `include/constants.hpp`
- [ ] Crear `include/island_model.hpp`
- [ ] Crear `src/island_model.cpp` con `run()` y `perform_migration()`
- [ ] Si `migrants_per_island_ > 1`, agregar `get_top_n(int n)` en `ParallelGeneticAlgorithm`
- [ ] Agregar rama `"islands"` en `src/main.cpp`
- [ ] Agregar `src/island_model.cpp` en `CMakeLists.txt`
- [ ] Compilar y ejecutar: `./mochila_ga --instance data/instance_small --variant islands --threads 4 --seed 42`
- [ ] Verificar que la variante `sequential` y `parallel` siguen dando los mismos resultados

---

Anterior: [08-guia-de-ejecucion-y-debug.md](./08-guia-de-ejecucion-y-debug.md)
