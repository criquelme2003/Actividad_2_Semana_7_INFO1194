# Integracion con el proyecto

Archivos nuevos y su rol:

- `include/models.hpp`: modelos de datos.
- `include/constants.hpp`: constantes globales (`inline constexpr`).
- `include/instance_loader.hpp` + `src/instance_loader.cpp`: carga de instancia extendida.
- `include/fitness.hpp` + `src/fitness.cpp`: fitness y penalizaciones.
- `include/selection.hpp` + `src/selection.cpp`: torneo.
- `include/operators.hpp` + `src/operators.cpp`: crossover/mutacion.
- `include/genetic_algorithm.hpp` + `src/genetic_algorithm.cpp`: motor GA secuencial.

`src/main.cpp` ahora:

1. parsea CLI,
2. carga instancia,
3. ejecuta variante `sequential`,
4. imprime resumen con `fmt`.

La integracion mantiene CMake + OpenMP + fmt + argparse + parser CSV actual.

Estado actual de variantes:

- implementada: `sequential`.
- pendiente para siguientes etapas: `openmp` e `islands`.

Fragmento de integracion en `main`:

```cpp
if (args.variant != "sequential") {
    throw std::runtime_error(
        "En esta etapa solo esta implementada la variante sequential");
}

GeneticAlgorithm ga(instance, build_config(args), args.seed);
const GARunResult run_result = ga.run();
```

---

Siguiente: [08-guia-de-ejecucion-y-debug.md](./08-guia-de-ejecucion-y-debug.md)
