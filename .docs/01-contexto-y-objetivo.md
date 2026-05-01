# Contexto y objetivo

La Actividad 2 solicita resolver una version extendida del problema de la mochila con algoritmo genetico y comparaciones de rendimiento entre variantes secuencial, paralela e islas.

Este entregable cubre exclusivamente la parte de Persona 1:

- base secuencial completa;
- representacion y evolucion de poblacion;
- fitness con penalizaciones;
- operadores geneticos;
- estructura mantenible para continuidad del equipo.

La implementacion se diseno para que Persona 2 pueda paralelizar por etapas y Persona 3 pueda insertar modelo de islas sin reescribir la base.

Fragmento relacionado (orquestacion inicial en `main`):

```cpp
CliArguments args(argc, argv);
auto instance_result =
    load_problem_instance(args.input_file, constants::DEFAULT_CAPACITY_RATIO);

GeneticAlgorithm ga(instance, build_config(args), args.seed);
const GARunResult run_result = ga.run();
```

---

Siguiente: [02-modelado-de-datos-y-restricciones.md](./02-modelado-de-datos-y-restricciones.md)
