# Variante 3 — Modelo de Islas: diseño, cambios y funcionamiento

---

## 1. Diseño anterior (antes del rediseño)

### Arquitectura

El modelo de islas original usaba `ParallelGeneticAlgorithm` dentro de cada isla. Esto
introducía **dos niveles de paralelismo anidados**:

```
IslandModel
  ├── num_islands_       = DEFAULT_NUM_ISLANDS (fijo = 4, siempre)
  ├── threads_per_island_ = num_threads / 4
  └── islands            = vector<ParallelGeneticAlgorithm>
                              └── cada isla usaba OpenMP internamente
```

Cuando el usuario pasaba `--threads 8`:

- Se creaban **4 islas fijas**
- Cada isla recibía `threads_per_island = 8 / 4 = 2` hilos internos
- Se activaba `omp_set_max_active_levels(2)` para permitir paralelismo anidado

### Diagrama de hilos (diseño anterior, `--threads 8`)

```
Nivel 1 (entre islas):
  Hilo 0 → isla 0 (ParallelGA con 2 hilos internos)
  Hilo 2 → isla 1 (ParallelGA con 2 hilos internos)
  Hilo 4 → isla 2 (ParallelGA con 2 hilos internos)
  Hilo 6 → isla 3 (ParallelGA con 2 hilos internos)

Nivel 2 (dentro de cada isla):
  isla 0: hilo 0 + hilo 1  ← #pragma omp parallel for interno
  isla 1: hilo 2 + hilo 3
  isla 2: hilo 4 + hilo 5
  isla 3: hilo 6 + hilo 7
```

### Problema del diseño anterior

Con `--threads 1`, `--threads 2` o `--threads 4`, el número de islas era **siempre 4**.
Solo cambiaba cuántos hilos internos tenía cada isla. Resultado: los experimentos con t=1,
t=2 y t=4 producían tiempos casi idénticos y resultados prácticamente iguales, lo que hacía
que los datos experimentales no tuvieran diferencias significativas que analizar.

| `--threads` | Islas | Hilos por isla  | ¿Resultados distintos? |
| :---------: | :---: | :-------------: | :--------------------: |
|      1      |   4   | 0.25 (inválido) |           No           |
|      2      |   4   | 0.5 (inválido)  |           No           |
|      4      |   4   |        1        |           No           |
|      8      |   4   |        2        |           Sí           |

---

## 2. Rediseño nuevo (diseño actual)

El profesor indicó que el modelo de islas debe implementarse con **paralelismo grueso**:
cada isla es un GA secuencial independiente, y el paralelismo ocurre entre islas, no
dentro de ellas.

### Arquitectura nueva

```
IslandModel
  ├── num_islands_  = args.num_threads  (variable, no fijo)
  └── islands       = vector<GeneticAlgorithm>   ← GA secuencial, no paralelo
```

No existe `threads_per_island_`. No existe `omp_set_max_active_levels`. Un único nivel
de paralelismo: el `#pragma omp parallel for` del bucle principal de `IslandModel::run()`.

### Tabla comparativa

| Aspecto                         | Antes                      | Ahora                           |
| ------------------------------- | -------------------------- | ------------------------------- |
| GA dentro de cada isla          | `ParallelGeneticAlgorithm` | `GeneticAlgorithm` (secuencial) |
| Niveles de paralelismo          | 2 (anidado)                | 1 (grueso, solo entre islas)    |
| Número de islas                 | Fijo = 4 siempre           | Variable = `--threads`          |
| `--threads 1`                   | 4 islas, 0.25 hilos c/u    | **1 isla**, 1 hilo              |
| `--threads 4`                   | 4 islas, 1 hilo c/u        | **4 islas**, 1 hilo c/u         |
| `--threads 8`                   | 4 islas, 2 hilos c/u       | **8 islas**, 1 hilo c/u         |
| `omp_set_max_active_levels(2)`  | Sí                         | No (eliminado)                  |
| Resultados distintos por t=1..8 | No                         | **Sí**                          |

---

## 3. Cambios realizados en los archivos

### `include/genetic_algorithm.hpp`

Se añadió estado persistente y una interfaz pública para que `IslandModel` pueda
controlar el GA generación a generación.

**Nuevos miembros privados:**

```cpp
std::vector<Chromosome> population_;   // sobrevive entre llamadas a run_n_generations
EvaluatedIndividual best_by_fitness_;  // mejor global rastreado a lo largo del tiempo
EvaluatedIndividual best_feasible_;    // mejor factible rastreado
bool has_feasible_ = false;
```

Antes, `population_` era una variable local dentro de `run()` y moría al terminar la
función. Ahora es miembro de la clase, lo que permite al GA "pausar y continuar" entre
batches de generaciones.

**Nuevos métodos públicos:**

```cpp
void run_n_generations(int n);                          // evoluciona exactamente n gen
void inject_individuals(const std::vector<Chromosome>&); // recibe migrantes (reemplaza peores)
EvaluatedIndividual get_best() const;
EvaluatedIndividual get_best_feasible() const;
bool has_feasible_solution() const;
std::vector<Chromosome> get_top_n(int n) const;        // selecciona los n mejores
```

### `src/genetic_algorithm.cpp`

**`run_n_generations(n)`**: igual al bucle de `run()` pero sin criterio de parada por
estancamiento. Esto es necesario porque el modelo de islas no puede detener una isla a
mitad del ciclo — todas deben completar su batch antes de la migración.

**`inject_individuals(migrants)`**: evalúa la población, la ordena, y reemplaza los
**peores** individuos con los migrantes recibidos. Los mejores se conservan.

**`get_top_n(n)`**: evalúa y ordena la población, devuelve los `n` mejores cromosomas.
Estos son los migrantes que la isla enviará en la migración.

### `include/island_model.hpp`

- `#include "parallel_genetic_algorithm.hpp"` → reemplazado por `#include "genetic_algorithm.hpp"`
- `vector<ParallelGeneticAlgorithm>` → `vector<GeneticAlgorithm>`
- Eliminado el parámetro y miembro `threads_per_island_`
- Actualizado el comentario de clase para reflejar paralelismo grueso

### `src/island_model.cpp`

**Constructor**: cada isla recibe una semilla derivada de la semilla base:

```cpp
islands.emplace_back(instance_, island_config_, base_seed_ + k * 1000003);
```

El factor `1000003` (primo grande) garantiza que las secuencias Mersenne Twister de
cada isla sean ortogonales entre sí.

**`run()`**: eliminado `omp_set_max_active_levels(2)`. El único pragma OpenMP es:

```cpp
#pragma omp parallel for schedule(static) num_threads(num_islands_)
for (int k = 0; k < num_islands_; ++k) {
    islands[k].run_n_generations(migration_interval_);
}
```

**`perform_migration()`**: sin cambios de concepto, pero ahora opera sobre
`vector<GeneticAlgorithm>` en lugar de `vector<ParallelGeneticAlgorithm>`.

### `src/main.cpp`

```cpp
// Antes:
int threads_per_island = args.num_threads / DEFAULT_NUM_ISLANDS;
IslandModel islands_model(instance, build_config(), DEFAULT_NUM_ISLANDS,
                          DEFAULT_MIGRATION_INTERVAL, DEFAULT_MIGRANTS_PER_ISLAND,
                          threads_per_island, args.seed);

// Ahora:
IslandModel islands_model(instance, build_config(), args.num_threads,
                          DEFAULT_MIGRATION_INTERVAL, DEFAULT_MIGRANTS_PER_ISLAND,
                          args.seed);
```

`args.num_threads` es directamente el número de islas. Sin cálculo de `threads_per_island`.

---

## 4. Cómo funciona el modelo de islas actualmente

### Parámetros de diseño

| Parámetro               | Valor                     | Constante                     |
| ----------------------- | ------------------------- | ----------------------------- |
| Número de islas         | 4 (diseño)                | `DEFAULT_NUM_ISLANDS = 4`     |
| Población por isla      | 120 individuos            | `DEFAULT_POPULATION_SIZE`     |
| Frecuencia de migración | Cada 25 gen               | `DEFAULT_MIGRATION_INTERVAL`  |
| Migrantes por isla      | 2 individuos              | `DEFAULT_MIGRANTS_PER_ISLAND` |
| Criterio de migrantes   | Los 2 mejores por fitness | `get_top_n(2)`                |
| Topología               | Anillo                    | isla k → isla (k+1) % N       |

### Flujo de ejecución con 4 islas, 200 generaciones

```
Creación:
  isla 0 (seed = base + 0)
  isla 1 (seed = base + 1000003)
  isla 2 (seed = base + 2000006)
  isla 3 (seed = base + 3000009)

Ciclo (se repite 8 veces: 200 / 25 = 8 migraciones):
┌──────────────────────────────────────────────────────────────┐
│  #pragma omp parallel for num_threads(4)                     │
│  isla 0: run_n_generations(25)  │  isla 1: run_n_generations(25) │
│  isla 2: run_n_generations(25)  │  isla 3: run_n_generations(25) │
└──── barrera implícita ──────────────────────────────────────┘
  perform_migration() [secuencial, hilo principal]:
    outbox[0] = isla 0 → top 2
    outbox[1] = isla 1 → top 2
    outbox[2] = isla 2 → top 2
    outbox[3] = isla 3 → top 2

    isla 1.inject(outbox[0])   ← isla 0 → isla 1
    isla 2.inject(outbox[1])   ← isla 1 → isla 2
    isla 3.inject(outbox[2])   ← isla 2 → isla 3
    isla 0.inject(outbox[3])   ← isla 3 → isla 0  (cierra el anillo)

Resultado final:
  Recorre las 4 islas y selecciona:
    - mejor individuo por fitness global
    - mejor individuo factible (si existe)
```

### Diagrama de tiempo

```
Tiempo →

Hilo 0: [isla 0: gen 1-25]────────|barrera|[isla 0: gen 26-50]────────|barrera|...
Hilo 1: [isla 1: gen 1-25]────────|barrera|[isla 1: gen 26-50]────────|barrera|...
Hilo 2: [isla 2: gen 1-25]────────|barrera|[isla 2: gen 26-50]────────|barrera|...
Hilo 3: [isla 3: gen 1-25]────────|barrera|[isla 3: gen 26-50]────────|barrera|...
                                    ↑
                              migración secuencial
                              (hilo 0, ~microsegundos)
```

La barrera es **implícita**: al salir del `#pragma omp parallel for`, OpenMP garantiza
que todos los hilos terminaron su batch antes de continuar. Solo entonces se ejecuta
`perform_migration()`.

---

## 5. Qué está PARALELO y qué está SECUENCIAL

### PARALELO (un único pragma)

```cpp
#pragma omp parallel for schedule(static) num_threads(num_islands_)
for (int k = 0; k < num_islands_; ++k) {
    islands[k].run_n_generations(migration_interval_);
}
```

Cada hilo ejecuta una isla completa (25 generaciones) de forma independiente. No hay
memoria compartida entre hilos durante este bloque: cada isla tiene su propio `rng_`,
su propia `population_` y sus propios contadores internos.

### SECUENCIAL (todo lo demás)

| Qué                                | Dónde                        | Por qué secuencial                            |
| ---------------------------------- | ---------------------------- | --------------------------------------------- |
| El interior de `run_n_generations` | `genetic_algorithm.cpp`      | No tiene ningún `#pragma omp`. Un hilo, un GA |
| La migración completa              | `perform_migration()`        | Ocurre tras la barrera, en el hilo principal  |
| Recolección del resultado final    | `IslandModel::run()`         | Loop secuencial sobre todas las islas         |
| Inicialización de islas            | Constructor de `IslandModel` | Se crea una isla a la vez                     |

### Regla de oro

> Lo paralelo es **entre islas** (cada isla en su propio hilo OpenMP).  
> Lo secuencial es **dentro de cada isla** (el GA completo) y **la migración**.

Esto se llama **paralelismo grueso** (_coarse-grained_): la unidad de trabajo paralelo
es todo un GA de 25 generaciones, no una operación individual.

---

## 6. Por qué este diseño es correcto

La variante 2 (`parallel`) ya se encarga del paralelismo fino dentro del GA: paraleliza
evaluación, cruzamiento y mutación. Duplicar ese paralelismo dentro de cada isla del
modelo de islas generaría:

- Paralelismo anidado (difícil de controlar, ineficiente en muchos sistemas)
- Contención de threads entre islas
- Menor beneficio real por la ley de Amdahl

El modelo de islas no busca **velocidad** sobre el secuencial — busca **mejor calidad de
solución** a través de diversidad genética entre subpoblaciones. El tradeoff es:

```
parallel t=4:  241ms    calidad media 3048   speedup 2.14×  ← más rápido
islands  t=4:  833ms    calidad media 3104   speedup 0.62×  ← mejor solución
```

Cada estrategia optimiza un objetivo distinto. Son complementarias, no competidoras.
