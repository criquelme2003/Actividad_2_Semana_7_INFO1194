# Seleccion, crossover, mutacion y elitismo

## Seleccion por torneo

`src/selection.cpp` implementa torneo de tamano fijo.

- se sortean `k` indices;
- gana el mayor fitness;
- retorna indice del padre seleccionado.

## Crossover

`src/operators.cpp` implementa crossover de un punto:

- si no se cumple `crossover_rate`, se copian padres;
- si se cumple, se intercambia la cola desde punto aleatorio.

## Mutacion

Mutacion bit-flip por gen:

- cada gen muta con probabilidad `mutation_rate`.

## Elitismo

El motor GA preserva los mejores `elitism_count` individuos de cada generacion sin alteracion.

Fragmentos breves:

```cpp
const std::size_t idx_a = tournament_select_index(evaluated, config_.tournament_size, rng_);
const std::size_t idx_b = tournament_select_index(evaluated, config_.tournament_size, rng_);
```

```cpp
auto [child_a, child_b] = crossover_one_point(
    evaluated[idx_a].chromosome,
    evaluated[idx_b].chromosome,
    config_.crossover_rate,
    rng_);

mutate_bit_flip(child_a, config_.mutation_rate, rng_);
mutate_bit_flip(child_b, config_.mutation_rate, rng_);
```

---

Siguiente: [06-decisiones-de-diseno-solid-raii.md](./06-decisiones-de-diseno-solid-raii.md)
