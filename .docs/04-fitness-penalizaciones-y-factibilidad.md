# Fitness, penalizaciones y factibilidad

Implementado en `src/fitness.cpp` mediante `FitnessEvaluator`.

Formula usada:

`fitness = valor_total - penalizacion_total`

Con:

- `alpha * exceso_peso`
- `beta * exceso_volumen`
- `gamma * violaciones_categoria`
- `delta * incompatibilidades`
- `epsilon * dependencias_incumplidas`

Factibilidad dura:

- exceso de peso = 0
- exceso de volumen = 0
- incompatibilidades = 0
- dependencias incumplidas = 0

`FitnessBreakdown` guarda detalle para analisis y debugging (valor, violaciones, penalty, fitness, feasible_hard).

Fragmento representativo:

```cpp
out.penalty =
    penalties_.alpha * excess_weight + penalties_.beta * excess_volume +
    penalties_.gamma * static_cast<double>(out.category_violations) +
    penalties_.delta * static_cast<double>(out.incompatibility_violations) +
    penalties_.epsilon * static_cast<double>(out.dependency_violations);

out.fitness = out.total_value - out.penalty;
out.feasible_hard = (excess_weight <= 0.0) && (excess_volume <= 0.0) &&
                    out.incompatibility_violations == 0 &&
                    out.dependency_violations == 0;
```

---

Siguiente: [05-seleccion-crossover-mutacion-elitismo.md](./05-seleccion-crossover-mutacion-elitismo.md)
