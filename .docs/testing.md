# Testing — validate_instance

La suite de tests valida el correcto funcionamiento del algoritmo genético usando una instancia
mínima de 5 ítems cuyo óptimo puede calcularse a mano.

## Cómo ejecutar

```bash
make test
```

Esto compila ambos ejecutables y corre los dos tests con CTest:

```
100% tests passed, 0 tests failed out of 2
```

---

## Los dos tests

### 1. `fitness_unit` — evaluador de fitness (53 asserts)

Binario: `build/test_fitness`  
Fuente: [tests/test_fitness.cpp](../tests/test_fitness.cpp)

Construye la instancia directamente en memoria (sin leer CSV) y llama a `FitnessEvaluator::evaluate`
con cromosomas fijos. Para cada uno verifica `value`, `weight`, `volume`, `penalty`, `fitness` y
`feasible_hard` contra los valores calculados a mano.

### 2. `ga_integration` — convergencia de las tres variantes (9 asserts)

Script: [tests/test_integration.sh](../tests/test_integration.sh)

Corre el binario `mochila_ga` con semilla fija sobre `data/validate_instance` y verifica que
las tres variantes encuentren el óptimo conocido:

| Variante     | threads |
|--------------|---------|
| `sequential` | 1       |
| `parallel`   | 2       |
| `islands`    | 4       |

Para cada variante comprueba: `factible=si`, `fitness=55.000`, `valor total=55.000`.

---

## La instancia de validación

Directorio: [data/validate_instance/](../data/validate_instance/)

### Ítems

| idx | id | value | weight | volume | category |
|-----|----|------:|-------:|-------:|----------|
| 0   | 1  |    10 |      3 |      2 | A        |
| 1   | 2  |    20 |      4 |      3 | A        |
| 2   | 3  |    30 |      5 |      4 | B        |
| 3   | 4  |    15 |      6 |      5 | B        |
| 4   | 5  |    25 |      2 |      2 | C        |

`total_weight=20`, `total_volume=16`, `capacity_ratio=0.40`  
→ `max_weight=8.0`, `max_volume=6.4`

### Restricciones

| Tipo             | Detalle                                          |
|------------------|--------------------------------------------------|
| Categoría        | Categoría A: mínimo 0, máximo 1 ítem             |
| Incompatibilidad | Ítems 2 y 3 no pueden seleccionarse juntos       |
| Dependencia      | Si se selecciona el ítem 3, debe incluirse el 5  |

### Solución óptima

**Genes `[0, 0, 1, 0, 1]`** — ítems 3 y 5 seleccionados.

```
value  = 30 + 25 = 55
weight = 5  +  2 = 7   ≤ max_weight (8.0)  ✓
volume = 4  +  2 = 6   ≤ max_volume (6.4)  ✓
Categoría A  = 0 ítems  → OK
Incompatible = 0        → OK
Dependencia  = ítem 3 seleccionado, ítem 5 seleccionado → OK

penalty = 0
fitness = 55 − 0 = 55
```

---

## Casos de prueba del test unitario

Las penalizaciones usadas: `α=8 β=8 γ=3 δ=40 ε=40`

| Test | Genes           | Qué ejercita                            | fitness  | feasible_hard |
|------|-----------------|-----------------------------------------|---------:|:-------------:|
| T1   | `[0,0,0,0,0]`  | Cromosoma vacío                         |    0     | ✓             |
| T2   | `[0,0,1,0,1]`  | Óptimo — ítems 3 y 5                   |   55     | ✓             |
| T3   | `[1,0,1,0,1]`  | Exceso de peso y volumen                |   36.2   | ✗             |
| T4   | `[0,1,1,0,0]`  | Incompatibilidad + dependencia + capac. |  −42.8   | ✗             |
| T5   | `[1,1,0,0,0]`  | Violación de categoría A (penalidad γ)  |   27     | ✓ *           |
| T6   | `[0,1,0,0,1]`  | Factible pero subóptimo                 |   45     | ✓             |

> \* Las violaciones de categoría son penalidad **blanda**: reducen el fitness pero no marcan
> `feasible_hard=false`. Solo peso, volumen, incompatibilidades y dependencias son restricciones
> duras en este modelo.

### Cálculo detallado — T4 (peor caso)

```
genes = [0, 1, 1, 0, 0]   → ítems 2 y 3

value  = 20 + 30 = 50
weight =  4 +  5 = 9   →  exceso = 9 − 8.0  = 1.0
volume =  3 +  4 = 7   →  exceso = 7 − 6.4  = 0.6
incompatibilidad(2,3)   = 1 violación
dependencia(3 sin 5)    = 1 violación

penalty = α·1.0 + β·0.6 + δ·1 + ε·1
        = 8·1   + 8·0.6 + 40·1 + 40·1
        =  8    +  4.8  +  40  +  40   = 92.8

fitness = 50 − 92.8 = −42.8
```
