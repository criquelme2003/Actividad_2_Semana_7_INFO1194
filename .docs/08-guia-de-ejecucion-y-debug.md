# Guia de ejecucion y debug

Compilar:

```bash
cmake --preset dev
cmake --build --preset dev
```

Ejecutar variante secuencial:

```bash
./build/mochila_ga --instance data/instance_small --variant sequential --seed 123 --threads 1 -v
```

Parametros CLI en esta etapa:

- `--instance`
- `--variant`
- `--threads`
- `--seed`

Los hiperparametros del GA (poblacion, generaciones, torneo, elitismo y tasas)
estan definidos como constantes internas en `include/constants.hpp`.
La semilla por defecto del CLI tambien esta centralizada alli (`DEFAULT_SEED = 123`).

Tips de debug:

- iniciar con poblaciones pequenas;
- subir verbose para confirmar parametros;
- revisar `feasible_hard` y `penalty` para validar restricciones;
- fijar `--seed` para reproducir errores.

Salida esperada (ejemplo resumido):

```text
Generaciones ejecutadas: 57
Mejor fitness: 69.000
Mejor valor total: 69.000
Factible (restricciones duras): si
Penalizacion total: 0.000
Mejor fitness factible: 69.000
```

---

Siguiente: [README.md](./README.md)
