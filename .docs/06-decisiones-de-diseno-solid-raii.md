# Decisiones de diseno (SOLID y RAII)

Aplicaciones practicas:

- Responsabilidad unica:
  - `InstanceLoader`: lectura y construccion de instancia.
  - `FitnessEvaluator`: solo evaluacion.
  - `GeneticAlgorithm`: solo evolucion y orquestacion.
- Abierto/cerrado:
  - operadores y seleccion desacoplados en modulos separados.
- Inversion de dependencias pragmatica:
  - `GeneticAlgorithm` depende de interfaces de comportamiento simples (funciones y structs), no de I/O.

RAII:

- uso exclusivo de contenedores STL y objetos automaticos;
- sin `new/delete` manual;
- recursos cerrados por ciclo de vida automatico.

Sobre comentarios:

- se priorizo codigo autoexplicativo;
- se evita comentario redundante respecto a lo evidente.

Fragmento que muestra RAII y ausencia de manejo manual de memoria:

```cpp
std::vector<Chromosome> next_population;
next_population.reserve(population.size());

next_population.push_back(std::move(child_a));
if (next_population.size() < population.size()) {
    next_population.push_back(std::move(child_b));
}
```

---

Siguiente: [07-integracion-con-el-proyecto.md](./07-integracion-con-el-proyecto.md)
