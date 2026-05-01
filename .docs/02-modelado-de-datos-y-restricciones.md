# Modelado de datos y restricciones

El modelo principal se define en `include/models.hpp`.

- `Item`: id, valor, peso, volumen, categoria.
- `CategoryRule`: minimo y maximo por categoria.
- `ProblemInstance`: items, reglas y restricciones de par/inclusion.
- `Chromosome`: vector binario (`0` no seleccionado, `1` seleccionado).

Restricciones implementadas:

- Duras: peso, volumen, incompatibilidades y dependencias.
- Blandas: categoria (se penaliza, no invalida por si sola).

La carga de instancia se realiza desde carpeta con estos archivos:

- `items.csv` (obligatorio)
- `category_rules.csv` (opcional)
- `incompatibilities.csv` (opcional)
- `dependencies.csv` (opcional)

El loader calcula capacidades `max_weight` y `max_volume` como porcentaje del total disponible (ratio configurable).

Fragmentos clave de modelado:

```cpp
struct Item {
    int id;
    double value;
    double weight;
    double volume;
    std::string category;
};
```

```cpp
struct ProblemInstance {
    std::vector<Item> items;
    std::vector<CategoryRule> category_rules;
    std::vector<std::pair<std::size_t, std::size_t>> incompatibility_indices;
    std::vector<std::pair<std::size_t, std::size_t>> dependency_indices;
    double max_weight;
    double max_volume;
};
```

---

Siguiente: [03-diseno-del-ga-secuencial.md](./03-diseno-del-ga-secuencial.md)
