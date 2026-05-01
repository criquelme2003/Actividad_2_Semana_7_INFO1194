# Actividad 02 - INFO1194

Proyecto C++ con CMake, C++20, OpenMP y dependencias via FetchContent.

## Requisitos (Ubuntu)

Instala toolchain, CMake y OpenMP:

```bash
sudo apt update
sudo apt install -y build-essential cmake git libomp-dev
```

Opcional (recomendado para IntelliSense en Zed):

```bash
sudo apt install -y clangd
```

Opcional para watch mode:

```bash
sudo apt install -y entr
```

## Compilar y ejecutar (CMake Presets)

```bash
cmake --preset dev
cmake --build --preset dev
./build/mochila_ga --instance data/instance_small --variant sequential --threads 1 --seed 123 -v
```

## Compilar y ejecutar (Makefile de soporte)

```bash
make build
make run
make watch
```

Con argumentos personalizados:

```bash
make run instance="data/instance_small" variant="sequential" threads="1" seed="123"
```

`make run` ejecuta en modo verbose por defecto (`-v`).

## Targets del Makefile

- `make configure`: genera archivos de build en `build/`.
- `make build`: compila el proyecto.
- `make run instance="..." variant="..." threads="..." seed="..."`: ejecuta el binario.
- `make watch`: recompila y ejecuta al guardar cambios (`src/`, `include/`, headers y `CMakeLists.txt`).
- `make rebuild`: limpia y recompila.
- `make clean`: elimina `build/`.
