BUILD_DIR := build
TARGET := mochila_ga
CONFIG ?= Debug
instance ?= data/medium
variant ?= islands
threads ?= 8
seed ?= 123

.PHONY: help configure build run watch rebuild clean

help:
	@printf "Targets disponibles:\n"
	@printf "  make configure              Genera archivos de build con CMake\n"
	@printf "  make build                  Compila el proyecto\n"
	@printf "  make run instance=\"...\" variant=\"...\" threads=\"...\" seed=\"...\"\n"
	@printf "  make watch                  Recompila y ejecuta al guardar cambios\n"
	@printf "  make rebuild                Limpia y recompila\n"
	@printf "  make clean                  Elimina build local\n"
	@printf "\nEjemplo:\n"
	@printf "  make run instance=\"data/medium\" variant=\"islands\" threads=\"8\" seed=\"123\"\n"

configure:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CONFIG)

build: configure
	cmake --build $(BUILD_DIR) -j

run: build
	./$(BUILD_DIR)/$(TARGET) --instance $(instance) --variant $(variant) --threads $(threads) --seed $(seed) -v

watch:
	@command -v entr >/dev/null 2>&1 || { printf "Instala 'entr' para usar watch mode (sudo apt install entr).\n"; exit 1; }
	@printf "Watch mode activo. Guarda cambios en src/, include/ o CMakeLists.txt para recompilar y ejecutar.\n"
	@{ git ls-files "*.cpp" "*.cc" "*.cxx" "*.h" "*.hpp" 2>/dev/null; printf "%s\n" CMakeLists.txt; } | sort -u | entr -c sh -lc 'cmake --build $(BUILD_DIR) -j && ./$(BUILD_DIR)/$(TARGET) --instance "$(instance)" --variant "$(variant)" --threads "$(threads)" --seed "$(seed)" -v'

rebuild: clean build

clean:
	rm -rf $(BUILD_DIR)
