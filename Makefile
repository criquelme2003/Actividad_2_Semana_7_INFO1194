BUILD_DIR := build
TARGET := mochila_ga
CONFIG ?= Debug
instance ?= data/instance_small
variant ?= sequential
threads ?= 1
seed ?= 123

.PHONY: help configure build run test watch lint format-check format rebuild clean

SRC_FILES := $(shell git ls-files '*.cpp' '*.hpp' '*.h')

help:
	@printf "Targets disponibles:\n"
	@printf "  make configure              Genera archivos de build con CMake\n"
	@printf "  make build                  Compila el proyecto\n"
	@printf "  make run instance=\"...\" variant=\"...\" threads=\"...\" seed=\"...\"\n"
	@printf "  make test                   Compila y ejecuta tests unitarios (validate_instance)\n"
	@printf "  make watch                  Recompila y ejecuta al guardar cambios\n"
	@printf "  make lint                   Ejecuta clang-tidy (si esta instalado)\n"
	@printf "  make format-check           Verifica formato con clang-format\n"
	@printf "  make format                 Aplica formato con clang-format\n"
	@printf "  make rebuild                Limpia y recompila\n"
	@printf "  make clean                  Elimina build local\n"
	@printf "\nEjemplo:\n"
	@printf "  make run instance=\"data/instance_small\" variant=\"sequential\" threads=\"1\" seed=\"123\"\n"

configure:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CONFIG)

build: configure
	cmake --build $(BUILD_DIR) -j

run: build
	./$(BUILD_DIR)/$(TARGET) --instance $(instance) --variant $(variant) --threads $(threads) --seed $(seed) -v

test: configure
	cmake --build $(BUILD_DIR) -j --target test_fitness --target $(TARGET)
	ctest --test-dir $(BUILD_DIR) --output-on-failure -V

watch:
	@command -v entr >/dev/null 2>&1 || { printf "Instala 'entr' para usar watch mode (sudo apt install entr).\n"; exit 1; }
	@printf "Watch mode activo. Guarda cambios en src/, include/ o CMakeLists.txt para recompilar y ejecutar.\n"
	@{ git ls-files "*.cpp" "*.cc" "*.cxx" "*.h" "*.hpp" 2>/dev/null; printf "%s\n" CMakeLists.txt; } | sort -u | entr -c sh -lc 'cmake --build $(BUILD_DIR) -j && ./$(BUILD_DIR)/$(TARGET) --instance "$(instance)" --variant "$(variant)" --threads "$(threads)" --seed "$(seed)" -v'

rebuild: clean build

lint: configure
	@command -v clang-tidy >/dev/null 2>&1 || { printf "Instala clang-tidy para ejecutar lint (sudo apt install clang-tidy).\n"; exit 1; }
	@printf "Ejecutando clang-tidy...\n"
	@clang-tidy -p $(BUILD_DIR) \
	  -checks='-*,clang-analyzer-*,bugprone-*,performance-*,readability-*,-readability-identifier-length,-clang-analyzer-optin.cplusplus.UninitializedObject' \
	  -header-filter='^$(PWD)/(src|include)/' \
	  $(shell git ls-files 'src/*.cpp')

format-check:
	@command -v clang-format >/dev/null 2>&1 || { printf "Instala clang-format para verificar formato (sudo apt install clang-format).\n"; exit 1; }
	@printf "Verificando formato...\n"
	@clang-format --dry-run --Werror $(SRC_FILES)

format:
	@command -v clang-format >/dev/null 2>&1 || { printf "Instala clang-format para aplicar formato (sudo apt install clang-format).\n"; exit 1; }
	@printf "Aplicando formato...\n"
	@clang-format -i $(SRC_FILES)

clean:
	rm -rf $(BUILD_DIR)
