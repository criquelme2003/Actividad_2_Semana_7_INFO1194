BUILD_DIR := build
TARGET := mochila_ga
CONFIG ?= Debug
ARGS ?= --instance data/medium --variant islands --threads 8 --seed 123

.PHONY: help configure build run rebuild clean

help:
	@printf "Targets disponibles:\n"
	@printf "  make configure              Genera archivos de build con CMake\n"
	@printf "  make build                  Compila el proyecto\n"
	@printf "  make run ARGS=\"...\"        Ejecuta el binario con argumentos\n"
	@printf "  make rebuild                Limpia y recompila\n"
	@printf "  make clean                  Elimina build local\n"
	@printf "\nEjemplo:\n"
	@printf "  make run ARGS=\"--instance data/medium --variant islands --threads 8 --seed 123\"\n"

configure:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CONFIG)

build: configure
	cmake --build $(BUILD_DIR) -j

run: build
	./$(BUILD_DIR)/$(TARGET) $(ARGS)

rebuild: clean build

clean:
	rm -rf $(BUILD_DIR)
