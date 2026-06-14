#pragma once

namespace constants {

inline constexpr int DEFAULT_GENERATIONS = 200;
inline constexpr int DEFAULT_POPULATION_SIZE = 1024;
inline constexpr int DEFAULT_ELITISM_COUNT = 2;
inline constexpr int RANK_SELECTION_COUNT = 10;
inline constexpr double RANK_SELECTION_P = 0.5;
inline constexpr int DEFAULT_SEED = 123;
inline constexpr int MAX_STAGNATION_GENERATIONS = 50;

inline constexpr double DEFAULT_CROSSOVER_RATE = 0.85;
inline constexpr double DEFAULT_MUTATION_RATE = 0.02;
inline constexpr double DEFAULT_CAPACITY_RATIO = 0.40;

// ─── Constantes CUDA ──────────────────────────────────────────────────────────
inline constexpr int DEFAULT_BLOCK_SIZE = 256;
inline constexpr int DEFAULT_TOURNAMENT_SIZE = 3;

// Tolerancia para comparaciones de punto flotante en verificación de factibilidad
inline constexpr double FEASIBILITY_EPSILON = 1e-9;

// Nivel 1: balance valor vs. penalización (α + β = 1)
inline constexpr double PENALTY_ALPHA = 0.6; // α: peso del valor normalizado
inline constexpr double PENALTY_BETA  = 0.4; // β: peso de la violación normalizada
// Nivel 2: pesos por tipo de restricción (Σwi = 1)
inline constexpr double PENALTY_W_WEIGHT   = 0.30; // w1: exceso de peso
inline constexpr double PENALTY_W_VOLUME   = 0.30; // w2: exceso de volumen
inline constexpr double PENALTY_W_CATEGORY = 0.05; // w3: violaciones de categoría (blanda)
inline constexpr double PENALTY_W_INCOMPAT = 0.20; // w4: incompatibilidades
inline constexpr double PENALTY_W_DEP      = 0.15; // w5: dependencias incumplidas

} // namespace constants
