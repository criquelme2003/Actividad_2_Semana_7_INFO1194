#pragma once

namespace constants {

inline constexpr int DEFAULT_GENERATIONS = 200;
inline constexpr int DEFAULT_POPULATION_SIZE = 120;
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

inline constexpr double PENALTY_ALPHA = 8.0;
inline constexpr double PENALTY_BETA = 8.0;
inline constexpr double PENALTY_GAMMA = 3.0;
inline constexpr double PENALTY_DELTA = 40.0;
inline constexpr double PENALTY_EPSILON = 40.0;

} // namespace constants
