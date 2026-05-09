// Unit tests for FitnessEvaluator using validate_instance (5 items, hand-calculated).
//
// Instance layout:
//   idx  id  value  weight  volume  category
//    0    1    10      3       2       A
//    1    2    20      4       3       A
//    2    3    30      5       4       B
//    3    4    15      6       5       B
//    4    5    25      2       2       C
//
//   total_weight=20, total_volume=16, capacity_ratio=0.40
//   max_weight = 8.0,  max_volume = 6.4
//
//   category_rules:  A: min=0, max=1
//   incompatibilities: (2,3) → indices (1,2)
//   dependencies:    item 3 requires item 5 → indices (2,4)
//
//   penalties: alpha=8, beta=8, gamma=3, delta=40, epsilon=40
//
// Hand-calculated expected values (see tests/README-validate.md for workings):
//
//   T1  [0,0,0,0,0]  → fitness=0,     feasible=true
//   T2  [0,0,1,0,1]  → fitness=55,    feasible=true   ← OPTIMAL
//   T3  [1,0,1,0,1]  → fitness=36.2,  feasible=false  (weight+volume excess)
//   T4  [0,1,1,0,0]  → fitness=-42.8, feasible=false  (incompat+dep+capacity)
//   T5  [1,1,0,0,0]  → fitness=27,    feasible=true   (only category soft penalty)
//   T6  [0,1,0,0,1]  → fitness=45,    feasible=true

#include "constants.hpp"
#include "fitness.hpp"
#include "models.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static int g_pass = 0;
static int g_fail = 0;

static void check_double(const char *label, double got, double expected, double tol = 1e-9) {
    if (std::abs(got - expected) <= tol) {
        std::printf("  [PASS] %-50s  got=%.4f\n", label, got);
        ++g_pass;
    } else {
        std::printf("  [FAIL] %-50s  got=%.4f  expected=%.4f\n", label, got, expected);
        ++g_fail;
    }
}

static void check_bool(const char *label, bool got, bool expected) {
    if (got == expected) {
        std::printf("  [PASS] %-50s  got=%s\n", label, got ? "true" : "false");
        ++g_pass;
    } else {
        std::printf("  [FAIL] %-50s  got=%s  expected=%s\n", label, got ? "true" : "false",
                    expected ? "true" : "false");
        ++g_fail;
    }
}

static void check_int(const char *label, int got, int expected) {
    if (got == expected) {
        std::printf("  [PASS] %-50s  got=%d\n", label, got);
        ++g_pass;
    } else {
        std::printf("  [FAIL] %-50s  got=%d  expected=%d\n", label, got, expected);
        ++g_fail;
    }
}

// ---------------------------------------------------------------------------
// Build the validate_instance in memory (no file I/O)
// ---------------------------------------------------------------------------

static ProblemInstance make_instance() {
    ProblemInstance inst;

    inst.items = {
        {1, 10.0, 3.0, 2.0, "A"},
        {2, 20.0, 4.0, 3.0, "A"},
        {3, 30.0, 5.0, 4.0, "B"},
        {4, 15.0, 6.0, 5.0, "B"},
        {5, 25.0, 2.0, 2.0, "C"},
    };

    inst.id_to_index = {{1, 0}, {2, 1}, {3, 2}, {4, 3}, {5, 4}};

    // total_weight=20, total_volume=16, capacity_ratio=0.40
    inst.max_weight = 8.0;
    inst.max_volume = 6.4;

    inst.category_rules = {{"A", 0, 1}};

    inst.incompatibilities = {{2, 3}};
    inst.incompatibility_indices = {{1, 2}};

    inst.dependencies = {{3, 5}};
    inst.dependency_indices = {{2, 4}};

    return inst;
}

static PenaltyConfig make_penalties() {
    return PenaltyConfig{
        .alpha      = constants::PENALTY_ALPHA,
        .beta       = constants::PENALTY_BETA,
        .w_weight   = constants::PENALTY_W_WEIGHT,
        .w_volume   = constants::PENALTY_W_VOLUME,
        .w_category = constants::PENALTY_W_CATEGORY,
        .w_incompat = constants::PENALTY_W_INCOMPAT,
        .w_dep      = constants::PENALTY_W_DEP
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// T1: empty — nothing selected
// Expected: value=0, weight=0, volume=0, no violations, fitness=0, feasible=true
static void test_t1_empty(const FitnessEvaluator &eval) {
    std::printf("\nT1: genes=[0,0,0,0,0] — nada seleccionado\n");
    Chromosome chr{{0, 0, 0, 0, 0}};
    auto r = eval.evaluate(chr);

    check_double("total_value",               r.total_value,              0.0);
    check_double("total_weight",              r.total_weight,             0.0);
    check_double("total_volume",              r.total_volume,             0.0);
    check_int   ("category_violations",       r.category_violations,      0);
    check_int   ("incompatibility_violations",r.incompatibility_violations,0);
    check_int   ("dependency_violations",     r.dependency_violations,    0);
    check_double("penalty",                   r.penalty,                  0.0);
    check_double("fitness",                   r.fitness,                  0.0);
    check_bool  ("feasible_hard",             r.feasible_hard,            true);
}

// T2: items 3+5 — optimal feasible solution
//   value=55, weight=7, volume=6, no violations, penalty=0, fitness=55
static void test_t2_optimal(const FitnessEvaluator &eval) {
    std::printf("\nT2: genes=[0,0,1,0,1] — OPTIMO (items 3 y 5)\n");
    Chromosome chr{{0, 0, 1, 0, 1}};
    auto r = eval.evaluate(chr);

    check_double("total_value",               r.total_value,              55.0);
    check_double("total_weight",              r.total_weight,             7.0);
    check_double("total_volume",              r.total_volume,             6.0);
    check_int   ("category_violations",       r.category_violations,      0);
    check_int   ("incompatibility_violations",r.incompatibility_violations,0);
    check_int   ("dependency_violations",     r.dependency_violations,    0);
    check_double("penalty",                   r.penalty,                  0.0);
    check_double("fitness",                   r.fitness,                  55.0);
    check_bool  ("feasible_hard",             r.feasible_hard,            true);
}

// T3: items 1+3+5 — over capacity
//   value=65, weight=10 (excess=2), volume=8 (excess=1.6)
//   penalty = 8*2 + 8*1.6 = 16 + 12.8 = 28.8
//   fitness = 65 - 28.8 = 36.2
static void test_t3_over_capacity(const FitnessEvaluator &eval) {
    std::printf("\nT3: genes=[1,0,1,0,1] — exceso de peso y volumen (items 1,3,5)\n");
    Chromosome chr{{1, 0, 1, 0, 1}};
    auto r = eval.evaluate(chr);

    check_double("total_value",               r.total_value,              65.0);
    check_double("total_weight",              r.total_weight,             10.0);
    check_double("total_volume",              r.total_volume,             8.0);
    check_int   ("category_violations",       r.category_violations,      0);
    check_int   ("incompatibility_violations",r.incompatibility_violations,0);
    check_int   ("dependency_violations",     r.dependency_violations,    0);
    check_double("penalty",                   r.penalty,                  28.8, 1e-9);
    check_double("fitness",                   r.fitness,                  36.2, 1e-9);
    check_bool  ("feasible_hard",             r.feasible_hard,            false);
}

// T4: items 2+3 — incompatibility + dependency + over capacity
//   value=50, weight=9 (excess=1), volume=7 (excess=0.6)
//   incompat(2,3)=1, dep(3 sin 5)=1
//   penalty = 8*1 + 8*0.6 + 40*1 + 40*1 = 8 + 4.8 + 40 + 40 = 92.8
//   fitness = 50 - 92.8 = -42.8
static void test_t4_all_violations(const FitnessEvaluator &eval) {
    std::printf("\nT4: genes=[0,1,1,0,0] — incompatibilidad + dependencia + capacidad (items 2,3)\n");
    Chromosome chr{{0, 1, 1, 0, 0}};
    auto r = eval.evaluate(chr);

    check_double("total_value",               r.total_value,              50.0);
    check_double("total_weight",              r.total_weight,             9.0);
    check_double("total_volume",              r.total_volume,             7.0);
    check_int   ("incompatibility_violations",r.incompatibility_violations,1);
    check_int   ("dependency_violations",     r.dependency_violations,    1);
    check_double("penalty",                   r.penalty,                  92.8, 1e-9);
    check_double("fitness",                   r.fitness,                  -42.8, 1e-9);
    check_bool  ("feasible_hard",             r.feasible_hard,            false);
}

// T5: items 1+2 — category A violation (2 items, max=1)
//   value=30, weight=7, volume=5  (within capacity)
//   category_violations=1 (A count=2 > max=1)
//   penalty = 3*1 = 3
//   fitness = 30 - 3 = 27
//   feasible_hard=true (category violations are SOFT in this model)
static void test_t5_category_violation(const FitnessEvaluator &eval) {
    std::printf("\nT5: genes=[1,1,0,0,0] — violacion de categoria A (items 1,2)\n");
    Chromosome chr{{1, 1, 0, 0, 0}};
    auto r = eval.evaluate(chr);

    check_double("total_value",               r.total_value,              30.0);
    check_double("total_weight",              r.total_weight,             7.0);
    check_double("total_volume",              r.total_volume,             5.0);
    check_int   ("category_violations",       r.category_violations,      1);
    check_int   ("incompatibility_violations",r.incompatibility_violations,0);
    check_int   ("dependency_violations",     r.dependency_violations,    0);
    check_double("penalty",                   r.penalty,                  3.0);
    check_double("fitness",                   r.fitness,                  27.0);
    check_bool  ("feasible_hard",             r.feasible_hard,            true);
}

// T6: items 2+5 — within capacity, no violations
//   value=45, weight=6, volume=5, penalty=0, fitness=45
static void test_t6_feasible_suboptimal(const FitnessEvaluator &eval) {
    std::printf("\nT6: genes=[0,1,0,0,1] — factible suboptimo (items 2,5)\n");
    Chromosome chr{{0, 1, 0, 0, 1}};
    auto r = eval.evaluate(chr);

    check_double("total_value",               r.total_value,              45.0);
    check_double("total_weight",              r.total_weight,             6.0);
    check_double("total_volume",              r.total_volume,             5.0);
    check_int   ("category_violations",       r.category_violations,      0);
    check_int   ("incompatibility_violations",r.incompatibility_violations,0);
    check_int   ("dependency_violations",     r.dependency_violations,    0);
    check_double("penalty",                   r.penalty,                  0.0);
    check_double("fitness",                   r.fitness,                  45.0);
    check_bool  ("feasible_hard",             r.feasible_hard,            true);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    std::printf("=== test_fitness: validate_instance ===\n");
    std::printf("Instancia: 5 items, max_weight=8.0, max_volume=6.4\n");
    std::printf("Reglas: A max=1 | Incompatible: (2,3) | Dependencia: 3->5\n");
    std::printf("Penalizaciones: alpha=8 beta=8 gamma=3 delta=40 epsilon=40\n");

    const ProblemInstance inst = make_instance();
    const FitnessEvaluator eval(inst, make_penalties());

    test_t1_empty(eval);
    test_t2_optimal(eval);
    test_t3_over_capacity(eval);
    test_t4_all_violations(eval);
    test_t5_category_violation(eval);
    test_t6_feasible_suboptimal(eval);

    std::printf("\n=== Resultado: %d/%d tests pasaron ===\n", g_pass, g_pass + g_fail);

    return g_fail == 0 ? 0 : 1;
}
