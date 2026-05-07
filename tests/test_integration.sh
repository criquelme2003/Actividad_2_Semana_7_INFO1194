#!/usr/bin/env bash
# Integration test: runs each GA variant on validate_instance and checks that
# all three converge to the known optimal (value=55, fitness=55, feasible).
#
# Optimal solution: items 3+5 selected → value=30+25=55, no violations.
# Instance is tiny (5 items) so any variant must find it reliably.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BINARY="${1:-$PROJECT_ROOT/build/mochila_ga}"
INSTANCE="$PROJECT_ROOT/data/validate_instance"
SEED=123

PASS=0
FAIL=0

# Colors (disabled when not a terminal)
if [ -t 1 ]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; RESET='\033[0m'
else
    RED=''; GREEN=''; RESET=''
fi

check() {
    local label="$1" value="$2" expected="$3"
    if [ "$value" = "$expected" ]; then
        printf "  ${GREEN}[PASS]${RESET} %-45s  got='%s'\n" "$label" "$value"
        ((PASS++)) || true
    else
        printf "  ${RED}[FAIL]${RESET} %-45s  got='%s'  expected='%s'\n" "$label" "$value" "$expected"
        ((FAIL++)) || true
    fi
}

run_variant() {
    local variant="$1" threads="$2"
    printf "\n--- Variante: %s (threads=%s, seed=%s) ---\n" "$variant" "$threads" "$SEED"

    local output
    output=$("$BINARY" --instance "$INSTANCE" --variant "$variant" \
                       --threads "$threads" --seed "$SEED" 2>&1)

    local feasible fitness value
    feasible=$(printf '%s' "$output" | grep -oP 'Factible.*?: \K\S+')
    fitness=$(printf '%s' "$output"  | grep -oP 'Mejor fitness: \K[\d.]+' | head -1)
    value=$(printf '%s' "$output"    | grep -oP 'Mejor valor total: \K[\d.]+')

    check "$variant: factible"    "$feasible" "si"
    check "$variant: fitness"     "$fitness"  "55.000"
    check "$variant: valor total" "$value"    "55.000"
}

printf '=== test_integration: validate_instance ===\n'
printf 'Optimo conocido: items {3,5} → valor=55, fitness=55, factible=si\n'

run_variant "sequential" 1
run_variant "parallel"   2
run_variant "islands"    4

printf '\n=== Resultado: %d/%d tests pasaron ===\n' "$PASS" "$((PASS + FAIL))"
[ "$FAIL" -eq 0 ]
