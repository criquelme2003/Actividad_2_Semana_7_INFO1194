#include "genetic_algorithm.hpp"

#include "operators.hpp"
#include "selection.hpp"
#include <algorithm>
#include <stdexcept>

using std::invalid_argument;
using std::move;
using std::size_t;
using std::sort;
using std::vector;

// Constructor: recibe el problema a resolver, la configuración del GA y una semilla
// para el generador de números aleatorios (reproducibilidad).
// Valida que los parámetros sean coherentes antes de ejecutar cualquier cosa.
GeneticAlgorithm::GeneticAlgorithm(const ProblemInstance &instance, GAConfig config, const int seed)
    : instance_(instance), config_(config), evaluator_(instance, config.penalties), rng_(seed) {

	// Sin ítems no hay problema que resolver
	if (instance_.items.empty()) {
		throw invalid_argument("La instancia debe tener al menos un item");
	}

	// Con 1 solo individuo no hay cruce posible
	if (config_.population_size <= 1) {
		throw invalid_argument("population_size debe ser mayor a 1");
	}

	if (config_.generations <= 0) {
		throw invalid_argument("generations debe ser mayor a 0");
	}

	// El torneo debe tener al menos 1 participante y no más que la población completa
	if (config_.tournament_size <= 0 || config_.tournament_size > config_.population_size) {
		throw invalid_argument("tournament_size invalido");
	}

	// El elitismo no puede vaciar la población (debe quedar espacio para hijos)
	if (config_.elitism_count < 0 || config_.elitism_count >= config_.population_size) {
		throw invalid_argument("elitism_count invalido");
	}

	// Tasas de probabilidad: deben estar en el rango [0, 1]
	if (config_.crossover_rate < 0.0 || config_.crossover_rate > 1.0) {
		throw invalid_argument("crossover_rate debe estar en [0,1]");
	}

	if (config_.mutation_rate < 0.0 || config_.mutation_rate > 1.0) {
		throw invalid_argument("mutation_rate debe estar en [0,1]");
	}
}

// Crea un individuo con genes aleatorios.
// Cada gen representa si el ítem i está incluido (1) o no (0) en la solución.
// Se usa distribución de Bernoulli con p=0.5: cada bit tiene igual probabilidad.
Chromosome GeneticAlgorithm::make_random_chromosome() {
	Chromosome chromosome;

	// Un gen por ítem del problema
	chromosome.genes.resize(instance_.items.size(), 0U);
	std::bernoulli_distribution bit(0.5);

	for (auto &gene : chromosome.genes) {
		gene = bit(rng_) ? 1U : 0U;
	}

	return chromosome;
}

// Genera la población inicial: un conjunto de individuos completamente aleatorios.
// Esta diversidad inicial es clave para que el GA explore bien el espacio de soluciones.
vector<Chromosome> GeneticAlgorithm::make_initial_population() {
	vector<Chromosome> population;
	population.reserve(static_cast<size_t>(config_.population_size));

	for (int i = 0; i < config_.population_size; ++i) {
		population.push_back(make_random_chromosome());
	}

	return population;
}

// Evalúa cada individuo de la población usando la función de fitness del evaluador.
// Devuelve pares (cromosoma, evaluación) para no perder el vínculo entre ambos.
vector<EvaluatedIndividual> GeneticAlgorithm::evaluate_population(const vector<Chromosome> &population) const {
	vector<EvaluatedIndividual> evaluated;
	evaluated.reserve(population.size());

	for (const auto &individual : population) {
		evaluated.push_back(EvaluatedIndividual{individual, evaluator_.evaluate(individual)});
	}

	return evaluated;
}

// Bucle principal del Algoritmo Genético.
// Cada iteración representa una generación: evaluar → seleccionar → cruzar → mutar.
// Retorna el mejor resultado encontrado (por fitness y el mejor factible).
GARunResult GeneticAlgorithm::run() {
	// --- INICIALIZACIÓN ---
	// Se genera la población de partida con individuos aleatorios
	vector<Chromosome> population = make_initial_population();

	GARunResult result{};
	result.has_feasible = false;
	int stagnation_counter = 0; // cuenta generaciones sin mejora del mejor fitness

	for (int generation = 0; generation < config_.generations; ++generation) {

		// --- EVALUACIÓN ---
		// Se calcula el fitness de cada individuo en la generación actual
		auto evaluated = evaluate_population(population);

		// Se ordenan de mayor a menor fitness para facilitar elitismo y búsqueda del mejor
		sort(evaluated.begin(), evaluated.end(), [](const EvaluatedIndividual &a, const EvaluatedIndividual &b) {
			return a.evaluation.fitness > b.evaluation.fitness;
		});

		// --- ACTUALIZACIÓN DEL MEJOR GLOBAL ---
		// Si el mejor de esta generación supera al histórico, se actualiza y se reinicia
		// el contador de estancamiento
		if (generation == 0 || evaluated.front().evaluation.fitness > result.best_by_fitness.evaluation.fitness) {
			result.best_by_fitness = evaluated.front();
			stagnation_counter = 0;
		} else {
			// No hubo mejora: se acumula una generación de estancamiento
			stagnation_counter++;
		}

		// --- MEJOR SOLUCIÓN FACTIBLE ---
		// Se busca el primer individuo (el mejor por fitness ya que están ordenados)
		// que cumpla todas las restricciones duras del problema
		for (const auto &individual : evaluated) {
			if (!individual.evaluation.feasible_hard) {
				continue;
			}
			if (!result.has_feasible || individual.evaluation.fitness > result.best_feasible.evaluation.fitness) {
				result.best_feasible = individual;
				result.has_feasible = true;
			}
			break; // Solo necesitamos el mejor factible de esta generación
		}

		// --- CRITERIO DE PARADA TEMPRANA (estancamiento) ---
		// Si el fitness no mejora durante max_stagnation_generations generaciones
		// consecutivas, se corta la ejecución para ahorrar cómputo
		if (stagnation_counter >= config_.max_stagnation_generations) {
			result.generations_executed = generation + 1;
			return result;
		}

		// --- CONSTRUCCIÓN DE LA NUEVA GENERACIÓN ---
		vector<Chromosome> next_population;
		next_population.reserve(population.size());

		// ELITISMO: los mejores N individuos pasan directamente a la siguiente generación
		// sin modificación. Esto garantiza que el mejor hallazgo nunca se pierda.
		for (int elite = 0; elite < config_.elitism_count; ++elite) {
			next_population.push_back(evaluated[static_cast<size_t>(elite)].chromosome);
		}

		// REPRODUCCIÓN: se rellena el resto de la población con hijos generados por
		// selección + cruce + mutación, hasta alcanzar el tamaño objetivo
		while (next_population.size() < population.size()) {

			// SELECCIÓN POR TORNEO: se elige el padre A y el padre B de forma independiente.
			// El torneo toma k individuos al azar y devuelve el de mayor fitness entre ellos.
			// Un torneo mayor = más presión selectiva (ganan casi siempre los mejores).
			const size_t idx_a = tournament_select_index(evaluated, config_.tournament_size, rng_);
			const size_t idx_b = tournament_select_index(evaluated, config_.tournament_size, rng_);

			// CRUCE ONE-POINT: con probabilidad crossover_rate se elige un punto de corte
			// aleatorio y se intercambian los segmentos finales de ambos padres,
			// produciendo dos hijos que combinan material genético de ambos.
			auto [child_a, child_b] = crossover_one_point(evaluated[idx_a].chromosome, evaluated[idx_b].chromosome,
			                                              config_.crossover_rate, rng_);

			// MUTACIÓN BIT-FLIP: cada gen de los hijos tiene probabilidad mutation_rate
			// de invertirse (0→1 o 1→0). Introduce diversidad y evita convergencia prematura.
			mutate_bit_flip(child_a, config_.mutation_rate, rng_);
			mutate_bit_flip(child_b, config_.mutation_rate, rng_);

			next_population.push_back(move(child_a));
			// Solo se añade child_b si aún hay espacio (la población puede ser impar)
			if (next_population.size() < population.size()) {
				next_population.push_back(move(child_b));
			}
		}

		// La nueva generación reemplaza completamente a la anterior (reemplazo generacional)
		population = move(next_population);
		result.generations_executed = generation + 1;
	}

	return result;
}
