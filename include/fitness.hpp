#pragma once

#include "models.hpp"

class FitnessEvaluator {
public:
	FitnessEvaluator(const ProblemInstance &instance, PenaltyConfig penalties);

	FitnessBreakdown evaluate(const Chromosome &chromosome) const;

private:
	const ProblemInstance &instance_;
	PenaltyConfig penalties_;
};
