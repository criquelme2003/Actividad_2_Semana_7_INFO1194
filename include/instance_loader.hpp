#pragma once

#include "models.hpp"
#include <expected>
#include <string>

std::expected<ProblemInstance, std::string>
load_problem_instance(const std::string &instance_path, double capacity_ratio);
