#include "instance_loader.hpp"

#include "csv_reader.hpp"
#include <filesystem>
#include <fmt/core.h>
#include <numeric>

using std::string;
using std::to_string;

namespace {
struct ItemRow {
	int id;
	double value;
	double weight;
	double volume;
	string category;
};

struct CategoryRuleRow {
	string category;
	int minimum;
	int maximum;
};

struct IncompatibilityRow {
	int id_item_a;
	int id_item_b;
};

struct DependencyRow {
	int id_item;
	int id_required;
};
} // namespace

template <> struct CsvSchema<ItemRow> {
	static constexpr unsigned int column_count = 5;

	static constexpr std::array<const char *, column_count> headers() {
		return {"id", "valor", "peso", "volumen", "categoria"};
	}

	static bool read_row(io::CSVReader<column_count> &reader, ItemRow &row) {
		return reader.read_row(row.id, row.value, row.weight, row.volume,
		                       row.category);
	}
};

template <> struct CsvSchema<CategoryRuleRow> {
	static constexpr unsigned int column_count = 3;

	static constexpr std::array<const char *, column_count> headers() {
		return {"categoria", "minimo", "maximo"};
	}

	static bool read_row(io::CSVReader<column_count> &reader,
	                     CategoryRuleRow &row) {
		return reader.read_row(row.category, row.minimum, row.maximum);
	}
};

template <> struct CsvSchema<IncompatibilityRow> {
	static constexpr unsigned int column_count = 2;

	static constexpr std::array<const char *, column_count> headers() {
		return {"id_item_a", "id_item_b"};
	}

	static bool read_row(io::CSVReader<column_count> &reader,
	                     IncompatibilityRow &row) {
		return reader.read_row(row.id_item_a, row.id_item_b);
	}
};

template <> struct CsvSchema<DependencyRow> {
	static constexpr unsigned int column_count = 2;

	static constexpr std::array<const char *, column_count> headers() {
		return {"id_item", "id_requerido"};
	}

	static bool read_row(io::CSVReader<column_count> &reader,
	                     DependencyRow &row) {
		return reader.read_row(row.id_item, row.id_required);
	}
};

std::expected<ProblemInstance, std::string>
load_problem_instance(const std::string &instance_path, double capacity_ratio) {
	namespace fs = std::filesystem;

	const fs::path base_path(instance_path);
	if (!fs::exists(base_path) || !fs::is_directory(base_path)) {
		return std::unexpected("La instancia debe ser un directorio: " +
		                       instance_path);
	}

	const fs::path items_path = base_path / "items.csv";
	auto items_result = read_csv<ItemRow>(items_path.string());
	if (!items_result) {
		return std::unexpected(items_result.error());
	}

	ProblemInstance instance;
	for (const auto &item_row : items_result.value()) {
		if (instance.id_to_index.contains(item_row.id)) {
			return std::unexpected("items.csv contiene IDs duplicados: " +
			                       to_string(item_row.id));
		}
		instance.id_to_index[item_row.id] = instance.items.size();
		instance.items.push_back(Item{item_row.id, item_row.value,
		                              item_row.weight, item_row.volume,
		                              item_row.category});
	}

	if (instance.items.empty()) {
		return std::unexpected("items.csv no contiene items");
	}

	const fs::path category_rules_path = base_path / "category_rules.csv";
	if (fs::exists(category_rules_path)) {
		auto category_result =
		    read_csv<CategoryRuleRow>(category_rules_path.string());
		if (!category_result) {
			return std::unexpected(category_result.error());
		}
		for (const auto &rule_row : category_result.value()) {
			instance.category_rules.push_back(CategoryRule{
			    rule_row.category, rule_row.minimum, rule_row.maximum});
		}
	}

	const fs::path incompat_path = base_path / "incompatibilities.csv";
	if (fs::exists(incompat_path)) {
		auto incompat_result =
		    read_csv<IncompatibilityRow>(incompat_path.string());
		if (!incompat_result) {
			return std::unexpected(incompat_result.error());
		}
		for (const auto &pair_row : incompat_result.value()) {
			instance.incompatibilities.push_back(
			    {pair_row.id_item_a, pair_row.id_item_b});
			auto it_a = instance.id_to_index.find(pair_row.id_item_a);
			auto it_b = instance.id_to_index.find(pair_row.id_item_b);
			if (it_a == instance.id_to_index.end() ||
			    it_b == instance.id_to_index.end()) {
				fmt::print(
				    stderr,
				    "[WARN] incompatibility con ID inexistente: ({}, {})\n",
				    pair_row.id_item_a, pair_row.id_item_b);
				continue;
			}
			instance.incompatibility_indices.push_back(
			    {it_a->second, it_b->second});
		}
	}

	const fs::path deps_path = base_path / "dependencies.csv";
	if (fs::exists(deps_path)) {
		auto deps_result = read_csv<DependencyRow>(deps_path.string());
		if (!deps_result) {
			return std::unexpected(deps_result.error());
		}
		for (const auto &dep_row : deps_result.value()) {
			instance.dependencies.push_back(
			    {dep_row.id_item, dep_row.id_required});
			auto it_item = instance.id_to_index.find(dep_row.id_item);
			auto it_required = instance.id_to_index.find(dep_row.id_required);
			if (it_item == instance.id_to_index.end() ||
			    it_required == instance.id_to_index.end()) {
				fmt::print(stderr,
				           "[WARN] dependency con ID inexistente: ({}, {})\n",
				           dep_row.id_item, dep_row.id_required);
				continue;
			}
			instance.dependency_indices.push_back(
			    {it_item->second, it_required->second});
		}
	}

	const double total_weight = std::accumulate(
	    instance.items.begin(), instance.items.end(), 0.0,
	    [](double sum, const Item &item) { return sum + item.weight; });
	const double total_volume = std::accumulate(
	    instance.items.begin(), instance.items.end(), 0.0,
	    [](double sum, const Item &item) { return sum + item.volume; });

	instance.max_weight = total_weight * capacity_ratio;
	instance.max_volume = total_volume * capacity_ratio;
	instance.total_weight = total_weight;
	instance.total_volume = total_volume;
	instance.max_possible_value = std::accumulate(
	    instance.items.begin(), instance.items.end(), 0.0,
	    [](double sum, const Item &item) { return sum + item.value; });

	return instance;
}
