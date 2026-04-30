#pragma once

#include "csv.h"
#include <array>
#include <cstddef>
#include <expected>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

template <typename T> struct CsvSchema;

using CsvError = std::string;

namespace csv_reader_detail {
template <std::size_t N>
std::expected<std::array<std::string, N>, CsvError>
split_header_line(const std::string &header_line) {
	std::array<std::string, N> parts{};
	std::stringstream ss(header_line);
	std::string token;
	std::size_t idx = 0;

	while (std::getline(ss, token, ',')) {
		if (idx >= N) {
			return std::unexpected("CSV tiene columnas extra en el header");
		}
		parts[idx++] = token;
	}

	if (idx != N) {
		return std::unexpected(
		    "CSV no coincide con la cantidad de columnas esperada");
	}

	return parts;
}

template <std::size_t N>
std::expected<void, CsvError>
validate_header_strict(const std::string &path,
                       const std::array<const char *, N> &expected_headers) {
	std::ifstream input(path);
	if (!input.is_open()) {
		return std::unexpected("No se pudo abrir el archivo CSV: " + path);
	}

	std::string header_line;
	if (!std::getline(input, header_line)) {
		return std::unexpected("CSV vacio o sin header: " + path);
	}

	auto actual_headers_result = split_header_line<N>(header_line);
	if (!actual_headers_result) {
		return std::unexpected(actual_headers_result.error());
	}

	auto actual_headers = std::move(actual_headers_result.value());
	for (std::size_t i = 0; i < N; ++i) {
		if (actual_headers[i] != expected_headers[i]) {
			return std::unexpected("Header CSV invalido en columna " +
			                       std::to_string(i) + ": esperado '" +
			                       expected_headers[i] + "' pero se obtuvo '" +
			                       actual_headers[i] + "'");
		}
	}

	return {};
}

template <unsigned int N, std::size_t... I>
void read_header(
    io::CSVReader<N> &reader,
    const std::array<const char *, static_cast<std::size_t>(N)> &headers,
    std::index_sequence<I...>) {
	reader.read_header(io::ignore_no_column, headers[I]...);
}
} // namespace csv_reader_detail

template <typename T>
std::expected<std::vector<T>, CsvError> read_csv(const std::string &path) {
	constexpr std::size_t kColumns = CsvSchema<T>::column_count;
	const auto headers = CsvSchema<T>::headers();

	auto header_validation =
	    csv_reader_detail::validate_header_strict(path, headers);
	if (!header_validation) {
		return std::unexpected(header_validation.error());
	}

	io::CSVReader<kColumns> reader(path);
	csv_reader_detail::read_header(reader, headers,
	                               std::make_index_sequence<kColumns>{});

	std::vector<T> rows;
	T row;
	while (CsvSchema<T>::read_row(reader, row)) {
		rows.push_back(std::move(row));
	}

	return rows;
}
