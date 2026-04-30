#pragma once

#include "csv_reader.hpp"
#include <array>
#include <string>

struct Record {
	std::string campo1;
	double campo2;
	int campo3;
};

template <> struct CsvSchema<Record> {
	static constexpr unsigned int column_count = 3;

	static constexpr std::array<const char *,
	                            static_cast<std::size_t>(column_count)>
	headers() {
		return {"col1", "col2", "col3"};
	}

	static bool read_row(io::CSVReader<column_count> &reader, Record &row) {
		return reader.read_row(row.campo1, row.campo2, row.campo3);
	}
};
