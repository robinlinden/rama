// SPDX-FileCopyrightText: 2026 Robin Lindén <dev@robinlinden.eu>
//
// SPDX-License-Identifier: BSD-2-Clause

#include "gguf/gguf.h"

#include <etest/etest2.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <print>

int main(int argc, char **argv) {
    if (argc != 2) {
        char const *program_name = argv[0] != nullptr ? argv[0] : "<bin>";
        std::println(stderr, "Usage: {} <GGUF path>", program_name);
        return 1;
    }

    char const *gguf_path = argv[1];
    std::ifstream is{gguf_path, std::ios::binary};
    if (!is) {
        std::println(stderr, "Failed to open file: {}", gguf_path);
        return 1;
    }

    etest::Suite s;

    s.add_benchmark("read_gguf_metadata", [&] {
        auto metadata = gguf::read_gguf_metadata(is);
        if (!metadata) {
            std::println(stderr, "Failed to parse metadata");
            std::exit(1);
        }

        is.clear();
        is.seekg(0, std::ios::beg);
    });

    return s.run();
}
