// SPDX-FileCopyrightText: 2026 Robin Lindén <dev@robinlinden.eu>
//
// SPDX-License-Identifier: BSD-2-Clause

#include "gguf/gguf.h"

#include <cstdio>
#include <fstream>
#include <print>
#include <string>

auto main(int argc, char **argv) -> int {
    if (argc < 2) {
        char const *program_name = argv[0] != nullptr ? argv[0] : "<bin>";
        std::println(stderr, "Usage: {} <GGUF path>", program_name);
        return 1;
    }

    char const *gguf_path = argv[1];
    std::ifstream file{gguf_path, std::ios::binary};
    if (!file) {
        std::println(stderr, "Failed to open file: {}", gguf_path);
        return 1;
    }

    auto metadata = gguf::read_gguf_metadata(file);
    if (!metadata) {
        std::println(stderr, "Failed to parse metadata");
        return 1;
    }

    std::println("GGUF version: {}", metadata->version);

    std::println("Tensor count: {}", metadata->tensor_infos.size());

    std::println("Metadata count: {}", metadata->metadata_kv.size());

    for (auto const &metadata_kv : metadata->metadata_kv) {
        // Only print the first 128 characters of the value to avoid flooding the terminal.
        std::println("* {}: {}", metadata_kv.key, to_string(metadata_kv.value).substr(0, 128));
    }

    std::println();

    for (auto const &tensor_info : metadata->tensor_infos) {
        std::println(
            "+ {}: type={}, offset={}",
            tensor_info.name,
            to_string(tensor_info.type),
            tensor_info.offset);
    }
}
