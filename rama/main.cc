// SPDX-FileCopyrightText: 2026 Robin Lindén <dev@robinlinden.eu>
//
// SPDX-License-Identifier: BSD-2-Clause

#include "gguf/gguf.h"

#include <cstdint>
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

    auto magic = gguf::read<std::uint32_t>(file);
    if (!magic) {
        std::println(stderr, "Failed to read magic number");
        return 1;
    }

    if (magic != 0x46554747) {
        std::println(stderr, "Invalid magic number: {:08x}", *magic);
        return 1;
    }

    auto version = gguf::read<std::uint32_t>(file);
    if (!version) {
        std::println(stderr, "Failed to read version");
        return 1;
    }

    std::println("GGUF version: {}", *version);

    if (*version != 3) {
        std::println(stderr, "Unsupported GGUF version: {}", *version);
        return 1;
    }

    auto tensor_count = gguf::read<std::uint64_t>(file);
    if (!tensor_count) {
        std::println(stderr, "Failed to read tensor count");
        return 1;
    }

    std::println("Tensor count: {}", *tensor_count);

    auto metadata_kv_count = gguf::read<std::uint64_t>(file);
    if (!metadata_kv_count) {
        std::println(stderr, "Failed to read metadata key-value count");
        return 1;
    }

    std::println("Metadata count: {}", *metadata_kv_count);

    for (std::uint64_t i = 0; i < *metadata_kv_count; ++i) {
        auto key = gguf::read<std::string>(file);
        if (!key) {
            std::println(stderr, "Failed to read metadata key");
            return 1;
        }

        auto type = gguf::read<gguf::GgufType>(file);
        if (!type) {
            std::println(stderr, "Failed to read metadata type");
            return 1;
        }

        auto value = gguf::read_gguf_value(file, *type);
        if (!value) {
            std::println(
                stderr, "Unsupported metadata key '{}' w/ type '{}'", *key, to_string(*type));
            return 1;
        }

        // Only print the first 128 characters of the value to avoid flooding the terminal.
        std::println("* {}: {}", *key, to_string(*value).substr(0, 128));
    }

    std::println();

    for (std::uint64_t i = 0; i < *tensor_count; ++i) {
        auto tensor_name = gguf::read<std::string>(file);
        if (!tensor_name) {
            std::println(stderr, "Failed to read tensor name");
            return 1;
        }

        auto dimension_count = gguf::read<std::uint32_t>(file);
        if (!dimension_count) {
            std::println(stderr, "Failed to read dimension count");
            return 1;
        }

        for (std::uint32_t j = 0; j < *dimension_count; ++j) {
            auto dim = gguf::read<std::uint64_t>(file);
            if (!dim) {
                std::println(stderr, "Failed to read tensor dimension");
                return 1;
            }
        }

        auto tensor_type = gguf::read<gguf::GgmlType>(file);
        if (!tensor_type) {
            std::println(stderr, "Failed to read tensor type");
            return 1;
        }

        auto tensor_offset = gguf::read<std::uint64_t>(file);
        if (!tensor_offset) {
            std::println(stderr, "Failed to read tensor offset");
            return 1;
        }

        std::println(
            "+ {}: type={}, offset={}", *tensor_name, to_string(*tensor_type), *tensor_offset);
    }
}
