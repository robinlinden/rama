// SPDX-FileCopyrightText: 2026 Robin Lindén <dev@robinlinden.eu>
//
// SPDX-License-Identifier: BSD-2-Clause

#include <bit>
#include <cstdint>
#include <cstdio>
#include <format>
#include <fstream>
#include <optional>
#include <print>

namespace {

template<typename T>
auto read(std::istream &stream) -> std::optional<T> {
    T value;
    if (!stream.read(reinterpret_cast<char *>(&value), sizeof(value))) {
        return std::nullopt;
    }

    // GGUF makes no promises about endianness, but is usually little-endian.
    if (std::endian::native != std::endian::little) {
        value = std::byteswap(value);
    }

    return value;
}

template<>
auto read<std::string>(std::istream &stream) -> std::optional<std::string> {
    auto length = read<std::uint64_t>(stream);
    if (!length) {
        return std::nullopt;
    }

    std::string value;
    value.resize(*length);
    if (!stream.read(value.data(), *length)) {
        return std::nullopt;
    }

    return value;
}

} // namespace

auto main(int argc, char **argv) -> int {
    if (argc < 2) {
        char const *program_name = argv[0] != nullptr ? argv[0] : "<bin>";
        std::println(stderr, "Usage: {} <GGUF path>", program_name);
        return 1;
    }

    char const *gguf_path = argv[1];
    std::ifstream file{gguf_path};
    if (!file) {
        std::println(stderr, "Failed to open file: {}", gguf_path);
        return 1;
    }

    auto magic = read<std::uint32_t>(file);
    if (!magic) {
        std::println(stderr, "Failed to read magic number");
        return 1;
    }

    if (magic != 0x46554747) {
        std::println(stderr, "Invalid magic number: {:08x}", *magic);
        return 1;
    }

    auto version = read<std::uint32_t>(file);
    if (!version) {
        std::println(stderr, "Failed to read version");
        return 1;
    }

    std::println("GGUF version: {}", *version);

    if (*version != 3) {
        std::println(stderr, "Unsupported GGUF version: {}", *version);
        return 1;
    }

    auto tensor_count = read<std::uint64_t>(file);
    if (!tensor_count) {
        std::println(stderr, "Failed to read tensor count");
        return 1;
    }

    std::println("Tensor count: {}", *tensor_count);

    auto metadata_kv_count = read<std::uint64_t>(file);
    if (!metadata_kv_count) {
        std::println(stderr, "Failed to read metadata key-value count");
        return 1;
    }

    std::println("Metadata count: {}", *metadata_kv_count);
}
