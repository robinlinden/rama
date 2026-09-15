// SPDX-FileCopyrightText: 2026 Robin Lindén <dev@robinlinden.eu>
//
// SPDX-License-Identifier: BSD-2-Clause

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
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
auto read<float>(std::istream &stream) -> std::optional<float> {
    auto value = read<std::uint32_t>(stream);
    if (!value) {
        return std::nullopt;
    }

    float result;
    std::memcpy(&result, &*value, sizeof(result));
    return result;
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

enum class GgufType : std::uint8_t {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
};

auto to_string(GgufType type) -> std::string_view {
    switch (type) {
    case GgufType::Uint8:
        return "uint8";
    case GgufType::Int8:
        return "int8";
    case GgufType::Uint16:
        return "uint16";
    case GgufType::Int16:
        return "int16";
    case GgufType::Uint32:
        return "uint32";
    case GgufType::Int32:
        return "int32";
    case GgufType::Float32:
        return "float32";
    case GgufType::Bool:
        return "bool";
    case GgufType::String:
        return "string";
    case GgufType::Array:
        return "array";
    case GgufType::Uint64:
        return "uint64";
    case GgufType::Int64:
        return "int64";
    case GgufType::Float64:
        return "float64";
    }

    return "<unknown>";
}

template<>
auto read<GgufType>(std::istream &stream) -> std::optional<GgufType> {
    auto value = read<std::int32_t>(stream);
    if (!value || *value < static_cast<std::int32_t>(GgufType::Uint8) ||
        *value > static_cast<std::int32_t>(GgufType::Float64)) {
        return std::nullopt;
    }

    return static_cast<GgufType>(*value);
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

    for (std::uint64_t i = 0; i < *metadata_kv_count; ++i) {
        auto key = read<std::string>(file);
        if (!key) {
            std::println(stderr, "Failed to read metadata key");
            return 1;
        }

        auto type = read<GgufType>(file);
        if (!type) {
            std::println(stderr, "Failed to read metadata type");
            return 1;
        }

        if (*type == GgufType::Uint32) {
            auto value = read<std::uint32_t>(file);
            if (!value) {
                std::println(stderr, "Failed to read metadata value for key '{}'", *key);
                return 1;
            }

            std::println("* {}: {}", *key, *value);
            continue;
        }

        if (*type == GgufType::Int32) {
            auto value = read<std::int32_t>(file);
            if (!value) {
                std::println(stderr, "Failed to read metadata value for key '{}'", *key);
                return 1;
            }

            std::println("* {}: {}", *key, *value);
            continue;
        }

        if (*type == GgufType::Float32) {
            auto value = read<float>(file);
            if (!value) {
                std::println(stderr, "Failed to read metadata value for key '{}'", *key);
                return 1;
            }

            std::println("* {}: {}", *key, *value);
            continue;
        }

        if (*type == GgufType::String) {
            auto value = read<std::string>(file);
            if (!value) {
                std::println(stderr, "Failed to read metadata value for key '{}'", *key);
                return 1;
            }

            std::println("* {}: {}", *key, *value);
            continue;
        }

        if (*type == GgufType::Array) {
            auto array_type = read<GgufType>(file);
            if (!array_type) {
                std::println(stderr, "Failed to read metadata array type for key '{}'", *key);
                return 1;
            }

            auto array_length = read<std::uint64_t>(file);
            if (!array_length) {
                std::println(stderr, "Failed to read metadata array length for key '{}'", *key);
                return 1;
            }

            std::println(
                "* {}: array of {} elements of type {}",
                *key,
                *array_length,
                to_string(*array_type));

            // Fallthrough to failure until future Robin deals with this.
        }

        std::println(stderr, "Unsupported metadata key '{}' w/ type '{}'", *key, to_string(*type));
        return 1;
    }
}
