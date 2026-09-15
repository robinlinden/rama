// SPDX-FileCopyrightText: 2026 Robin Lindén <dev@robinlinden.eu>
//
// SPDX-License-Identifier: BSD-2-Clause

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <format>
#include <fstream>
#include <optional>
#include <print>
#include <string>
#include <utility>
#include <variant>
#include <vector>

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

struct GgufValue;
struct GgufValue {
    std::variant<std::uint32_t, std::int32_t, float, std::string, std::vector<GgufValue>> v;
};

auto to_string(GgufValue const &value) -> std::string {
    struct GgufStringifier {
        std::string operator()(std::uint32_t v) const { return std::to_string(v); }
        std::string operator()(std::int32_t v) const { return std::to_string(v); }
        std::string operator()(float v) const { return std::to_string(v); }
        std::string operator()(std::string const &v) const { return v; }
        std::string operator()(std::vector<GgufValue> const &v) const {
            return "[" +
                   std::ranges::fold_left(
                       v,
                       std::string{},
                       [](std::string acc, GgufValue const &value) {
                           return acc.empty() ? to_string(value)
                                              : std::move(acc) + ", " + to_string(value);
                       }) +
                   "]";
        }
    };

    return std::visit(GgufStringifier{}, value.v);
}

auto read_gguf_value(std::istream &stream, GgufType type) -> std::optional<GgufValue> {
    switch (type) {
    case GgufType::Uint32: {
        auto value = read<std::uint32_t>(stream);
        if (!value) {
            return std::nullopt;
        }

        return GgufValue{*value};
    }
    case GgufType::Int32: {
        auto value = read<std::int32_t>(stream);
        if (!value) {
            return std::nullopt;
        }

        return GgufValue{*value};
    }
    case GgufType::Float32: {
        auto value = read<float>(stream);
        if (!value) {
            return std::nullopt;
        }

        return GgufValue{*value};
    }
    case GgufType::Bool: {
        auto value = read<std::int8_t>(stream);
        if (!value) {
            return std::nullopt;
        }

        return GgufValue{static_cast<std::uint32_t>(*value != 0)};
    }
    case GgufType::String: {
        auto value = read<std::string>(stream);
        if (!value) {
            return std::nullopt;
        }

        return GgufValue{*value};
    }
    case GgufType::Array: {
        auto array_type = read<GgufType>(stream);
        if (!array_type) {
            return std::nullopt;
        }

        auto array_length = read<std::uint64_t>(stream);
        if (!array_length) {
            return std::nullopt;
        }

        std::vector<GgufValue> values;
        values.reserve(*array_length);

        for (std::uint64_t i = 0; i < *array_length; ++i) {
            auto value = read_gguf_value(stream, *array_type);
            if (!value) {
                return std::nullopt;
            }

            values.push_back(*value);
        }

        return GgufValue{std::move(values)};
    }
    default:
        std::println(stderr, "Unsupported GGUF type: {}", to_string(type));
        return std::nullopt;
    }
}

} // namespace

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

        auto value = read_gguf_value(file, *type);
        if (!value) {
            std::println(
                stderr, "Unsupported metadata key '{}' w/ type '{}'", *key, to_string(*type));
            return 1;
        }

        // Only print the first 128 characters of the value to avoid flooding the terminal.
        std::println("* {}: {}", *key, to_string(*value).substr(0, 128));
    }
}
