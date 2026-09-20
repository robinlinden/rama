// SPDX-FileCopyrightText: 2026 Robin Lindén <dev@robinlinden.eu>
//
// SPDX-License-Identifier: BSD-2-Clause

#ifndef RAMA_GGUF_GGUF_H_
#define RAMA_GGUF_GGUF_H_

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <istream>
#include <optional>
#include <print>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace gguf {

// https://github.com/ggml-org/ggml/blob/456172ec733a135778adcd32d00e576a58232e45/docs/gguf.md
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
inline auto read<float>(std::istream &stream) -> std::optional<float> {
    auto value = read<std::uint32_t>(stream);
    if (!value) {
        return std::nullopt;
    }

    float result;
    std::memcpy(&result, &*value, sizeof(result));
    return result;
}

template<>
inline auto read<std::string>(std::istream &stream) -> std::optional<std::string> {
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

constexpr auto to_string(GgufType type) -> std::string_view {
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
inline auto read<GgufType>(std::istream &stream) -> std::optional<GgufType> {
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

constexpr auto to_string(GgufValue const &value) -> std::string {
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

inline auto read_gguf_value(std::istream &stream, GgufType type) -> std::optional<GgufValue> {
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

struct GgufMetadataKV {
    std::string key;
    GgufType valueType;
    GgufValue value;
};

constexpr auto read_gguf_metadata_kv(std::istream &stream) -> std::optional<GgufMetadataKV> {
    auto key = gguf::read<std::string>(stream);
    if (!key) {
        std::println(stderr, "Failed to read metadata key");
        return std::nullopt;
    }

    auto type = gguf::read<gguf::GgufType>(stream);
    if (!type) {
        std::println(stderr, "Failed to read metadata type");
        return std::nullopt;
    }

    auto value = gguf::read_gguf_value(stream, *type);
    if (!value) {
        std::println(stderr, "Unsupported metadata key '{}' w/ type '{}'", *key, to_string(*type));
        return std::nullopt;
    }

    return GgufMetadataKV{*key, *type, *value};
}

enum class GgmlType : std::uint8_t {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    IQ2_XXS,
    IQ2_XS,
    IQ3_XXS,
    IQ1_S,
    IQ4_NL,
    IQ3_S,
    IQ2_S,
    IQ4_XS,
    I8,
    I16,
    I32,
    I64,
    F64,
    IQ1_M,
    BF16,
    TQ1_0,
    TQ2_0,
    MXFP4,
};

template<>
inline auto read<GgmlType>(std::istream &stream) -> std::optional<GgmlType> {
    auto value = read<std::uint32_t>(stream);
    if (!value) {
        return std::nullopt;
    }

    switch (*value) {
    case 0:
        return GgmlType::F32;
    case 1:
        return GgmlType::F16;
    case 2:
        return GgmlType::Q4_0;
    case 3:
        return GgmlType::Q4_1;
    case 4:
        std::println(stderr, "GGML type Q4_2 has been dropped from GGUF and is unsupported");
        return std::nullopt;
    case 5:
        std::println(stderr, "GGML type Q4_3 has been dropped from GGUF and is unsupported");
        return std::nullopt;
    case 6:
        return GgmlType::Q5_0;
    case 7:
        return GgmlType::Q5_1;
    case 8:
        return GgmlType::Q8_0;
    case 9:
        return GgmlType::Q8_1;
    case 10:
        return GgmlType::Q2_K;
    case 11:
        return GgmlType::Q3_K;
    case 12:
        return GgmlType::Q4_K;
    case 13:
        return GgmlType::Q5_K;
    case 14:
        return GgmlType::Q6_K;
    case 15:
        return GgmlType::Q8_K;
    case 16:
        return GgmlType::IQ2_XXS;
    case 17:
        return GgmlType::IQ2_XS;
    case 18:
        return GgmlType::IQ3_XXS;
    case 19:
        return GgmlType::IQ1_S;
    case 20:
        return GgmlType::IQ4_NL;
    case 21:
        return GgmlType::IQ3_S;
    case 22:
        return GgmlType::IQ2_S;
    case 23:
        return GgmlType::IQ4_XS;
    case 24:
        return GgmlType::I8;
    case 25:
        return GgmlType::I16;
    case 26:
        return GgmlType::I32;
    case 27:
        return GgmlType::I64;
    case 28:
        return GgmlType::F64;
    case 29:
        return GgmlType::IQ1_M;
    case 30:
        return GgmlType::BF16;
    case 31:
        std::println(stderr, "GGML type Q4_0_4_4 has been dropped from GGUF and is unsupported");
        return std::nullopt;
    case 32:
        std::println(stderr, "GGML type Q4_0_4_8 has been dropped from GGUF and is unsupported");
        return std::nullopt;
    case 33:
        std::println(stderr, "GGML type Q4_0_8_8 has been dropped from GGUF and is unsupported");
        return std::nullopt;
    case 34:
        return GgmlType::TQ1_0;
    case 35:
        return GgmlType::TQ2_0;
    case 36:
        std::println(stderr, "GGML type IQ4_NL_4_4 has been dropped from GGUF and is unsupported");
        return std::nullopt;
    case 37:
        std::println(stderr, "GGML type IQ4_NL_4_8 has been dropped from GGUF and is unsupported");
        return std::nullopt;
    case 38:
        std::println(stderr, "GGML type IQ4_NL_8_8 has been dropped from GGUF and is unsupported");
        return std::nullopt;
    case 39:
        return GgmlType::MXFP4;
    default:
        std::println(stderr, "Unsupported GGML type: {}", *value);
        return std::nullopt;
    }
}

constexpr auto to_string(GgmlType type) -> std::string_view {
    switch (type) {
    case GgmlType::F32:
        return "f32";
    case GgmlType::F16:
        return "f16";
    case GgmlType::Q4_0:
        return "q4_0";
    case GgmlType::Q4_1:
        return "q4_1";
    case GgmlType::Q5_0:
        return "q5_0";
    case GgmlType::Q5_1:
        return "q5_1";
    case GgmlType::Q8_0:
        return "q8_0";
    case GgmlType::Q8_1:
        return "q8_1";
    case GgmlType::Q2_K:
        return "q2_k";
    case GgmlType::Q3_K:
        return "q3_k";
    case GgmlType::Q4_K:
        return "q4_k";
    case GgmlType::Q5_K:
        return "q5_k";
    case GgmlType::Q6_K:
        return "q6_k";
    case GgmlType::Q8_K:
        return "q8_k";
    case GgmlType::IQ2_XXS:
        return "iq2_xxs";
    case GgmlType::IQ2_XS:
        return "iq2_xs";
    case GgmlType::IQ3_XXS:
        return "iq3_xxs";
    case GgmlType::IQ1_S:
        return "iq1_s";
    case GgmlType::IQ4_NL:
        return "iq4_nl";
    case GgmlType::IQ3_S:
        return "iq3_s";
    case GgmlType::IQ2_S:
        return "iq2_s";
    case GgmlType::IQ4_XS:
        return "iq4_xs";
    case GgmlType::I8:
        return "i8";
    case GgmlType::I16:
        return "i16";
    case GgmlType::I32:
        return "i32";
    case GgmlType::I64:
        return "i64";
    case GgmlType::F64:
        return "f64";
    case GgmlType::IQ1_M:
        return "iq1_m";
    case GgmlType::BF16:
        return "bf16";
    case GgmlType::TQ1_0:
        return "tq1_0";
    case GgmlType::TQ2_0:
        return "tq2_0";
    case GgmlType::MXFP4:
        return "mxfp4";
    }

    return "<unknown>";
}

struct GgufTensorInfo {
    std::string name;
    std::vector<std::uint64_t> dimensions;
    GgmlType type;
    std::uint64_t offset;
};

constexpr auto read_gguf_tensor_info(std::istream &stream) -> std::optional<GgufTensorInfo> {
    auto tensor_name = gguf::read<std::string>(stream);
    if (!tensor_name) {
        std::println(stderr, "Failed to read tensor name");
        return std::nullopt;
    }

    auto dimension_count = gguf::read<std::uint32_t>(stream);
    if (!dimension_count) {
        std::println(stderr, "Failed to read dimension count");
        return std::nullopt;
    }

    std::vector<std::uint64_t> dimensions;
    dimensions.reserve(*dimension_count);

    for (std::uint32_t j = 0; j < *dimension_count; ++j) {
        auto dim = gguf::read<std::uint64_t>(stream);
        if (!dim) {
            std::println(stderr, "Failed to read tensor dimension");
            return std::nullopt;
        }
        dimensions.push_back(*dim);
    }

    auto tensor_type = gguf::read<gguf::GgmlType>(stream);
    if (!tensor_type) {
        std::println(stderr, "Failed to read tensor type");
        return std::nullopt;
    }

    auto tensor_offset = gguf::read<std::uint64_t>(stream);
    if (!tensor_offset) {
        std::println(stderr, "Failed to read tensor offset");
        return std::nullopt;
    }

    return gguf::GgufTensorInfo{
        std::move(*tensor_name),
        std::move(dimensions),
        *tensor_type,
        *tensor_offset,
    };
}

struct GgufMetadata {
    std::uint32_t magic;
    std::uint32_t version;
    std::vector<GgufMetadataKV> metadata_kv;
    std::vector<GgufTensorInfo> tensor_infos;
};

constexpr auto read_gguf_metadata(std::istream &stream) -> std::optional<GgufMetadata> {
    auto magic = gguf::read<std::uint32_t>(stream);
    if (!magic) {
        std::println(stderr, "Failed to read magic number");
        return std::nullopt;
    }

    if (magic != 0x46554747) {
        std::println(stderr, "Invalid magic number: {:08x}", *magic);
        return std::nullopt;
    }

    auto version = gguf::read<std::uint32_t>(stream);
    if (!version) {
        std::println(stderr, "Failed to read version");
        return std::nullopt;
    }

    if (*version != 3) {
        std::println(stderr, "Unsupported GGUF version: {}", *version);
        return std::nullopt;
    }

    auto tensor_count = gguf::read<std::uint64_t>(stream);
    if (!tensor_count) {
        std::println(stderr, "Failed to read tensor count");
        return std::nullopt;
    }

    auto metadata_kv_count = gguf::read<std::uint64_t>(stream);
    if (!metadata_kv_count) {
        std::println(stderr, "Failed to read metadata key-value count");
        return std::nullopt;
    }

    std::vector<gguf::GgufMetadataKV> metadata_kvs;
    metadata_kvs.reserve(*metadata_kv_count);

    for (std::uint64_t i = 0; i < *metadata_kv_count; ++i) {
        auto metadata_kv = gguf::read_gguf_metadata_kv(stream);
        if (!metadata_kv) {
            std::println(stderr, "Failed to read metadata key-value");
            return std::nullopt;
        }

        metadata_kvs.push_back(std::move(*metadata_kv));
    }

    std::vector<gguf::GgufTensorInfo> tensor_infos;
    tensor_infos.reserve(*tensor_count);

    for (std::uint64_t i = 0; i < *tensor_count; ++i) {
        auto tensor_info = gguf::read_gguf_tensor_info(stream);
        if (!tensor_info) {
            std::println(stderr, "Failed to read tensor info");
            return std::nullopt;
        }

        tensor_infos.push_back(std::move(*tensor_info));
    }

    return GgufMetadata{*magic, *version, std::move(metadata_kvs), std::move(tensor_infos)};
}

} // namespace gguf

#endif
