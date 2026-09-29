// SPDX-FileCopyrightText: 2026 Robin Lindén <dev@robinlinden.eu>
//
// SPDX-License-Identifier: BSD-2-Clause

#include "gguf/gguf.h"

#include <etest/etest2.h>

#include <bit>
#include <cstdio>
#include <cstdlib>
#include <ostream>
#include <print>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

template<typename T>
void write(std::ostream &os, T value) {
    // GGUF makes no promises about endianness, but is usually little-endian.
    if (std::endian::native != std::endian::little) {
        value = std::byteswap(value);
    }

    os.write(reinterpret_cast<char const *>(&value), sizeof(T));
}

template<>
void write(std::ostream &os, gguf::GgmlType type) {
    std::uint32_t serialized_type = [=] {
        switch (type) {
        case gguf::GgmlType::F32:
            return 0;
        case gguf::GgmlType::F16:
            return 1;
        case gguf::GgmlType::Q4_0:
            return 2;
        case gguf::GgmlType::Q4_1:
            return 3;
        case gguf::GgmlType::Q5_0:
            return 6;
        case gguf::GgmlType::Q5_1:
            return 7;
        case gguf::GgmlType::Q8_0:
            return 8;
        case gguf::GgmlType::Q8_1:
            return 9;
        case gguf::GgmlType::Q2_K:
            return 10;
        case gguf::GgmlType::Q3_K:
            return 11;
        case gguf::GgmlType::Q4_K:
            return 12;
        case gguf::GgmlType::Q5_K:
            return 13;
        case gguf::GgmlType::Q6_K:
            return 14;
        case gguf::GgmlType::Q8_K:
            return 15;
        case gguf::GgmlType::IQ2_XXS:
            return 16;
        case gguf::GgmlType::IQ2_XS:
            return 17;
        case gguf::GgmlType::IQ3_XXS:
            return 18;
        case gguf::GgmlType::IQ1_S:
            return 19;
        case gguf::GgmlType::IQ4_NL:
            return 20;
        case gguf::GgmlType::IQ3_S:
            return 21;
        case gguf::GgmlType::IQ2_S:
            return 22;
        case gguf::GgmlType::IQ4_XS:
            return 23;
        case gguf::GgmlType::I8:
            return 24;
        case gguf::GgmlType::I16:
            return 25;
        case gguf::GgmlType::I32:
            return 26;
        case gguf::GgmlType::I64:
            return 27;
        case gguf::GgmlType::F64:
            return 28;
        case gguf::GgmlType::IQ1_M:
            return 29;
        case gguf::GgmlType::BF16:
            return 30;
        case gguf::GgmlType::TQ1_0:
            return 34;
        case gguf::GgmlType::TQ2_0:
            return 35;
        case gguf::GgmlType::MXFP4:
            return 39;
        }

        std::println(stderr, "Unhandled GGML type: {}", to_string(type));
        std::exit(1);
    }();

    write<std::uint32_t>(os, serialized_type);
}

auto make_gguf_header(
    std::span<gguf::GgufMetadataKV const> kvs, std::span<gguf::GgufTensorInfo const> tensor_infos)
    -> std::string {
    std::ostringstream ss;
    // magic
    ss << "GGUF";

    // version
    write<std::uint32_t>(ss, 3);

    // tensor count
    write<std::uint64_t>(ss, tensor_infos.size());

    // metadata count
    write<std::uint64_t>(ss, kvs.size());

    // metadata
    for (auto const &kv : kvs) {
        write<std::uint64_t>(ss, kv.key.size());
        ss << kv.key;

        switch (kv.valueType) {
        case gguf::GgufType::Uint32:
            write<std::int32_t>(ss, 4);
            write<std::uint32_t>(ss, std::get<std::uint32_t>(kv.value.v));
            break;
        case gguf::GgufType::Bool:
            write<std::int32_t>(ss, 7);
            write<std::int8_t>(ss, std::get<bool>(kv.value.v) ? 1 : 0);
            break;
        default:
            std::println(stderr, "Unhandled kv-type: {}", gguf::to_string(kv.valueType));
            std::exit(1);
        }
    }

    // tensor info
    for (auto const &info : tensor_infos) {
        write<std::uint64_t>(ss, info.name.size());
        ss << info.name;

        write<std::uint32_t>(ss, static_cast<std::uint32_t>(info.dimensions.size()));
        for (auto dimension : info.dimensions) {
            write<std::uint64_t>(ss, dimension);
        }

        write<gguf::GgmlType>(ss, info.type);
        write<std::uint64_t>(ss, info.offset);
    }

    return std::move(ss).str();
}

} // namespace

int main() {
    etest::Suite s{};

    s.add_test("empty model", [](etest::IActions &a) {
        auto gguf = std::istringstream{make_gguf_header({}, {})};
        auto metadata = gguf::read_gguf_metadata(gguf);
        a.require(metadata.has_value());
        a.expect_eq(metadata->metadata_kv.size(), 0);
        a.expect_eq(metadata->tensor_infos.size(), 0);
    });

    s.add_test("one metadata", [](etest::IActions &a) {
        std::vector<gguf::GgufMetadataKV> kvs{
            gguf::GgufMetadataKV{.key = "hello", .valueType = gguf::GgufType::Uint32, .value{16u}},
        };

        auto gguf = std::istringstream{make_gguf_header(kvs, {})};

        auto metadata = gguf::read_gguf_metadata(gguf);
        a.require(metadata.has_value());
        a.expect_eq(metadata->metadata_kv, kvs);
        a.expect_eq(metadata->tensor_infos.size(), 0);
    });

    s.add_test("one tensor info", [](etest::IActions &a) {
        std::vector<gguf::GgufTensorInfo> infos{
            gguf::GgufTensorInfo{
                .name = "yo",
                .dimensions = {6072, 6085},
                .type = gguf::GgmlType::I8,
                .offset = 6063,
            },
        };

        auto gguf = std::istringstream{make_gguf_header({}, infos)};

        auto metadata = gguf::read_gguf_metadata(gguf);
        a.require(metadata.has_value());
        a.expect_eq(metadata->metadata_kv.size(), 0);
        a.expect_eq(metadata->tensor_infos, infos);
    });

    s.add_test("both metadata and tensor info", [](etest::IActions &a) {
        std::vector<gguf::GgufMetadataKV> kvs{
            gguf::GgufMetadataKV{.key = "hello", .valueType = gguf::GgufType::Uint32, .value{16u}},
        };

        std::vector<gguf::GgufTensorInfo> infos{
            gguf::GgufTensorInfo{
                .name = "yo",
                .dimensions = {6072, 6085},
                .type = gguf::GgmlType::I8,
                .offset = 6063,
            },
        };

        auto gguf = std::istringstream{make_gguf_header(kvs, infos)};

        auto metadata = gguf::read_gguf_metadata(gguf);
        a.require(metadata.has_value());
        a.expect_eq(metadata->metadata_kv, kvs);
        a.expect_eq(metadata->tensor_infos, infos);
    });

    s.add_test("metadata types", [](etest::IActions &a) {
        std::vector<gguf::GgufMetadataKV> kvs{
            gguf::GgufMetadataKV{.key = "hello", .valueType = gguf::GgufType::Uint32, .value{16u}},
            gguf::GgufMetadataKV{.key = "ohayou", .valueType = gguf::GgufType::Bool, .value{true}},
        };

        auto gguf = std::istringstream{make_gguf_header(kvs, {})};

        auto metadata = gguf::read_gguf_metadata(gguf);
        a.require(metadata.has_value());
        a.expect_eq(metadata->metadata_kv, kvs);
        a.expect_eq(metadata->tensor_infos.size(), 0);
    });

    return s.run();
}
