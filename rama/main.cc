// SPDX-FileCopyrightText: 2026 Robin Lindén <dev@robinlinden.eu>
//
// SPDX-License-Identifier: BSD-2-Clause

#include "ende/ende.h"
#include "gguf/gguf.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <variant>

namespace {

constexpr auto split_once(std::string_view str, char sep)
    -> std::pair<std::string_view, std::string_view> {
    if (auto p = str.find(sep); p != std::string_view::npos) {
        return {str.substr(0, p), str.substr(p + 1)};
    }

    return {str, ""};
}

auto parse_merges(std::span<gguf::GgufValue const> gguf_merges) -> std::vector<ende::Merge> {
    std::vector<ende::Merge> merges;
    merges.reserve(gguf_merges.size());

    for (std::size_t i = 0; i < gguf_merges.size(); ++i) {
        auto const &gguf_merge = gguf_merges[i].v;
        assert(std::holds_alternative<std::string>(gguf_merge));

        auto [lhs, rhs] = split_once(std::get<std::string>(gguf_merge), ' ');
        merges.emplace_back(std::string{lhs}, std::string{rhs}, static_cast<std::uint32_t>(i));
    }

    return merges;
}

} // namespace

auto main(int argc, char **argv) -> int {
    if (argc < 2) {
        char const *program_name = argv[0] != nullptr ? argv[0] : "<bin>";
        std::println(stderr, "Usage: {} <GGUF path> [prompt]", program_name);
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

    if (argc < 3) {
        return 0;
    }

    auto maybe_model = std::ranges::find(
        metadata->metadata_kv, "tokenizer.ggml.model", &gguf::GgufMetadataKV::key);
    if (maybe_model == std::end(metadata->metadata_kv)) {
        std::println(stderr, "Missing tokenizer.ggml.model metadata? :(");
        return 1;
    }

    assert(std::holds_alternative<std::string>(maybe_model->value.v));
    auto const &model = std::get<std::string>(maybe_model->value.v);
    if (model != "gemma4") {
        std::println(stderr, "Only gemma4 models are supported right now");
        return 1;
    }

    auto maybe_merges = std::ranges::find(
        metadata->metadata_kv, "tokenizer.ggml.merges", &gguf::GgufMetadataKV::key);
    if (maybe_merges == std::end(metadata->metadata_kv)) {
        std::println(stderr, "Missing tokenizer.ggml.merges metadata? :(");
        return 1;
    }

    std::println();

    // TODO(robinlinden): If this exists, it's required to be array[string]. We
    // should probably enforce these things when parsing.
    assert(std::holds_alternative<std::vector<gguf::GgufValue>>(maybe_merges->value.v));
    auto const &raw_merges = std::get<std::vector<gguf::GgufValue>>(maybe_merges->value.v);
    auto merges = parse_merges(raw_merges);

    auto prompt_tokens = ende::starting_tokens_for_prompt(argv[2]);
    prompt_tokens = ende::apply_merges(std::move(prompt_tokens), merges);

    std::println("Merged tokens:");
    for (std::size_t i = 0; i < prompt_tokens.size(); ++i) {
        auto const &token = prompt_tokens[i];
        std::println("{}: {}", i, token);
    }

    // TODO(robinlinden): String tokens -> actual numerical tokens from the metadata.
}
