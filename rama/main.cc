// SPDX-FileCopyrightText: 2026 Robin Lindén <dev@robinlinden.eu>
//
// SPDX-License-Identifier: BSD-2-Clause

#include "ende/gemma4.h"
#include "ende/vocabulary.h"
#include "gguf/gguf.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <format>
#include <fstream>
#include <print>
#include <span>
#include <string>
#include <variant>

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
        std::println(stderr, "{}", gguf::to_string(metadata.error()));
        return 1;
    }

    std::println("GGUF version: {}", metadata->version);

    std::println("Tensor count: {}", metadata->tensor_infos.size());

    std::println("Metadata count: {}", metadata->metadata_kv.size());

    for (auto const &metadata_kv : metadata->metadata_kv) {
        // Only print the first 128 characters of the value to avoid flooding the terminal.
        std::println(
            "* {}: {}", metadata_kv.key, gguf::to_string(metadata_kv.value).substr(0, 128));
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

    assert(std::holds_alternative<std::string>(maybe_model->value));
    auto const &model = std::get<std::string>(maybe_model->value);
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

    auto maybe_tokens = std::ranges::find(
        metadata->metadata_kv, "tokenizer.ggml.tokens", &gguf::GgufMetadataKV::key);
    if (maybe_tokens == std::end(metadata->metadata_kv)) {
        std::println(stderr, "Missing tokenizer.ggml.tokens metadata? :(");
        return 1;
    }

    auto maybe_unknown_token_id = std::ranges::find(
        metadata->metadata_kv, "tokenizer.ggml.unknown_token_id", &gguf::GgufMetadataKV::key);

    auto opts = ende::VocabOptions{};
    if (maybe_unknown_token_id != std::end(metadata->metadata_kv)) {
        assert(std::holds_alternative<std::uint32_t>(maybe_unknown_token_id->value));
        opts.unknown_token_id = std::get<std::uint32_t>(maybe_unknown_token_id->value);
    }

    std::println();

    // TODO(robinlinden): If these exist, they're required to be array[string].
    // We should probably enforce these things when parsing.
    assert(std::holds_alternative<std::vector<std::string>>(maybe_merges->value));
    assert(std::holds_alternative<std::vector<std::string>>(maybe_tokens->value));

    auto vocab = ende::Vocabulary::from_tokens(
        std::get<std::vector<std::string>>(std::move(maybe_tokens->value)), opts);

    auto const &raw_merges = std::get<std::vector<std::string>>(maybe_merges->value);
    auto prompt_tokens = ende::gemma4::into_tokens(vocab, argv[2], raw_merges);
    if (!prompt_tokens) {
        std::println(stderr, "Things went wrong trying to tokenize the input.");
        return 1;
    }

    std::println("Merged tokens:");
    for (std::size_t i = 0; i < prompt_tokens->size(); ++i) {
        auto const &token = prompt_tokens->at(i);
        std::println("{}: {} ({})", i, token, vocab.token_by_id(token));
    }

    // TODO(robinlinden): String tokens -> actual numerical tokens from the metadata.
}
