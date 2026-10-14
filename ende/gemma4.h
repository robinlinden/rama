// SPDX-FileCopyrightText: 2026 Robin Lindén <dev@robinlinden.eu>
//
// SPDX-License-Identifier: BSD-2-Clause

#ifndef RAMA_ENDE_GEMMA4_H_
#define RAMA_ENDE_GEMMA4_H_

#include "ende/vocabulary.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ende::gemma4 {
namespace detail {

struct Merge {
    std::string_view lhs;
    std::string_view rhs;
};

constexpr auto parse_merges(std::span<std::string const> gguf_merges)
    -> std::optional<std::vector<Merge>> {
    static constexpr auto kSplitOnce =
        [](std::string_view str, char sep) -> std::pair<std::string_view, std::string_view> {
        if (auto p = str.find(sep); p != std::string_view::npos) {
            return {str.substr(0, p), str.substr(p + 1)};
        }

        return {str, ""};
    };

    std::vector<Merge> merges;
    merges.reserve(gguf_merges.size());

    for (auto const &merge : gguf_merges) {
        auto [lhs, rhs] = kSplitOnce(merge, ' ');
        if (rhs.empty()) {
            return std::nullopt;
        }

        merges.emplace_back(lhs, rhs);
    }

    return merges;
}

constexpr auto apply_merges(std::vector<std::string> symbols, std::span<Merge const> merges)
    -> std::vector<std::string> {
    while (symbols.size() > 1) {
        std::optional<std::size_t> best_position;
        std::size_t best_rank = merges.size();

        for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
            for (std::size_t rank = 0; rank < merges.size(); ++rank) {
                if (rank < best_rank && symbols[i] == merges[rank].lhs &&
                    symbols[i + 1] == merges[rank].rhs) {
                    best_rank = rank;
                    best_position = i;
                }
            }
        }

        if (!best_position) {
            break;
        }

        symbols[*best_position] += symbols[*best_position + 1];
        symbols.erase(symbols.begin() + *best_position + 1);
    }

    return symbols;
}

constexpr auto into_symbols(std::string_view text) -> std::vector<std::string> {
    static constexpr std::string_view kEscapedSpace = "\xE2\x96\x81";

    static constexpr auto kUtf8CodePointLength = [](char first) {
        if ((first & 0b1110'0000) == 0b1100'0000) {
            return 2;
        }

        if ((first & 0b1111'0000) == 0b1110'0000) {
            return 3;
        }

        if ((first & 0b1111'1000) == 0b1111'0000) {
            return 4;
        }

        return 1;
    };

    static constexpr auto kIsContinuationByte = [](char ch) {
        return (ch & 0b1100'0000) == 0b1000'0000;
    };

    std::string normalized;

    // Spaces are encoded as U+2581 (\xE2\x96\x81 in UTF-8), so they take up 3
    // bytes instead of 1.
    normalized.reserve(text.size() + std::ranges::count(text, ' ') * 2);
    for (char c : text) {
        normalized += c == ' ' ? kEscapedSpace : std::string_view{&c, 1};
    }

    std::vector<std::string> symbols;
    for (std::size_t offset = 0; offset < normalized.size();) {
        std::size_t length = kUtf8CodePointLength(normalized[offset]);

        // Handle partial and malformed code points by treating them as byte-symbols.
        if (offset + length > normalized.size()) {
            length = 1;
        } else {
            for (std::size_t i = 1; i < length; ++i) {
                if (!kIsContinuationByte(normalized[offset + i])) {
                    length = 1;
                    break;
                }
            }
        }

        symbols.emplace_back(normalized.substr(offset, length));
        offset += length;
    }

    return symbols;
}

} // namespace detail

// TODO(robinlinden): Sane error messages.
constexpr auto into_tokens(
    Vocabulary const &vocab, std::string_view text, std::span<std::string const> gguf_merges)
    -> std::optional<std::vector<std::uint32_t>> {
    auto merges = detail::parse_merges(gguf_merges);
    if (!merges) {
        return std::nullopt;
    }

    auto symbols = apply_merges(detail::into_symbols(text), *merges);

    std::vector<std::uint32_t> tokens;
    for (auto const &symbol : symbols) {
        // TODO(robinlinden): Nicer vocab API.
        if (auto token = vocab.tokens_by_value.find(symbol); token != vocab.tokens_by_value.end()) {
            tokens.push_back(token->second);
            continue;
        }

        for (char byte : symbol) {
            auto byte_token = std::format("<0x{:2X}>", byte);
            auto token = vocab.tokens_by_value.find(byte_token);
            if (token == vocab.tokens_by_value.end()) {
                return std::nullopt;
            }

            tokens.push_back(token->second);
        }
    }

    return tokens;
}

} // namespace ende::gemma4

#endif
