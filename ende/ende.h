// SPDX-FileCopyrightText: 2026 Robin Lindén <dev@robinlinden.eu>
//
// SPDX-License-Identifier: BSD-2-Clause

#ifndef RAMA_ENDE_ENDE_H_
#define RAMA_ENDE_ENDE_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <flat_map>
#include <limits>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ende {

struct Merge {
    std::uint32_t lhs{};
    std::uint32_t rhs{};
    std::uint32_t merged{};

    bool operator==(Merge const &) const = default;
};

struct VocabOptions {
    std::optional<std::uint32_t> unknown_token_id{};

    bool operator==(VocabOptions const &) const = default;
};

struct Vocabulary {
    std::vector<std::string> tokens_by_id;
    std::flat_map<std::string_view, std::uint32_t> tokens_by_value;
    VocabOptions opts;

    bool operator==(Vocabulary const &) const = default;

    constexpr static auto
    from_tokens(std::vector<std::string> tokens, VocabOptions const &opts = {}) -> Vocabulary {
        std::vector<std::uint32_t> token_ids;
        token_ids.reserve(tokens.size());
        std::vector<std::string_view> token_values;
        token_values.reserve(tokens.size());

        assert(tokens.size() < std::numeric_limits<std::uint32_t>::max());
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            token_ids.push_back(static_cast<std::uint32_t>(i));
            token_values.push_back(tokens[i]);
        }

        return Vocabulary{
            .tokens_by_id{std::move(tokens)},
            .tokens_by_value{std::move(token_values), std::move(token_ids)},
            .opts{opts},
        };
    }

    constexpr auto token_by_value(char c) const -> std::uint32_t {
        auto token = tokens_by_value.find(std::string_view{&c, 1});
        if (token == tokens_by_value.end()) {
            return opts.unknown_token_id.value();
        }

        return token->second;
    }

    constexpr auto token_by_value(std::string_view s) const -> std::uint32_t {
        return tokens_by_value.at(s);
    }

    constexpr auto token_by_id(std::uint32_t id) const -> std::string_view {
        assert(id < tokens_by_id.size());
        return tokens_by_id[id];
    }
};

constexpr auto split_once(std::string_view str, char sep)
    -> std::pair<std::string_view, std::string_view> {
    if (auto p = str.find(sep); p != std::string_view::npos) {
        return {str.substr(0, p), str.substr(p + 1)};
    }

    return {str, ""};
}

constexpr auto parse_merges(Vocabulary const &vocab, std::span<std::string const> gguf_merges)
    -> std::vector<Merge> {
    std::vector<Merge> merges;
    merges.reserve(gguf_merges.size());

    for (std::size_t i = 0; i < gguf_merges.size(); ++i) {
        auto [lhs, rhs] = split_once(gguf_merges[i], ' ');
        merges.emplace_back(
            vocab.token_by_value(lhs),
            vocab.token_by_value(rhs),
            vocab.token_by_value(std::format("{}{}", lhs, rhs)));
    }

    return merges;
}

constexpr auto into_byte_tokens(Vocabulary const &vocab, std::string_view text)
    -> std::vector<std::uint32_t> {
    std::vector<std::uint32_t> tokens;
    tokens.reserve(text.size());
    for (auto c : text) {
        tokens.push_back(vocab.token_by_value(c));
    }

    return tokens;
}

// TODO(robinlinden): This is the most naive implementation. Do something better.
inline auto apply_merges(std::vector<std::uint32_t> tokens_to_merge, std::span<Merge const> merges)
    -> std::vector<std::uint32_t> {
    while (true) {
        std::optional<std::size_t> best_merge_index;
        std::optional<std::uint32_t> merged_token;
        std::uint32_t lowest_rank = std::numeric_limits<std::uint32_t>::max();

        assert(merges.size() < std::numeric_limits<std::uint32_t>::max());

        // Find the pair w/ the lowest rank, if any.
        for (std::size_t i = 0; i < tokens_to_merge.size() - 1; ++i) {
            for (std::uint32_t merge_idx = 0; merge_idx < merges.size(); ++merge_idx) {
                auto const &merge = merges[merge_idx];
                if (tokens_to_merge[i] == merge.lhs && tokens_to_merge[i + 1] == merge.rhs) {
                    if (merge_idx < lowest_rank) {
                        lowest_rank = merge_idx;
                        merged_token = merge.merged;
                        best_merge_index = i;
                    }
                }
            }
        }

        // No merges left to do.
        if (!best_merge_index.has_value()) {
            break;
        }

        std::println(
            "Merging {} and {}!",
            tokens_to_merge[*best_merge_index],
            tokens_to_merge[*best_merge_index + 1]);

        // Perform the merge.
        tokens_to_merge[*best_merge_index] = *merged_token;
        tokens_to_merge.erase(tokens_to_merge.begin() + *best_merge_index + 1);
    }

    return tokens_to_merge;
}

} // namespace ende

#endif
