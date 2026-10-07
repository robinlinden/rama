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
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ende {

struct Merge {
    std::string lhs;
    std::string rhs;
    std::int32_t rank{};
};

struct Vocabulary {
    std::vector<std::string> tokens_by_id;
    std::flat_map<std::string_view, std::uint32_t> tokens_by_value;

    bool operator==(Vocabulary const &) const = default;

    constexpr static auto from_tokens(std::vector<std::string> tokens) -> Vocabulary {
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
        };
    }

    constexpr auto token_by_value(char c) const -> std::uint32_t {
        return tokens_by_value.at(std::string_view{&c, 1});
    }

    constexpr auto token_by_value(std::string_view s) const -> std::uint32_t {
        return tokens_by_value.at(s);
    }

    constexpr auto token_by_id(std::uint32_t id) const -> std::string_view {
        assert(id < tokens_by_id.size());
        return tokens_by_id[id];
    }
};

constexpr auto into_byte_tokens(std::string_view text) -> std::vector<std::string> {
    std::vector<std::string> tokens;
    tokens.reserve(text.size());
    for (auto c : text) {
        tokens.push_back(std::string{c});
    }

    return tokens;
}

// TODO(robinlinden): This is the most naive implementation. Do something better.
inline auto apply_merges(std::vector<std::string> tokens_to_merge, std::span<Merge const> merges)
    -> std::vector<std::string> {
    while (true) {
        std::optional<std::size_t> best_merge_index;
        std::int32_t lowest_rank = std::numeric_limits<std::int32_t>::max();

        // Find the pair w/ the lowest rank, if any.
        for (std::size_t i = 0; i < tokens_to_merge.size() - 1; ++i) {
            for (auto const &merge : merges) {
                if (tokens_to_merge[i] == merge.lhs && tokens_to_merge[i + 1] == merge.rhs) {
                    if (merge.rank < lowest_rank) {
                        lowest_rank = merge.rank;
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
        tokens_to_merge[*best_merge_index] += tokens_to_merge[*best_merge_index + 1];
        tokens_to_merge.erase(tokens_to_merge.begin() + *best_merge_index + 1);
    }

    return tokens_to_merge;
}

} // namespace ende

#endif
