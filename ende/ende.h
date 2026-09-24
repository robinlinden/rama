// SPDX-FileCopyrightText: 2026 Robin Lindén <dev@robinlinden.eu>
//
// SPDX-License-Identifier: BSD-2-Clause

#ifndef RAMA_ENDE_ENDE_H_
#define RAMA_ENDE_ENDE_H_

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

        for (std::size_t i = 0; i < tokens.size(); ++i) {
            token_ids.push_back(i);
            token_values.push_back(tokens[i]);
        }

        return Vocabulary{
            .tokens_by_id{std::move(tokens)},
            .tokens_by_value{std::move(token_values), std::move(token_ids)},
        };
    }
};

constexpr auto starting_tokens_for_prompt(std::string_view prompt) -> std::vector<std::string> {
    std::vector<std::string> tokens;
    tokens.resize(prompt.size());
    for (auto c : prompt) {
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
