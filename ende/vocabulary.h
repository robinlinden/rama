// SPDX-FileCopyrightText: 2026 Robin Lindén <dev@robinlinden.eu>
//
// SPDX-License-Identifier: BSD-2-Clause

#ifndef RAMA_ENDE_VOCABULARY_H_
#define RAMA_ENDE_VOCABULARY_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <flat_map>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ende {

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
            assert(opts.unknown_token_id.has_value());
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

} // namespace ende

#endif
