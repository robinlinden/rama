// SPDX-FileCopyrightText: 2026 Robin Lindén <dev@robinlinden.eu>
//
// SPDX-License-Identifier: BSD-2-Clause

#include "ende/ende.h"

#include <etest/etest2.h>

#include <string>
#include <vector>

int main() {
    etest::Suite s{};

    s.add_test("Merge::parse", [](etest::IActions &a) {
        std::vector<std::string> gguf_tokens{"h", "e", "he", "ll", "hell", "hello", "o", "l"};
        auto vocab = ende::Vocabulary::from_tokens(gguf_tokens);

        auto gguf_merges = std::vector<std::string>{"h e", "he ll", "hell o", "l l"};
        auto merges = ende::parse_merges(vocab, gguf_merges);

        a.expect_eq(
            merges,
            std::vector<ende::Merge>{
                {0, 1, 2, 0}, // h+e => he
                {2, 3, 4, 1}, // he+ll => hell
                {4, 6, 5, 2}, // hell+o => hello
                {7, 7, 3, 3}, // l+l => ll
            });
    });

    s.add_test("Vocabulary", [](etest::IActions &a) {
        std::vector<std::string> tokens{"hi", "hello", "goodbye", "farewell", "?"};
        auto vocab = ende::Vocabulary::from_tokens(tokens);
        a.expect_eq(
            vocab,
            ende::Vocabulary{
                .tokens_by_id{tokens},
                .tokens_by_value{
                    {"hi", 0},
                    {"hello", 1},
                    {"goodbye", 2},
                    {"farewell", 3},
                    {"?", 4},
                },
            });

        a.expect_eq(vocab.token_by_id(2), "goodbye");
        a.expect_eq(vocab.token_by_value("goodbye"), 2);
        a.expect_eq(vocab.token_by_value('?'), 4);
    });

    s.add_test("into_byte_tokens", [](etest::IActions &a) {
        using Tokens = std::vector<std::uint32_t>;
        auto vocab = ende::Vocabulary::from_tokens({"h", "e", "l", "o", "\xCE", "\xBB"});
        a.expect_eq(ende::into_byte_tokens(vocab, "hello"), Tokens{0, 1, 2, 2, 3});

        // UTF-8 characters are split into bytes.
        // U+03BB (GREEK SMALL LETTER LAMBDA).
        a.expect_eq(ende::into_byte_tokens(vocab, "\u03BB"), Tokens{4, 5});
    });

    return s.run();
}
