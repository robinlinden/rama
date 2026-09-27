// SPDX-FileCopyrightText: 2026 Robin Lindén <dev@robinlinden.eu>
//
// SPDX-License-Identifier: BSD-2-Clause

#include "ende/ende.h"

#include <etest/etest2.h>

#include <string>
#include <vector>

int main() {
    etest::Suite s{};

    s.add_test("Vocabulary::from_tokens", [](etest::IActions &a) {
        std::vector<std::string> tokens{"hi", "hello", "goodbye", "farewell"};
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
                },
            });
    });

    s.add_test("into_byte_tokens", [](etest::IActions &a) {
        using Tokens = std::vector<std::string>;
        a.expect_eq(ende::into_byte_tokens("hello"), Tokens{"h", "e", "l", "l", "o"});

        // UTF-8 characters are split into bytes.
        // U+03BB (GREEK SMALL LETTER LAMBDA).
        a.expect_eq(ende::into_byte_tokens("\u03BB"), Tokens{"\xCE", "\xBB"});
    });

    return s.run();
}
