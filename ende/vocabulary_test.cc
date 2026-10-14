// SPDX-FileCopyrightText: 2026 Robin Lindén <dev@robinlinden.eu>
//
// SPDX-License-Identifier: BSD-2-Clause

#include "ende/vocabulary.h"

#include <etest/etest2.h>

#include <string>
#include <vector>

int main() {
    etest::Suite s{};

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

    return s.run();
}
