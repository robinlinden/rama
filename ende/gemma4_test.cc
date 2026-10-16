// SPDX-FileCopyrightText: 2026 Robin Lindén <dev@robinlinden.eu>
//
// SPDX-License-Identifier: BSD-2-Clause

#include "ende/gemma4.h"

#include "ende/vocabulary.h"

#include <etest/etest2.h>

#include <cstdint>
#include <string>
#include <vector>

int main() {
    etest::Suite s{};

    s.add_test("into_gemma4_tokens", [](etest::IActions &a) {
        using Tokens = std::vector<std::uint32_t>;
        // The weird first token is how gemma4 represents space tokens.
        auto vocab = ende::Vocabulary::from_tokens(
            {"▁", "h", "i", "hi", "<0xF0>", "<0x9F>", "<0x90>", "<0xA6>"});
        auto merges = std::vector<std::string>{"h i"};

        a.expect_eq(ende::gemma4::into_tokens(vocab, " ", merges), Tokens{0});
        a.expect_eq(ende::gemma4::into_tokens(vocab, "h i", merges), Tokens{1, 0, 2});
        a.expect_eq(ende::gemma4::into_tokens(vocab, "hi", merges), Tokens{3});
        a.expect_eq(ende::gemma4::into_tokens(vocab, "🐦", merges), Tokens{4, 5, 6, 7});
    });

    return s.run();
}
