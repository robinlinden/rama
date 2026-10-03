// SPDX-FileCopyrightText: 2026 Robin Lindén <dev@robinlinden.eu>
//
// SPDX-License-Identifier: BSD-2-Clause

#include "gguf/gguf.h"

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(std::uint8_t const *data, std::size_t size);

extern "C" int LLVMFuzzerTestOneInput(std::uint8_t const *data, std::size_t size) {
    auto ss = std::stringstream{std::string{reinterpret_cast<char const *>(data), size}};
    std::ignore = gguf::read_gguf_metadata(ss);
    return 0;
}
