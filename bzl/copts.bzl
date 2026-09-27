# SPDX-FileCopyrightText: 2022-2026 Robin Lindén <dev@robinlinden.eu>
#
# SPDX-License-Identifier: BSD-2-Clause

"""Common copts for rama targets. Originally from robinlinden/hastur."""

RAMA_LINUX_WARNING_FLAGS = [
    "-Wall",
    "-Wextra",
    "-pedantic-errors",
    "-Wctad-maybe-unsupported",
    "-Wdouble-promotion",
    "-Wformat=2",
    "-Wmissing-declarations",
    "-Wnon-virtual-dtor",
    "-Wnull-dereference",
    "-Woverloaded-virtual",
    "-Wshadow",
    "-Wsign-compare",
    "-Wundef",
    "-Wunreachable-code",
    "-Wuninitialized",
    "-Wdeprecated",
    "-Wunused",

    # Common idiom for zeroing members.
    "-Wno-missing-field-initializers",
]

RAMA_CLANG_WARNING_FLAGS = RAMA_LINUX_WARNING_FLAGS + [
    "-Wused-but-marked-unused",
    "-Wundefined-func-template",
    "-Wundefined-reinterpret-cast",
]

RAMA_MSVC_WARNING_FLAGS = [
    # More warnings.
    "/W4",
]

RAMA_COPTS = select({
    "@rules_cc//cc/compiler:clang": RAMA_CLANG_WARNING_FLAGS,
    "@rules_cc//cc/compiler:gcc": RAMA_LINUX_WARNING_FLAGS,
    "@rules_cc//cc/compiler:msvc-cl": RAMA_MSVC_WARNING_FLAGS,
})
