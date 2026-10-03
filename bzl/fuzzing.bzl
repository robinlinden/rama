# SPDX-FileCopyrightText: 2026 Robin Lindén <dev@robinlinden.eu>
#
# SPDX-License-Identifier: BSD-2-Clause

RAMA_FUZZ_PLATFORMS = select({
    "@platforms//os:linux": [],
    "//conditions:default": ["@platforms//:incompatible"],
})
