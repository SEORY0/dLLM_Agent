# jsoncons UBJSON non-empty continuation libFuzzer harness

Write one minimal C++17 libFuzzer harness for `jsoncons` at commit
`af497a1c1ea9043f4ad8d799617b7d00d6676153`.

This is a follow-up campaign after empty input was independently preserved and
triaged. Return `0` immediately when `size == 0` so the continuation campaign can
reach deeper non-empty paths. For every non-empty input, exercise both APIs:

1. `jsoncons::ubjson::try_decode_ubjson<jsoncons::json>` on a
   `std::vector<std::uint8_t>` built from the fuzzer bytes.
2. `jsoncons::ubjson::ubjson_bytes_cursor` on the same vector, using a
   `std::error_code` and repeatedly calling `next(ec)` until `done()` or an error.

Requirements:

- Output only compilable source code, without Markdown fences or explanation.
- Define exactly
  `extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size)`.
- Return `0` for normal and malformed inputs.
- The source must begin with all seven of these exact include directives:
  `#include <cstddef>`, `#include <cstdint>`, `#include <system_error>`,
  `#include <vector>`, `#include <jsoncons/json.hpp>`,
  `#include <jsoncons_ext/ubjson/ubjson.hpp>`, and
  `#include <jsoncons_ext/ubjson/ubjson_cursor.hpp>`.
- Do not use any other headers.
- Do not use files, streams, networking, processes, threads, logging, assertions,
  exceptions for control flow, global mutable state, or a `main` function.
- Use these exact API shapes; do not invent output-parameter or iterator overloads:
  - `auto result = jsoncons::ubjson::try_decode_ubjson<jsoncons::json>(bytes);`
    followed by `(void)result;`.
  - `std::error_code ec;` followed by
    `jsoncons::ubjson::ubjson_bytes_cursor cursor(bytes, ec);`.
  - `while (!ec && !cursor.done()) { cursor.next(ec); }`.
- Do not create a separate `jsoncons::json` output variable for the decode call.
- Keep the harness deterministic and under 60 lines. Omit comments if necessary
  and finish `return 0;` plus the closing function brace within the token budget.

This is a defensive fuzzing harness for a locally pinned open-source target. The
generated source will only be compiled and executed in a network-disabled,
read-only, capability-dropped container.
