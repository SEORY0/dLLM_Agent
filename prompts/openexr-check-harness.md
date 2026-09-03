# OpenEXR in-memory check harness

Output only a minimal deterministic C++17 libFuzzer source file, with no Markdown or explanation.

Requirements:

- Include only `cstddef`, `cstdint`, `ImfNamespace.h`, and `ImfCheckFile.h`.
- Define `extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size)` and return `0`.
- For every input, call `OPENEXR_IMF_NAMESPACE::checkOpenEXRFile` twice on the in-memory bytes.
- Pass `reinterpret_cast<const char*>(data)` as the first argument in both calls.
- Pass `reduceMemory=true` and `reduceTime=true` in both calls; pass `runCoreCheck=false` once and `runCoreCheck=true` once.
- Do not use files, processes, networking, threads, logging, a `main` function, or global mutable state.
- Keep the complete source well within 192 generated tokens. Omit comments.
