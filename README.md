# dLLM Agent

Offline deterministic vulnerability-discovery MVP using a synthetic, versioned
Stable-DiffCoder-compatible replay. `ReplaySampler` to validator to hashed JSON is the
reproducible minimum spine, not a fully dLLM-optimized harness. `ReplaySampler` is a
deterministic test double for the first-class dLLM-native seam: trajectory capture plus
trajectory-aware ranking and budget routing. Stock Stable tools would run only after a
sample is finalized; this MVP never downloads a model or executes an intermediate
denoising state.

```bash
uv sync --all-groups
uv run dllm-agent scan fixtures/cwe121/vuln.c \
  --fixture fixtures/replay/stable-diffcoder-cwe121.json \
  --output /tmp/dllm-scan.json
uv run dllm-agent plan configs/owned-fixture-mvp.json \
  --output /tmp/dllm-plan.json
```

The scan compiles the repository-owned fixture with AddressSanitizer, validates only
the highest-ranked finalized candidate, and writes an auditable JSON report.
