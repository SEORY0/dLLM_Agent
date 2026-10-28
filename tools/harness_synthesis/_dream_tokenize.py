r"""Tokenize text with Dream-Coder's tokenizer only. CPU, no model weights.

Runs under tools/dream_runtime/venv, like ``_dream_generate.py``. Loads only the
tokenizer files from the model directory. ``AutoModel`` is never imported, so nothing
touches torch or the GPU. RQ4b uses it to turn a benchmark body, which the model did
not generate, into the token ids a remask round needs, and to count a span's tokens.

    tools/dream_runtime/venv/bin/python tools/harness_synthesis/_dream_tokenize.py \\
      --model-path /data/dllm-agent/cache-dream-coder --texts texts.json --out out.json
"""

import argparse
import json

from transformers import AutoTokenizer


def main() -> None:
    """Write ``[{"ids": [...], "pieces": [...]}, ...]``, one entry per input text."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--texts", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with open(args.texts) as handle:  # noqa: PTH123 - tools/ script.
        texts: list[str] = json.load(handle)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    result = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False).input_ids
        pieces = [tokenizer.decode([i], skip_special_tokens=False) for i in ids]
        result.append({"ids": ids, "pieces": pieces})
    special = {"mask_token": tokenizer.mask_token, "mask_token_id": tokenizer.mask_token_id}
    with open(args.out, "w") as handle:  # noqa: PTH123 - tools/ script.
        json.dump({"special": special, "tokens": result}, handle)


if __name__ == "__main__":
    main()
