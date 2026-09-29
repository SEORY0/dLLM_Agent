r"""Run one Dream-Coder generation and dump its raw commit trajectory as JSON.

Runs under tools/dream_runtime/venv (transformers==4.46.2), which cannot import
dllm_agent -- its editable install lives in a .pth the project venv's site-packages
holds, invisible once the Dream venv's sys.path is extended by a plain-path .pth
line rather than a real site-directory scan (see tools/dream_runtime/setup.sh).

So this script does only the raw tensor work (softmax/entropy/margin for
newly-committed positions, read off Dream's own tokens hook) and dumps plain JSON;
dream_backend_smoke.py, run under the project venv, replays it through the real
TrajectoryRecorder/regions/remask modules.

    tools/dream_runtime/venv/bin/python tools/harness_synthesis/_dream_generate.py \\
      --model-path /data/dllm-agent/cache-dream-coder --prompt "..." --seed 1 --out out.json
"""

import argparse
import json

import torch
from transformers import AutoModel, AutoTokenizer


def main() -> None:
    """Generate once, recording every step's token snapshot and new commits."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--new-tokens", type=int, default=256)
    parser.add_argument("--steps", type=int, default=128)
    # alg_temp=0.0 (the default below) makes Dream's reveal-order selection an
    # argmax/topk, not a torch.multinomial draw -- the seed then never actually enters
    # token selection, and different --seed values mostly reproduce the same output
    # (observed: RQ1 measurement, only floating-point-kernel jitter breaks the tie
    # occasionally). A nonzero alg_temp routes reveal-order through
    # torch.multinomial(..., generator seeded by --seed), giving genuine independent
    # draws -- required for any experiment (like RQ1) that assumes N seeds are N
    # independent samples. Task 8/8b's conclusions don't need this: they asked whether
    # a prompt design reliably reproduces one outcome, so the determinism there was
    # incidental, not a flaw in what was measured.
    parser.add_argument("--alg-temp", type=float, default=0.0)
    # For a remask round (RQ3): the full post-prompt continuation, as a JSON list of
    # ints, with mask_token_id at the positions to regenerate and every other position
    # holding the token the previous round already committed. When given, this
    # overrides --new-tokens (set to len(continuation)) and is injected as the initial
    # state instead of Dream's own default all-mask continuation.
    parser.add_argument("--init-continuation-file", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    init_continuation: list[int] | None = None
    if args.init_continuation_file is not None:
        with open(args.init_continuation_file) as handle:  # noqa: PTH123 - tools/ script.
            init_continuation = json.load(handle)
        args.new_tokens = len(init_continuation)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        .to("cuda")
        .eval()
    )
    mask_token_id = model.config.mask_token_id

    chat_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    ).input_ids
    prefill_ids = tokenizer("```cpp\n", return_tensors="pt", add_special_tokens=False).input_ids
    prompt_ids = torch.cat([chat_ids, prefill_ids], dim=1).cuda()
    prompt_length = int(prompt_ids.shape[1])

    steps_log: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    previous_masked: set[int] | None = None

    def tokens_hook(
        step: int | None, x: torch.Tensor, logits: torch.Tensor | None
    ) -> torch.Tensor:
        nonlocal previous_masked
        if step is None and init_continuation is not None:
            continuation = torch.tensor([init_continuation], device=x.device, dtype=x.dtype)
            x = torch.cat([prompt_ids, continuation], dim=1)
        token_ids = x[0].tolist()
        steps_log.append({"step": step, "token_ids": token_ids})
        masked = {position for position, token in enumerate(token_ids) if token == mask_token_id}
        if step is not None and logits is not None and previous_masked is not None:
            newly_committed = previous_masked - masked
            if newly_committed:
                probs = torch.softmax(logits[0].float(), dim=-1)
                top_two = probs.topk(2, dim=-1).values
                entropy_all = -(probs * (probs + 1e-12).log()).sum(dim=-1)
                for position in sorted(newly_committed):
                    token_id = int(token_ids[position])
                    events.append(
                        {
                            "step": step,
                            "position": position,
                            "token_id": token_id,
                            "confidence": float(probs[position, token_id]),
                            "entropy": float(entropy_all[position]),
                            "margin": float(top_two[position, 0] - top_two[position, 1]),
                        }
                    )
        previous_masked = masked
        return x

    torch.manual_seed(args.seed)
    output = model.diffusion_generate(
        prompt_ids,
        attention_mask=torch.ones_like(prompt_ids),
        max_new_tokens=args.new_tokens,
        steps=args.steps,
        temperature=0.5,
        top_p=0.95,
        alg="entropy",
        alg_temp=args.alg_temp,
        return_dict_in_generate=True,
        generation_tokens_hook_func=tokens_hook,
    )
    generated_ids = output.sequences[0, prompt_length:].tolist()
    generated_pieces = [
        tokenizer.decode([token_id], skip_special_tokens=False) for token_id in generated_ids
    ]

    payload = {
        "prompt_length": prompt_length,
        "mask_token_id": int(mask_token_id),
        "generated_token_ids": generated_ids,
        "generated_token_pieces": generated_pieces,
        "decoded_text": "".join(generated_pieces),
        "steps_log": steps_log,
        "events": events,
    }
    with open(args.out, "w") as handle:  # noqa: PTH123 - tools/ script, not under src/
        json.dump(payload, handle)


if __name__ == "__main__":
    main()
