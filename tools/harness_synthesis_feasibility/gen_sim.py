"""Feasibility sim: Dream-Coder harness generation with trajectory trace + remask arms."""
import json, random, re, sys, torch
from transformers import AutoModel, AutoTokenizer

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
OUT = sys.argv[2] if len(sys.argv) > 2 else "variants.json"
P = "/data/dllm-agent/cache-dream-coder"
tok = AutoTokenizer.from_pretrained(P, trust_remote_code=True)
m = AutoModel.from_pretrained(P, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
MASK = m.config.mask_token_id
PROMPT = (
    "Write a complete libFuzzer harness in C++ for the OpenEXR library (v3.4). "
    "Include <ImfCheckFile.h>, <ImfNamespace.h> and any other OpenEXR headers you need. "
    "Define extern \"C\" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) that feeds the "
    "fuzz bytes into OpenEXR parsing APIs (for example checkOpenEXRFile, InputFile or "
    "MultiPartInputFile over an in-memory stream) and returns 0. "
    "Output only one ```cpp code block."
)
ids = tok.apply_chat_template([{"role": "user", "content": PROMPT}], return_tensors="pt",
                              return_dict=True, add_generation_prompt=True)
pre = tok("```cpp\n", return_tensors="pt", add_special_tokens=False).input_ids
X0 = torch.cat([ids.input_ids, pre], 1).cuda()
AM = torch.ones_like(X0)
PL = X0.shape[1]
NEW, STEPS = 384, 192


def gen(seed, init_x=None, steps=STEPS):
    """Returns (x_final, commit_step[pos], commit_conf[pos])."""
    torch.manual_seed(seed)
    total = PL + NEW
    commit_step = [-1] * total
    commit_conf = [0.0] * total
    state = {"prev": None}

    def tokens_hook(step, x, logits):
        if step is None:
            if init_x is not None:
                x = init_x.clone()
            state["prev"] = (x == MASK)
            return x
        cur = (x == MASK)
        newly = state["prev"] & ~cur
        if newly.any():
            probs = torch.softmax(logits.float(), -1)
            for pos in newly[0].nonzero().flatten().tolist():
                commit_step[pos] = step
                commit_conf[pos] = float(probs[0, pos, x[0, pos]])
        state["prev"] = cur
        return x

    out = m.diffusion_generate(X0, attention_mask=AM, max_new_tokens=NEW, steps=steps,
                               temperature=0.5, top_p=0.95, alg="entropy", alg_temp=0.,
                               return_dict_in_generate=True,
                               generation_tokens_hook_func=tokens_hook)
    return out.sequences, commit_step, commit_conf


def extract(x):
    """Decode generated tokens; return code text plus per-line token spans."""
    gen_ids = x[0, PL:].tolist()
    pieces = [tok.decode([t], skip_special_tokens=False) for t in gen_ids]
    text, offs = "", []
    for p in pieces:
        offs.append((len(text), len(text) + len(p)))
        text += p
    end = len(text)
    for stop in ("```", "<|"):
        i = text.find(stop)
        if i != -1:
            end = min(end, i)
    if "LLVMFuzzerTestOneInput" not in text[:end]:
        return None
    return text, offs, 0, end


def lines_of(text, offs, cs, ce, conf):
    """Statement lines inside the code block with mean commit confidence."""
    res, pos = [], cs
    for line in text[cs:ce].split("\n"):
        s, e = pos, pos + len(line)
        pos = e + 1
        st = line.strip()
        if not (st.endswith(";") or st.startswith("#include")) or "LLVMFuzzerTestOneInput" in st:
            continue
        toks = [i for i, (a, b) in enumerate(offs) if a < e and b > s]
        if not toks:
            continue
        res.append({"text": st, "toks": toks,
                    "conf": sum(conf[PL + i] for i in toks) / len(toks)})
    return res


def code_of(x):
    ex = extract(x)
    return None if ex is None else ex[0][ex[2]:ex[3]]


variants = []
rng = random.Random(0)
for k in range(N):
    seed = 1000 + k
    x, cs, cc = gen(seed)
    ex = extract(x)
    rec = {"seed": seed, "full": code_of(x)}
    if ex is not None:
        text, offs, a, b = ex
        lines = lines_of(text, offs, a, b, cc)
        rec["n_lines"] = len(lines)
        rec["conf_spread"] = [round(l["conf"], 3) for l in lines]
        if len(lines) >= 2:
            low = min(lines, key=lambda l: l["conf"])
            rnd = rng.choice(lines)
            for arm, ln in (("lowconf", low), ("random", rnd)):
                init = x.clone()
                for i in ln["toks"]:
                    init[0, PL + i] = MASK
                x2, _, _ = gen(seed + 500, init_x=init, steps=max(4, len(ln["toks"])))
                rec[arm] = code_of(x2)
                rec[arm + "_line"] = ln["text"]
                rec[arm + "_changed"] = (rec[arm] != rec["full"])
    variants.append(rec)
    print(k, "ok" if rec["full"] else "no-code", rec.get("n_lines"), rec.get("conf_spread"), flush=True)
json.dump(variants, open(OUT, "w"), indent=1)
