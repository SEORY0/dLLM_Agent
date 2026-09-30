"""Compile-error feedback + knowledge-base arms for Dream-Coder harness synthesis (feasibility)."""
import json, os, re, subprocess, sys, tempfile, time
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer

SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
OUT = sys.argv[2] if len(sys.argv) > 2 else "fb_results.json"
ROUNDS, NEW = 3, 384
import difflib
IMG = "dllm-openexr-runtime"
P = "/data/dllm-agent/cache-dream-coder"
tok = AutoTokenizer.from_pretrained(P, trust_remote_code=True)
m = AutoModel.from_pretrained(P, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
MASK = m.config.mask_token_id
BASE = (
    "Write a complete libFuzzer harness in C++ for the OpenEXR library (v3.4). "
    "Define extern \"C\" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) that feeds the "
    "fuzz bytes into OpenEXR parsing APIs and returns 0. Output only one ```cpp code block."
)


def sh(*a, timeout=120):
    return subprocess.run(a, capture_output=True, text=True, timeout=timeout)


# ---- static knowledge base, extracted once from the pinned image ------------------------------
INC = "/opt/openexr-coverage/include"
hdrs = sh("docker", "run", "--rm", IMG, "sh", "-c",
          f"ls {INC}/OpenEXR | grep -E '^Imf.*\\.h$' | sort").stdout.split()
check_h = sh("docker", "run", "--rm", IMG, "cat", f"{INC}/OpenEXR/ImfCheckFile.h").stdout
decl = re.findall(r"(IMF_EXPORT\s+bool\s+checkOpenEXRFile\([^;]*?\);)", check_h, re.S)
SHORT = [h for h in hdrs if re.search(r"Check|^ImfInputFile|MultiPartInputFile|^ImfIO|^ImfNamespace|^ImfHeader", h)]
STATIC_KB = (
    "Reference facts about the installed OpenEXR (use only these):\n"
    "Relevant headers: " + ", ".join(SHORT) + "\n"
    "The namespace is OPENEXR_IMF_NAMESPACE (from ImfNamespace.h); there is no 'OpenEXR' namespace and no "
    "in-memory stream class. To parse a byte buffer call:\n"
    + "\n".join(re.sub(r"\s+", " ", d) for d in decl[:2])
    + "\nKeep the harness short: include only the headers you use."
)


def hint_for(msg):
    """Rule-based KB lookup: map a compiler error to a fix hint from the real header index."""
    out = []
    for h in re.findall(r"'([^']+\.h)' file not found", msg):
        base = h.split("/")[-1]
        near = difflib.get_close_matches(base, hdrs, n=3, cutoff=0.5)
        out.append(f"'{h}' does not exist." + (f" Existing headers with similar names: {', '.join(near)}." if near else ""))
    return " ".join(out)


def build_prompt(kb_text="", prev_code="", err="", hint=""):
    body = BASE
    if kb_text:
        body += "\n\n" + kb_text
    if err:
        body += ("\n\nYour previous attempt failed to compile.\nPrevious code:\n```cpp\n"
                 + prev_code + "```\nCompiler error:\n" + err
                 + ("\nHint: " + hint if hint else "") + "\nFix it.")
    ids = tok.apply_chat_template([{"role": "user", "content": body}], return_tensors="pt",
                                  return_dict=True, add_generation_prompt=True).input_ids
    pre = tok("```cpp\n", return_tensors="pt", add_special_tokens=False).input_ids
    return torch.cat([ids, pre], 1).cuda()


def gen(prompt, seed, init_gen=None, steps=192):
    torch.manual_seed(seed)
    PL = prompt.shape[1]

    def hook(step, x, logits):
        if step is None and init_gen is not None:
            x = torch.cat([prompt, init_gen], 1)
        return x

    out = m.diffusion_generate(prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=NEW,
                               steps=steps, temperature=0.5, top_p=0.95, alg="entropy", alg_temp=0.,
                               return_dict_in_generate=True, generation_tokens_hook_func=hook)
    return out.sequences[:, PL:]


def decode(g):
    pieces = [tok.decode([t], skip_special_tokens=False) for t in g[0].tolist()]
    text, offs = "", []
    for p in pieces:
        offs.append(len(text)); text += p
    end = len(text)
    for stop in ("```", "<|"):
        i = text.find(stop)
        if i != -1:
            end = min(end, i)
    return text[:end], offs


FLAGS = ["-std=c++17", "-fsyntax-only", "-isystem", INC, "-isystem", INC + "/OpenEXR",
         "-isystem", INC + "/Imath"]


def syntax_check(code):
    with tempfile.TemporaryDirectory(dir=".") as d:
        d = Path(d).resolve(); (d / "harness.cc").write_text(code); os.chmod(d, 0o777)
        r = sh("docker", "run", "--rm", "-u", f"{os.getuid()}:{os.getgid()}", "-v", f"{d}:/work", IMG,
               "clang++", *FLAGS, "/work/harness.cc", timeout=120)
    if r.returncode == 0:
        return True, "", []
    errs = [l for l in r.stderr.splitlines() if re.search(r"harness\.cc:\d+:\d+: (fatal )?error:", l)]
    lines = [int(re.search(r"harness\.cc:(\d+):", l).group(1)) for l in errs]
    msg = "\n".join(e.replace("/work/", "") for e in errs[:3])
    return False, msg, lines


def has_entry(code):
    return "LLVMFuzzerTestOneInput" in code


# ---- arms ------------------------------------------------------------------------------------
shared = {"bad_headers": set(), "verified": None}


USAGE = (
    "\nVerified usage example (calls that are known to compile against this OpenEXR):\n```cpp\n"
    "#include <ImfCheckFile.h>\n#include <ImfNamespace.h>\n"
    "const char* bytes = reinterpret_cast<const char*>(data);\n"
    "(void)OPENEXR_IMF_NAMESPACE::checkOpenEXRFile(bytes, size, true, true, false);\n```"
)


def kb_for(arm):
    if arm in ("kb", "kb_shared", "kb_usage"):
        t = STATIC_KB + (USAGE if arm == "kb_usage" else "")
        if arm == "kb_shared":
            if shared["bad_headers"]:
                t += "\nHeaders known NOT to exist: " + ", ".join(sorted(shared["bad_headers"]))
            if shared["verified"]:
                t += "\nA verified compiling harness for reference:\n```cpp\n" + shared["verified"] + "```"
        return t
    return ""


def run(arm, seed):
    kb = kb_for(arm)
    prompt = build_prompt(kb)
    g = gen(prompt, seed)
    hist = []
    for r in range(ROUNDS + 1):
        code, offs = decode(g)
        ok, msg, lines = syntax_check(code) if has_entry(code) else (False, "missing LLVMFuzzerTestOneInput", [])
        hist.append({"round": r, "ok": ok, "err": msg[:200]})
        if arm == "kb_shared":
            for h in re.findall(r"'([^']+\.h)' file not found", msg):
                shared["bad_headers"].add(h)
        if ok:
            if arm == "kb_shared":
                shared["verified"] = code
            return {"arm": arm, "seed": seed, "ok": True, "rounds": r, "code": code, "hist": hist}
        if r == ROUNDS:
            break
        if arm == "resample":
            g = gen(build_prompt(kb, code, msg), seed * 7 + r + 1)
            continue
        hint = hint_for(msg) if arm in ("kb", "kb_shared", "kb_usage") else ""
        # remask from the first failing line to the end (fixed-span infill cannot resize a line)
        first = min(lines) if lines else 1
        src = code.split("\n")
        start = sum(len(l) + 1 for l in src[: first - 1])
        keep = sum(1 for o in offs if o < start)
        init = g.clone(); init[0, keep:] = MASK
        g = gen(build_prompt(kb, code, msg, hint), seed * 7 + r + 1, init_gen=init,
                steps=max(8, (NEW - keep) // 2))
    return {"arm": arm, "seed": seed, "ok": False, "rounds": ROUNDS, "code": code, "hist": hist}


results = []
t0 = time.time()
for seed in range(SEEDS):
    for arm in os.environ.get("ARMS", "resample,feedback,kb,kb_shared").split(","):
        res = run(arm, 2000 + seed)
        results.append(res)
        print(f"{time.time()-t0:6.0f}s seed={seed} {arm:10} ok={res['ok']} rounds={res['rounds']} "
              f"{[h['err'][-60:] for h in res['hist'][-1:] if not res['ok']]}", flush=True)
        json.dump({"results": results, "kb_chars": len(STATIC_KB), "bad": sorted(shared["bad_headers"])},
                  open(OUT, "w"), indent=1, default=str)
