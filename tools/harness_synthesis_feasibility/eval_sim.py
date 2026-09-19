import json, os, re, subprocess, shutil, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
IMG="dllm-openexr-runtime"; W=Path("work"); shutil.rmtree(W,ignore_errors=True); W.mkdir()
ref=Path("/home/jackb/workspace/dLLM_Agent/tools/openexr_runtime/poc_probe.cc").read_text()
seed=bytes.fromhex(Path("/home/jackb/workspace/dLLM_Agent/fixtures/openexr/c0-validity-preserving-zero.hex").read_text().strip())
jobs=[("reference",-1,ref)]
for i,r in enumerate(json.load(open("variants.json"))):
    for arm in ("full","lowconf","random"):
        if r.get(arm): jobs.append((arm,i,r[arm]))
uid=f"{os.getuid()}:{os.getgid()}"
def run(job):
    arm,i,code=job; d=(W/f"{arm}-{i}").resolve(); d.mkdir(); (d/"harness.cc").write_text(code); (d/"corpus").mkdir(); (d/"corpus"/"c0").write_bytes(seed)
    base=["docker","run","--rm","-u",uid,"-v",f"{d}:/work"]
    b=subprocess.run(base+[IMG,"/opt/dllm/bin/build-coverage"],capture_output=True,text=True,timeout=300)
    ok=(d/"coverage_target").exists()
    res={"arm":arm,"i":i,"compiled":ok,"cov":None}
    if not ok:
        errs=[l for l in b.stdout.splitlines()+b.stderr.splitlines() if "error:" in l]
        res["err"]=errs[0][-140:] if errs else "?"
        return res
    f=subprocess.run(base+["-w","/work",IMG,"/work/coverage_target","-max_total_time=20","-seed=1","-print_final_stats=0","/work/corpus"],capture_output=True,text=True,timeout=120)
    covs=re.findall(r"cov: (\d+)",f.stderr+f.stdout)
    res["cov"]=int(covs[-1]) if covs else None
    return res
with ThreadPoolExecutor(4) as ex: rs=list(ex.map(run,jobs))
json.dump(rs,open("eval.json","w"),indent=1)
for r in rs: print(r["arm"],r["i"],r["compiled"],r["cov"],r.get("err",""))
