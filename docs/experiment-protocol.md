# 실험 프로토콜 및 사전등록 부록

**프로젝트:** dLLM vulnerability-agent · **버전:** 1.0 · **기준일:** 2026-08-20
**상태:** canonical OpenEXR PoC pilot 완료 · primary efficacy verdict는 no-go

이 부록은 `research-report.md`의 기제 우선 가설을 실행 가능한 manifest·schema·gate로 번역한다. empirical dLLM efficacy 논문은 아직 아니지만, trusted fixture QA에 더해 pin된 `jsoncons`에서 실제 dLLM 후보 생성과 격리 퍼징을 수행했다. 여기의 임계값은 사전 판정 규칙이지 관측 결과가 아니다. `[MEASURED]`는 현 host·fixture·live engineering run의 관찰, `[INFERRED]`는 설계 판단, `[FUTURE]`는 아직 수행하지 않은 실험을 뜻한다.

**2026-08-20 update.** 이 문서는 Stable-DiffCoder×jsoncons 경로, OpenEXR bootstrap
baseline, native-controller bring-up, 그리고 feedback-conditioned OpenEXR PoC revision의
canonical 5-arm×5-rep pilot을 포함한다. PoC revision pilot의 primary analyzer verdict는
`NO-GO`다. 모든 arm의 `validated_success`가 false였고 sanitizer signal이 없어 proof는
skipped였다. exploratory coverage 차이는 hypothesis-generating evidence일 뿐 취약점 발견,
zero-day 또는 dLLM 우월성 근거가 아니다.

## A. 고정 pin과 환경 gate

| 항목 | pin/값 | 용도 | 상태 |
|---|---|---|---|
| dLLM | `ByteDance-Seed/Stable-DiffCoder-8B-Instruct@10cdaf9b486f1cf0273ad968459dbe2d21a1482a` | 주 checkpoint/tokenizer | MEASURED engineering |
| Stable source | Hub `modeling_stable_diffcoder.py` SHA-256 `298b19…3104` | sampler audit | MEASURED |
| fallback | `Dream-Coder@79d43878c55ba4e7474d5e0b6057d110b43acfcd` | Stable gate 실패 시 | FUTURE |
| PrimeVul code | `DLVulDet/PrimeVul@6f54687c84947b1d17486495440b37030d147289` | Phase 1 | FUTURE |
| CyberGym-E2E | `sunblaze-ucb/cybergym-e2e@b861317f11641b14ab6ba08b5179d0b044601057` | Phase 2 smoke | FUTURE |
| runtime | Python 3.12.13, Torch 2.5.1+cu121, Transformers 5.3.0 | Stable load/live sample | MEASURED |

**[MEASURED]** 초기 조사에서는 free disk 17 GiB와 미설치 runtime이 blocker였으나, 정리 후 root 131 GiB free와 `/data` 1.4 TiB free가 확인됐다. 별도 Python 3.12.13 venv에 Torch 2.5.1+cu121과 Transformers 5.3.0을 설치하고 네 BF16 shard의 SHA-256을 검증했다. RTX 3090 `cuda:0` load는 allocated 16,500,925,440 bytes, reserved 16,636,706,816 bytes였으며 후보 두 개를 실제 생성했다. 8개 GPU는 download/load 가속보다 다수 후보의 data-parallel sampling에 적합하다. 이 측정은 engineering feasibility gate 통과이며 model efficacy gate 통과가 아니다.

가중치 license gate는 별도다. **[MEASURED]** Hub metadata는 MIT를 보이지만 해당 revision의 file tree에는 LICENSE artifact가 확인되지 않았고, GitHub source LICENSE는 weights grant의 증거가 아니다.[1], [3] **[FUTURE]** publisher의 exact-weight license artifact 또는 명시 statement를 저장하기 전에는 checkpoint 재배포/상용 이용을 하지 않으며, 연구용 local evaluation도 기관 정책 검토를 거친다.

### A.1 실행 중단 조건

다음 하나라도 발생하면 cell을 중단하고 재시작 전 manifest version을 올린다.

- model import/load/OOM 실패, vendor quicksort smoke 또는 empty-output regression smoke 실패
- hash-audited custom model source와 offline loading trust boundary, exact checkpoint weight-license decision 중 하나가 없을 때
- JSON classification smoke validity <90%, realised NFE가 설정 예산과 모순, model/tokenizer revision 불일치
- dataset SHA-256·schema·pair ID가 누락 또는 test label을 보며 parameter/feature를 수정
- FLOP parity가 ±5% 밖, latency parity가 ±10% 밖인데 해당 regime으로 보고하려는 경우
- analyzer hash/span/hint, candidate hash, validation seed, oracle image digest 중 하나라도 arm 간 불일치
- forbidden gold field가 prompt/retrieval에 발견, out-of-span edit, resource accounting 누락

### A.2 현재 validator와 third-party target 안전 gate

**[MEASURED]** 기존 `scan`은 계속 trusted repository-owned CWE-121 fixture만 host에서 실행한다. 별도 `hunt` 경로는 manifest에 고정된 repository URL, clean commit, tree/archive hash와 `SECURITY.md` hash를 검증하고, 해당 commit의 `git archive`만 UID/GID 65534가 읽을 수 있게 staging한다. Digest-pinned Docker는 `--network none`, read-only root, `--cap-drop ALL`, no-new-privileges, PID/CPU/memory/tmpfs 제한, source read-only mount와 `--pull never`를 강제한다. 실제 `jsoncons` header를 이 경계에서 비특권 전처리하고 libFuzzer/coverage까지 실행했다. 일반 임의 target은 같은 manifest·pin·disclosure gate 없이는 허용하지 않는다.

## B. canonical manifest와 event schema

모든 run은 append-only `run_manifest.json`으로 시작한다. byte string은 SHA-256, timestamp는 RFC 3339 UTC, source code는 raw blob와 canonicalized blob 모두 hash한다.

```json
{
  "schema_version": "1.0",
  "run_id": "uuid", "created_at": "RFC3339Z",
  "git_commit": "40hex", "container_digest": "sha256:...",
  "model": {"id": "...", "revision": "40hex", "tokenizer_revision": "40hex", "license_status": "unresolved|cleared"},
  "sampler": {"name": "stable_block", "requested_steps": 256, "gen_length": 512,
              "block_length": 8, "temperature": 0.0, "order": "low_confidence", "eos_id": null},
  "target": {"dataset": "primevul", "artifact_sha256": "...", "task_id": "...",
             "repo": "...", "commit": "...", "timestamp": "...", "cwe": ["CWE-..."], "split": "test"},
  "cell": {"backbone": "stable|ar_parent|strong_ar", "scaffold": "minimal|validate",
           "budget_regime": "equal_flop|practical_time|practical_dollar", "information_level": "I0"},
  "seeds": {"model": 0, "retrieval": 0, "validator": 0, "permutation": 0},
  "policy": {"target_trust": "owned_fixture|third_party", "tool_side_effects_after_finalization": true,
             "forbidden_gold_scan": true, "isolation_evidence_sha256": "required-for-third-party"}
}
```

`target` schema는 `source_dataset`, `raw_row_sha256`, `repo`, `commit`, `commit_time`, `CVE`, `CWE`, `before_code`, `after_code`, `path`, `changed_lines`, `changed_statements`, `license_flag`, `provenance_flag`, `raw_hash`, `normalized_lexer_hash`, `ast_shape_hash`, `minhash`를 포함한다. 누락된 provenance identity는 추정하지 않고 exclusion reason으로 기록한다.

`trajectory_event`는 다음의 완전한 batch 구조다. Stable의 stock sampler에서는 `commit_bitmap`만 token을 확정하며 `remask_bitmap`은 기본적으로 empty여야 한다. committed token을 되돌리는 event가 생기면 stock-Stable claim이 아니라 new-sampler treatment로 분기한다.[2], [13]

```json
{
  "run_id": "uuid", "candidate_id": "c0", "event_index": 12,
  "nfe": 12, "timestep": 0.52, "block": 3,
  "mask_bitmap_sha256": "...", "commit_bitmap_sha256": "...",
  "remask_bitmap_sha256": "...", "replace_bitmap_sha256": "...",
  "position_updates": [{"position": 41, "token_id": 17, "prob": 0.82, "entropy": 0.61, "margin": 0.58}],
  "state_sha256": "...", "latency_ms": 0, "peak_memory_bytes": 0
}
```

`generation_batch`에는 `candidate_id`, `final_text`, `output_sha256`, `parse_status`, `typed_action`, `requested_steps`, `realised_nfe`, `input_tokens`, `output_tokens`, `forward_calls`, `wall_ms`, `gpu_seconds`, `peak_memory_bytes`, `validator_calls`, `dollar_estimate`를 기록한다. intermediate trajectory는 tool input이 아니며, `typed_action`은 완성 candidate에서 schema parse와 policy check를 통과한 뒤에만 실행한다.

**구현 경계.** `ReplaySampler → trajectory score → selection → sanitizer → hashed JSON`은 모델에 무관한 최소 재현 spine이다. `ReplaySampler`는 test double이며 oracle/evidence hash는 AR과 dLLM 양쪽에 동일하게 적용한다. **[MEASURED]** live 경로는 실제 Stable-DiffCoder token-transfer를 계측하는 `DiffusionSampler`, candidate-specific ranking과 realised NFE를 기록한다. Stock Stable의 finalized sample 후에만 typed compile/fuzz action을 실행한다. 여러 후보에 validator 시간을 동적으로 나누는 step-aware budget router와 AR 대조는 아직 **[FUTURE]**다.[2], [13]

`validation_evidence`에는 `target_sha`, `tool_name/version/hash`, `container_digest`, `command_template_hash`, `network_off`, `timeout_ms`, `memory_limit`, `exit_code`, `stdout/stderr_normalized_hash`, `oracle_verdict`, `finding_location`, `cause`, `reproducer_hash`, `evidence_hash`를 기록한다. `proved`는 non-timeout oracle success일 때만 가능하다.

## C. 데이터, split, 중복 제거

### C.1 Phase 1: PrimeVul 및 SecVulEval

**[FUTURE]** PrimeVul은 chronological C/C++ before/patched pair로 주 endpoint를 구성한다.[4], [5] 먼저 raw artifact의 immutable URL, SHA-256, row count, schema dump를 기록하고, 그 뒤에만 fixed 4-pair smoke와 32-pair mini-study를 sampling한다. full study의 목표는 hash·schema audit 뒤 simulation-derived sample size와 multiplicity plan을 먼저 고정한 후 정하며, 숫자는 실제 artifact의 valid component 수보다 우선하지 않는다.

**[FUTURE]** SecVulEval은 CVE/project/time grouped holdout을 새로 만든 statement-localization 교차검증이다. official split을 그대로 headline result로 쓰지 않고, function/pair label과 patch-diff line proxy를 명시적으로 분리한다.[6] 최소 600 valid function은 목표이지 확보됐다는 주장이 아니다.

Canonicalization 절차:

1. pinned C/C++ lexer로 raw, identifier/literal-normalized tokens, AST shape를 만든다.
2. exact hash 및 MinHash prefilter 후 shingle Jaccard와 normalized-token Levenshtein로 near clone을 확인한다.
3. exact/near clone, shared `repo+commit`, shared CVE, before/after link를 undirected union-find component로 합친다.
4. component 단위로 temporal/project-disjoint train/dev/test를 배정한다. component가 split을 넘으면 모두 재배정 또는 제외한다.
5. missing source/time/commit, parse failure, cross-label duplicate, ambiguous provenance는 reason code로 제외한다.

이 규칙은 temporal split만으로 clone·provenance leakage를 해소하지 못하며 vulnerability benchmark의 data quality가 측정치를 크게 바꿀 수 있다는 audit 결과에 대응한다.[7] 세 report tier는 `original_replication`, `strict_project_temporal`, `model_specific_post_public_label`이다. 세 번째는 model이 source code를 보지 않았음을 뜻하지 않는다.

### C.2 Phase 2: VulnGym 및 CyberGym-E2E

**[FUTURE]** VulnGym은 repository-level localization/validation cross-check로 사용하며 official evaluator가 제공하지 않는 precision/evidence field를 본 protocol으로 확장한다.[8] discovery, localization/reproduction, repair는 동일 분모로 합산하지 않는다.

**[FUTURE]** CyberGym-E2E는 codebase-only input과 S1–S4 validation을 가진 overall capability 평가다.[9], [14] `b861…1057`에서 서로 다른 project 10개를 replacement 없이 예약한다. 이 subset은 licensing, gating, 대용량 storage, container/host privilege 및 A.2 isolation gate를 통과하기 전에는 실행하지 않으며, 실행돼도 **smoke-only**다. n=10 성공률/유의확률/모델 순위를 논문 근거로 사용하지 않는다.

Conditional mechanism arm은 전체 E2E와 분리한다. 동일 successful S1 scout, bounded source windows, one-hop call graph, stack frames, build/run/test scripts를 `retrieval_pack_sha256`로 freeze한다. AR-FIM과 dLLM은 동일 context/candidate/time/output budget을 받으며 evaluator feedback은 금지한다. `S1_shared`, `S2_shared`, S3, S4를 별도로 보고한다.

## D. model×scaffold×budget cell

| 축 | level | 고정 요소 |
|---|---|---|
| backbone | Stable dLLM, AR parent, strong independent AR | model revision, decode policy별 manifest |
| scaffold | minimal common, security-validation | tool access, retry, context, prompt contract |
| budget | equal-FLOP, practical-time, practical-dollar | 다른 축은 log 후 보정하지 않음 |
| information | I0 repo-only, I1 coarse CWE, I2 sanitized behavior, I3 diagnostic | rendered prompt gold scan |
| task | discrimination, localization, reproduction, repair, discovery | endpoint와 denominator 분리 |

`minimal common`은 read-only source/retrieval, final typed action, 동일 candidate count만 갖는다. `security-validation`은 같은 candidate budget에 static analyzer/compiler/sanitizer/PoC oracle 및 evidence ledger를 추가한다. scaffold가 backbone보다 더 큰 효과를 낼 수 있으므로 이 교차를 생략하지 않는다.[10]

퍼징 endpoint에는 backbone 비교와 별도로 다음 generator/scaffold baseline을 둔다.

| baseline | 고정 계약 | 비교 목적 |
| --- | --- | --- |
| 공식 또는 human-written harness + stock libFuzzer | 같은 target pin, corpus, sanitizer, wall-clock | 모델 없는 최저선 |
| `oss-fuzz-gen` 계열 strong AR harness agent | 같은 source/context, 후보 수, compile/fuzz 예산 | 실용적인 LLM harness 기준선 |
| agent-synthesized structured generator | 같은 target oracle, generator build·run 비용 포함 | 구조 인식 효과와 model family 분리 |
| Stable-DiffCoder harness/tournament | 같은 후보·validator budget, trajectory real/pseudo 분리 | dLLM trace allocation 효과 |
| AR-FIM/structured local edit | 같은 field/span, feedback, revision 횟수 | selective revision의 AR 대조군 |
| compatible dLLM T2T 또는 T2M | 같은 model 가능 시 decoder policy만 변경 | committed-token replace와 remask 기제 분리 |

`oss-fuzz-gen`, Gentoo류 generator agent와 FuzzingBrain V2는 최신 agentic fuzzing 비교축이지만
각 논문의 reported score를 본 프로젝트의 기대값으로 재사용하지 않는다.[15]–[17] LLaDA2.1
T2T와 T2M도 일반 benchmark에서 제안된 decoder이므로 sanitizer-guided field revision의
성공을 전제하지 않는다.[18], [19]

각 resource record는 parameter 수 대신 `input/output_tokens`, `forward_calls`, `realised_nfe`, FLOP estimate와 method, GPU model/count, wall clock, GPU seconds, peak memory, validator/retry time, storage/network cost, dollar estimate와 tariff date를 기록한다. equal-FLOP cell은 realized measure ±5%, equal-latency cell은 ±10%를 gate로 한다. practical-time/dollar cell은 parity를 강제하지 않고 Pareto frontier를 보고한다.

## E. candidate-bank trace permutation control

이 control은 H1의 주 식별 설계다.

1. 동일 `task_id`, prompt, model seed, sampling config로 real Stable candidate 4개를 생성하고 trajectory와 final hash를 freeze한다.
2. candidate별 preregistered trace feature vector를 계산한다. feature code와 scaling/calibration은 dev split에서 lock한다.
3. **real arm:** feature score로 한 validation slot을 배정한다.
4. **pseudo arm:** candidate proposal/validator/cost를 그대로 두고, task 내 no-fixed-point deterministic derangement로 trace score를 다른 candidate에 붙인다. trace score distribution은 보존한다.
5. 각 arm은 abstention 없이 한 후보를 반드시 동일 isolated oracle에 1회 제출한다. `validated_success@1`은 그 선택 후보가 dynamic oracle로 `proved`이면 1, 아니면 0인 task-level indicator다. 따라서 primary-arm coverage는 100%다.

real−pseudo의 paired mean/proportion difference가 점추정치 ≥+5점 및 paired lower 95% CI >0일 때만 H1을 지지한다. upper 95% CI <+5점이면 meaningful H1 benefit을 기각한다. 이 비교는 trace가 fixed validation allocation에 유용한지 검정할 뿐 dLLM 대 AR 우월성을 검정하지 않으며, 후자는 D절의 crossed study에서만 판정한다. 순열은 block/mask stratum을 보존하고 candidate ID·final output·validator seed를 바꾸지 않는다.

**[MEASURED]** 현재 `rotate_trace_for_qa`/`synthetic_trace_rotation`은 synthetic fixture에서 한 trace의 문자열을 회전해 내보내는 QA field다. candidate-to-candidate score derangement가 아니므로 H1 pseudo arm 또는 mechanism evidence가 아니다.

**[MEASURED]** OpenEXR PoC revision canonical pilot은 5 arms×5 reps로 실행됐다.
artifact root는
`/home/seory0/.cache/dllm-agent/openexr-poc-5arm-pilot-canonical-20260820-183739`이며
top-level hashes는 `raw-arm-reports.jsonl`
`8204d8a07cc778e6f6810f6a69d2d68ee16da362b78322d9b4c191dcf1b633ae`,
`paired-summary.json` `f4e35bbf97f08679fc49e43bb8001abecc88575904306674b9ed40cfb844d2d7`,
`live-index.json` `f0b6c36f314c000c12174c0214f56e31ea71baf7c07a678f2b7c148dfd719934`이다.
Primary `validated_success@1`는 모든 arm에서 false였고 paired summary는 `NO-GO`다.
sanitizer signal은 false였으며 proof는 skipped였으므로 취약점 또는 zero-day를 발견하지
않았다. Exploratory mean `new_edges`는 `libfuzzer-only` 1640.8,
`structured-one-shot` 1606.2, `feedback-no-remask` 1610.0, `selective-revision` 1626.0,
`pseudo-feedback` 1614.4였다. `selective-revision - pseudo-feedback`의 exploratory
차이는 +11.6, CI `[-1.6, 25.2]`로 inconclusive다. selective와 pseudo field action은
3/5 repetitions에서 diverge했고, pseudo arm의 field attribution은 1/5 repetitions에서만
발생했다. deterministic successor selector는 confidence와 trajectory evidence를 기록하지만
semantic selector value를 sampled token identity가 아니라 catalog successor class로 commit하는
pilot caveat가 있다.

## F. intervention 및 control 사양

### F.1 PrimeVul trace localization

**[FUTURE]** before/patched pair마다 label-blind aligned mask view 8개를 ratio `0.15/0.30/0.50/0.70`×fixed seed 2개로 생성한다. development에서만 feature selector와 calibrator를 fitting한다. line aggregation은 normalised mean/max/percentile이며 raw token sum은 사용하지 않는다.

필수 control: (a) 동일 aggregation AR token NLL, (b) equal-forward-call AR self-review, (c) final dLLM confidence, (d) commit time을 지운 non-temporal trace, (e) lexical rarity/length/static-analysis, (f) random label, (g) pair swap. metric은 P-C, AUROC/AUPRC, Top-k/MRR, ECE, Brier skill이다. H2는 strongest equal-compute AR 대비 P-C 점추정치와 lower 95% CI가 모두 +5점 초과일 때만 지지한다. chronology test ECE ≤0.05와 positive Brier skill이 없으면 calibrated probability로 표현하지 않는다.

### F.2 analyzer-guided local replacement

모든 controller는 같은 insecure source, analyzer witness/hint, smallest enclosing statement, 32-lexer-token edit cap, before/after analyzer 1회, compile/security oracle를 받는다. arm은 `no_repair`, `dLLM_detached_local`, `AR_FIM_local`, `AR_structured_local`, `AR_whole_program`, `random_equal_span`이다. CDC/Dream MDFI는 compatible true-revision sampler가 별도 구현됐을 때만 추가한다; Stable stock config의 `remasking`으로 표기하지 않는다.[2], [11]

primary outcome은 `secure_functional@1`: build/test function과 security oracle 모두 성공한 candidate의 비율이다. lower paired CI가 best AR local보다 +8점 초과일 때만 meaningful benefit을 지지한다. upper CI가 +8점 미만이거나 AR가 accuracy-cost Pareto를 지배하면 benefit을 기각한다. source span 밖 edit는 그 trial을 invalid로 하고 해당 cell의 원인을 조사한다.

### F.3 locality moderation

locality label은 L0 statement/basic block, L1 intraprocedural, L2 interprocedural same file, L3 cross-file structural, L4 stateful/semantic invariant다. 여러 label이면 `L4>L3>L2>L1>L0`을 적용한다. 주 비교는 L0–L1 대 L2–L4이며, expert annotation agreement와 adjudication log를 낸다. H3는 nonlocal dLLM advantage ≥8점 및 family×nonlocality DID ≥5점, 두 lower 95% CI >0을 요구한다.

## G. metric, 통계, power, seed

| endpoint | 주 metric | 보조 | unit |
|---|---|---|---|
| pair discrimination | P-C, PR-AUC, MCC, VD-score | AUROC, ECE/Brier | before/patch pair |
| localization | Top-k, MRR, F1 | line calibration | function/statement |
| trace prioritization (H1) | validated_success@1 | cost per proved finding | task; one obligatory selection per arm |
| reproduction | isolated-oracle/PoC success | time-to-proof | task |
| repair | secure_functional@1 | edit locality, compile rate | insecure candidate |
| discovery | validated_discovery@1 | S1–S4 | advisory/repository |

`validated_discovery@1`은 location/cause agreement와 isolated oracle/PoC를 모두 요구한다. I2/I3에서만 gain이 나면 guided localization이라고 보고 discovery로 쓰지 않는다. **[FUTURE]** MVP seed는 3, powered study seed는 5이며 model/retrieval/validator/permutation seed를 별도 manifest field로 둔다. task/advisory가 unit이고 repository clustering은 block bootstrap과 random effect로 반영한다. H1의 paired binary outcome에는 paired proportion CI와 McNemar를 함께 보고하며, continuous/ranked outcome에는 cluster bootstrap CI 및 preregistered hierarchical model, multiple family에는 simulation-derived multiplicity plan의 Holm correction을 쓴다.

Power는 H1/H2의 5점, H3의 8점/5점, H4의 8점 SESOI와 2.5/5/10점 sensitivity, paired correlation, repository ICC, expected exclusion/dropout을 hash-locked pilot에서 lock한 뒤 simulation으로 산정한다. power를 위해 label을 보며 task를 추가/제거하지 않는다. official artifact hash, valid component count, simulation-derived sample size, and multiplicity plan이 없으므로 현재 numerical power/required-n은 의도적으로 미기입이다.

## H. inclusion/exclusion 및 판정 gate

**포함:** hash-pinned code/artifact, legal access confirmed, provenance complete, A.2 isolation gate를 통과한 buildable oracle, prompt gold scan pass, resource event complete task. 현재 예외는 repository-owned fixture이며, 실제 OSS `hunt`는 manifest의 commit·origin·security-policy·container pin을 모두 충족해야 한다.

**제외:** source/commit/time identity missing, cross-split component, parser failure, label conflict, unknown license/access restriction, leaked gold field, validator instability, timeout definition violation. 제외는 arm별로 같은 이유 code/row hash와 함께 공개한다. performance가 나쁜 task만 사후 배제할 수 없다.

| 주장 | 지지 gate | reject / no-go |
|---|---|---|
| H1 trace priority | `validated_success@1` coverage=100%, abstention=0, real−pseudo point estimate ≥+5점, lower CI >0, cost invariant | permutation/validator mismatch 또는 upper CI <+5점 |
| H2 dLLM mechanism | strongest equal-compute AR 대비 P-C point estimate와 lower CI 모두 >+5점 | AR equal-compute Pareto dominance 또는 lower CI ≤+5점 |
| H3 locality | nonlocal advantage ≥+8점 및 family×nonlocality DID ≥+5점, 두 lower CI >0 | 어느 lower CI라도 ≤0 |
| calibrated signal | ECE ≤.05 + positive Brier skill | ordinal-only reporting |
| local repair | +8점 lower CI over AR local | upper CI <+8 또는 out-of-span edit |
| broad security result | executable evidence, temporal/project split | prose judge, leaked task, random split only |
| speed claim | ≥20% wall-time, ≤2점 noninferiority loss | unreported validator/retry time |

Pilot quality gate는 30–50 recent project-disjoint executable-oracle cases+safe negatives이며, final sample size와 multiplicity는 pilot의 hash-locked 분산에서 simulation으로 정한다. Clean rerun task-level agreement는 ≥95%, aggregate shift는 ≤2점이어야 한다. 이 수치는 prespecified decision thresholds이며 현재 달성됐다는 뜻이 아니다.

## I. resource accounting과 재현 package

각 run은 hardware/model/driver/library/container hash, power/energy meter가 있으면 energy와 sampling interval, 없으면 absence를 기록한다. dollar는 model GPU lease·validator CPU·storage·retrieval을 분리하며 local hardware의 historical acquisition cost를 invention하지 않는다. candidate rejected by parser도 생성·validator 비용에서 제거하지 않는다.

release package에는 (1) environment lockfile/container digest, (2) all manifests/events, (3) model/dataset/repository SHA, (4) prompt/retrieval template와 forbidden-field scanner, (5) candidate/output/evidence hashes, (6) dedup components/exclusion ledger, (7) analysis code·seed·figure source, (8) raw/aggregate result and negative results, (9) disclosure redaction policy가 들어간다. secure PoC가 공개 위험이면 hash와 verifier protocol을 공개하고 payload는 controlled disclosure로 보관한다.

## J. future-only 항목

다음은 구현되어 있지 않거나 현 host gate 뒤의 일이다.

- 64/256/512 전 schedule 비교와 exact weight license decision. 실제 64/256 smoke, hash-audited custom source와 offline load는 완료됨
- 4-candidate candidate-to-candidate score-derangement H1 study. 실제 2-candidate trace 계측은 완료됨
- PrimeVul immutable data checksum/schema acquisition 및 cross-dataset component count
- simulation-derived sample size/multiplicity plan, powered study, expert locality labels
- powered analyzer-guided committed-token revision/MDFI study와 forced-order generation
  scheduler. 구조화 particle·schema·policy·ledger·probe·selective revision runtime과
  canonical OpenEXR pilot은 완료됐지만, primary 결과는 `NO-GO`이고 coverage signal은
  inconclusive임
- 일반화된 third-party manifest onboarding; VulnGym full evaluation, CyberGym shared-scout retrieval pack 및 source-only E2E
- exact Stable weight license clarification

**[MEASURED]** owned CWE-121 fixture, pin된 `jsoncons` live engineering target, OpenEXR 공식-harness baseline, native-controller bring-up, OpenEXR PoC revision canonical pilot을 검증했다. `jsoncons` live run은 두 UBSan-only zero-length copy 위반을 재현했고 ASan-only continuation은 121초·2,011,697 units에서 ASan crash를 관찰하지 않았다. OpenEXR native/static 단일 비교도 양쪽 sanitizer finding이 없었다. OpenEXR PoC revision pilot은 all-false `validated_success`, proof skipped, `NO-GO`였다. 나머지 fixture와 powered model-science 표는 future다.

## 참고문헌

[1] ByteDance-Seed, “Stable-DiffCoder-8B-Instruct model card,” revision `10cdaf9b486f1cf0273ad968459dbe2d21a1482a`. Available: https://huggingface.co/ByteDance-Seed/Stable-DiffCoder-8B-Instruct/tree/10cdaf9b486f1cf0273ad968459dbe2d21a1482a (accessed 2026-08-19).

[2] ByteDance-Seed, “modeling_stable_diffcoder.py,” revision `10cdaf9b486f1cf0273ad968459dbe2d21a1482a`. Available: https://huggingface.co/ByteDance-Seed/Stable-DiffCoder-8B-Instruct/blob/10cdaf9b486f1cf0273ad968459dbe2d21a1482a/modeling_stable_diffcoder.py#L46 (accessed 2026-08-19).

[3] ByteDance-Seed, “Stable-DiffCoder source LICENSE,” revision `ece1ffb526ee44fb6c9f212a8ef118d204cb9145`. Available: https://github.com/ByteDance-Seed/Stable-DiffCoder/blob/ece1ffb526ee44fb6c9f212a8ef118d204cb9145/LICENSE (accessed 2026-08-19).

[4] DLVulDet, “PrimeVul,” source repository. Available: https://github.com/DLVulDet/PrimeVul/tree/6f54687c84947b1d17486495440b37030d147289 (accessed 2026-08-19).

[5] Y. Ding *et al.*, “Vulnerability Detection with Code Language Models: How Far Are We?,” arXiv:2403.18624, 2024. Available: https://arxiv.org/abs/2403.18624 (accessed 2026-08-19).

[6] B. D. Basim, “SecVulEval,” source repository. Available: https://github.com/basimbd/secvuleval (accessed 2026-08-19).

[7] R. Croft *et al.*, “Data quality matters: a case study on vulnerability detection datasets,” ICSE 2023. Available: https://rolandcroft.github.io/assets/publications/ICSE_23.pdf (accessed 2026-08-19).

[8] Tencent, “VulnGym,” v0.1.4 source repository. Available: https://github.com/Tencent/VulnGym (accessed 2026-08-19).

[9] CyberGym, “CyberGym-E2E.” Available: https://www.cybergym.io/cybergym-e2e/ (accessed 2026-08-19).

[10] OpenAI, “Introducing SWE-bench Verified,” 2024. Available: https://openai.com/index/introducing-swe-bench-verified/ (accessed 2026-08-19).

[11] “CDC/MDFI: Analyzer-guided diffusion for secure code generation,” 2026. Available: https://arxiv.org/pdf/2605.16829 (accessed 2026-08-19).

[12] dLLM_Agent 연구 journal, “Wave 2: Model Feasibility and Reproduction Manifest,” local runtime inspection, 2026-08-19. Available: [wave-2-model-feasibility.md](../.omo/ulw-research/20260819-134557/wave-2-model-feasibility.md).

[13] ByteDance-Seed, “modeling_stable_diffcoder.py,” assignment/write loop, lines 211–223, revision `10cdaf9b486f1cf0273ad968459dbe2d21a1482a`. Available: https://huggingface.co/ByteDance-Seed/Stable-DiffCoder-8B-Instruct/blob/10cdaf9b486f1cf0273ad968459dbe2d21a1482a/modeling_stable_diffcoder.py#L211-L223 (accessed 2026-08-19).

[14] Sunblaze UCB, “CyberGym-E2E,” source repository, revision `b861317f11641b14ab6ba08b5179d0b044601057`. Available: https://github.com/sunblaze-ucb/cybergym-e2e/tree/b861317f11641b14ab6ba08b5179d0b044601057 (accessed 2026-08-19).

[15] Google, “oss-fuzz-gen: LLM powered fuzzing via OSS-Fuzz,” source repository. Available: https://github.com/google/oss-fuzz-gen (accessed 2026-08-20).

[16] V. Vikram and R. Padhye, “Fuzzing with Agents? Generators Are All You Need,” arXiv:2604.01442, 2026. Available: https://arxiv.org/abs/2604.01442 (accessed 2026-08-20).

[17] Z. Sheng *et al.*, “FuzzingBrain V2: A Multi-Agent LLM System for Automated Vulnerability Discovery and Reproduction,” arXiv:2605.21779, 2026. Available: https://arxiv.org/abs/2605.21779 (accessed 2026-08-20).

[18] T. Bie *et al.*, “LLaDA2.1: Speeding Up Text Diffusion via Token Editing,” arXiv:2602.08676, v3, 2026. Available: https://arxiv.org/abs/2602.08676 (accessed 2026-08-20).

[19] L. Yao, “Remask, Don't Replace: Token-to-Mask Refinement in Diffusion Large Language Models,” arXiv:2604.18738, v2, 2026. Available: https://arxiv.org/abs/2604.18738 (accessed 2026-08-20).
