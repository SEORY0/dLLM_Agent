# Diffusion Trajectory as an Observable Control Signal for a Fuzzing Agent — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:test-driven-development` for
> Tasks 1–7 (fake/protocol-backed unit tests before production code, following the
> `ar_slot_sampler.py` precedent) and `superpowers:executing-plans` or
> `superpowers:subagent-driven-development` to execute task-by-task. Steps use checkbox
> (`- [ ]`) syntax for tracking. This plan operationalizes
> `docs/superpowers/specs/2026-09-15-trajectory-guided-harness-synthesis.md` §11 step 3, now
> that §10's open risks have decisions (below) and a feasibility simulation has run on the real
> workstation GPU.

**Goal:** Measure whether a diffusion model's per-token commit-time trajectory (confidence,
entropy, margin) is a control signal a fuzzing agent can act on — first as correlation with
harness correctness (RQ1, `2026-09-15` spec §2), then as a remask-targeting policy that beats
random remasking under a matched compile-capable baseline (RQ3) — and separately check whether a
shared knowledge base with several verified usage examples produces harnesses that diverge from
each other and from the reference, rather than collapsing onto one example.

**Architecture:** New top-level package `src/dllm_agent/harness_synthesis/`, independent of the
`openexr_one_day` 50-cell campaign and the `revision/` pipeline (spec §4). Dream-Coder-7B runs in
its own pinned venv, invoked as a subprocess from the project venv (matching the
`runtime_python`/`--runtime-python` pattern this repo already uses for Stable-DiffCoder). The
target stays OpenEXR's `LLVMFuzzerTestOneInput`; compilation goes through a new volume-mounted
compile-only image, not the existing `dllm-openexr-runtime` fixed-harness image.

**Tech Stack:** Python 3.12, Dream-Coder-v0-Instruct-7B (transformers 4.46.2, its own venv), torch
2.11.0+cu128 (shared with the project venv), one RTX PRO 6000 Blackwell, Docker/clang 18 (ASan +
coverage), Pydantic v2, pytest.

**Design inputs:**
[`2026-09-15-trajectory-guided-harness-synthesis.md`](../specs/2026-09-15-trajectory-guided-harness-synthesis.md),
[`harness-synthesis-feasibility-20260920.md`](../../harness-synthesis-feasibility-20260920.md),
[`openexr-1gpu-smoke-and-harness-feasibility-20260920.md`](../../openexr-1gpu-smoke-and-harness-feasibility-20260920.md),
[`research-report.md`](../../research-report.md) §5 (claim tagging, power calc from observed
variance), `tools/harness_synthesis_feasibility/` (prototype scripts this plan formalizes).

## 1. 결론부터 — 팀 피드백으로 바뀐 프레이밍

기존 스펙(§2)의 RQ1/RQ2/RQ3는 그대로 유지한다. 바뀐 것은 **1차 주장의 위치**다.

- 이전 프레이밍: "모델이 harness를 스스로 합성한다"가 1차 목표였고, 타당성 시뮬레이션은
  이것이 예시 없이는 0/48이라는 부정적 결과를 냈다(`harness-synthesis-feasibility-20260920.md`
  Result 2).
- 팀원 피드백: 무작위 대비 confidence 국소화 결과(Result 1: 11/16 대 3/16)가 "꽤 잘 나왔다"는
  평가와 함께, **"diffusion generation trajectory를 fuzzing agent가 관찰 가능한
  uncertainty/control signal로 쓴다"**는 축으로 포커스를 옮기라는 지시.
- 이 계획의 1차 주장: trajectory 신호(confidence/entropy/margin)가 (a) harness correctness와
  상관관계가 있고(RQ1, 적정 표본수로 재측정) (b) 이 신호로 remask 위치를 고르면 무작위
  remask보다 나은가(RQ3)를 **컴파일이 실제로 되는 baseline 위에서** 측정한다. "모델이 스스로
  합성"은 부산물이지 1차 주장이 아니다.
- 이걸 가능하게 하려면 Result 3(검증된 사용 예시가 있어야 컴파일된다)을 받아들이고, 사용자가
  지시한 **a안**(여러 사용 예시를 가진 공유 KB가 서로 다르고 reference와도 다른 harness를
  만드는지 검증)을 RQ1/RQ3 이전에 배치해 baseline의 다양성 붕괴 문제를 먼저 확인한다.

## 2. §10 리스크에 대한 결정 (이미 확정됨)

| 리스크 | 결정 |
|---|---|
| harness identity 감사 체인 | **반복마다 harness SHA-256을 append-only JSONL 원장에 기록**(`qualification.py`의 evidence-tree 선례를 확장, Task 6) |
| Dream-Coder 실행 환경 | **프로젝트 venv와 분리된 별도 pinned venv**(transformers==4.46.2), torch는 프로젝트 venv를 재사용. 다만 지금은 세션 scratchpad에만 존재해 재현 불가 — **Task 1이 이를 저장소 파일로 고정**하는 것이 첫 번째 일 |
| Dream hook 삽입 지점 | `generation_tokens_hook_func(step, x, logits)` / `generation_logits_hook_func`로 확인됨(스크립트 검증 완료, `2026-09-20` 문서) |

## 3. NO-START 게이트

아래 네 가지는 Task 8(실제 GPU 실험) 착수 전에 반드시 끝나 있어야 한다:

1. **Dream 실행 환경이 scratchpad에만 있다.** `/tmp/.../scratchpad/dreamvenv`는 세션이 끝나면
   사라진다. Task 1에서 `tools/dream_runtime/requirements.txt` + 빌드 스크립트로 저장소에
   고정한다.
2. **GPU 경쟁.** 이 문서 작성 시점에 GPU 0·1 모두 다른 사용자(`gyu`)의
   `research_v8/train_*_xlsr_speech.py` 학습이 90%+ 사용률로 돌고 있다. Task 1–7(단위 테스트,
   KB 추출, compile-only 이미지 빌드)은 GPU가 필요 없어 그대로 진행 가능하지만, Task 8 이후는
   GPU 단독 사용이 필요하다 — **착수 전 사용자에게 GPU 가용 시간대를 확인**한다.
3. **compile-only 이미지가 없다.** `tools/openexr_runtime/Dockerfile`은 `poc_probe.cc`를
   `COPY`로 굽는다. Task 5가 볼륨 마운트 변형을 만들기 전까지 harness를 iteration마다 바꿔
   컴파일할 수 없다.
4. **표본수 미정.** Task 9/10을 시작하기 전, feasibility 시뮬레이션의 관측 분산(11/16 대
   3/16, 8/8 대 0/8×4)에서 power calculation을 돌려 RQ1/RQ3 각각의 필요 표본수를 정한다
   (`research-report.md` §5 원칙, 임의로 정하지 않음).

## 4. 모듈 설계 (spec §7 기준, 변경점만 표시)

```
src/dllm_agent/harness_synthesis/
├── __init__.py
├── token_trace.py         # Task 2 — TokenTrace + permanence_step()
├── trajectory_hook.py      # Task 3 — Dream 실행 hook 배선
├── regions.py              # Task 4 — statement 경계 탐지(정규식/괄호 매칭)
├── api_knowledge.py        # Task 1b — [신규, spec에 없던 모듈] 헤더/API 지식베이스 +
│                            #   복수 검증 사용 예시. fb_sim.py의 STATIC_KB/hint_for를 승격.
│                            #   spec §7엔 없었지만 feasibility Result 3이 이걸 "있으면 좋은 것"
│                            #   에서 "없으면 0%"로 바꿔놔서 core 모듈로 올린다.
├── remask.py               # Task 5b — 저신뢰/무작위 영역 remask (RQ3용 2-arm)
├── harness_compile.py      # Task 5a — 볼륨 마운트 컴파일 실행
└── fitness.py              # Task 7 — compile/target-reach/coverage 채점(§8 그대로)

src/dllm_agent/targets/openexr/
└── harness_reference.py    # Task 4 — 기존 poc_probe.cc 파싱(diff 비교용)

tools/openexr_runtime/harness-compile-only/   # Task 5a — Dockerfile 변형
tools/dream_runtime/requirements.txt          # Task 1 — Dream venv 고정
docs/harness-synthesis-audit-ledger.md        # Task 6 — 원장 스키마 설계 문서
configs/openexr-harness-synthesis-smoke.json  # Task 8 — MVP pilot config
```

`api_knowledge.py`가 유일한 스펙 대비 추가 모듈이다. 나머지는 spec §7과 동일.

**Task 4 실측 후 추가된 것**: `tools/harness_synthesis/`(`_dream_generate.py` — Dream venv 전용
raw 텐서 스크립트, `dream_backend_smoke.py` — 프로젝트 venv에서 subprocess로 호출해 production
모듈까지 재생하는 GPU 스모크). `src/dllm_agent/harness_synthesis/trajectory_hook.py`는 torch를
전혀 import하지 않는다 — 두 venv가 서로의 패키지를 못 보는 실제 제약(§3 NO-START 게이트 1과 같은
근본 원인) 때문에, 텐서 연산은 Dream venv 스크립트가 계산해 JSON으로 넘기고 `src/`는 대조·조립만
한다.

## 5. RQ 운용화

### RQ1 (상관관계) — 재측정, 적정 N

- 신호: top-1 commit confidence(1차, spec §6과 동일), entropy·margin은 2차 feature로 함께
  기록하되 1차 주장에는 top-1만 쓴다(§5의 "1단계는 top-1만" 기각 원칙 유지 — 팀 피드백은
  프레이밍을 바꿨을 뿐 스코프를 늘리라는 지시는 아니었다).
- 정답: statement/#include 단위, 실제 컴파일 오류 유무(1차 컴파일 오류 줄과의 일치) —
  기존 시뮬레이션과 동일 정의.
- N: Task 3의 power calc 결과를 그대로 쓴다. 기존 11/16 대 3/16(효과크기 큼)을 예비 추정치로
  사용.
- 개입 없음 — 순수 관측(spec §9와 동일).

### RQ3 (피드백 유도) — 컴파일 가능한 baseline 위에서

- **전제**: Task 8(a안 KB 다양성 확인)이 먼저 "여러 예시를 가진 KB로도 다수가 컴파일된다"를
  확인해야 한다. 그렇지 않으면 이전 실험처럼 두 arm 모두 0%로 수렴해 RQ3를 측정할 신호가
  없다(`harness-synthesis-feasibility-20260920.md` §"resample"/"feedback" 결과 그대로).
- 2-arm: confidence-국소화 remask 대 무작위 remask, 같은 KB(복수 예시), 같은 NFE, seed로 pairing.
- 지표: (1) 컴파일 성공률, (2) 컴파일된 것들의 coverage-per-NFE(20 s 또는 300 s 창, 기존
  `qualification-smoke`가 쓰는 예산 재사용), (3) reference `poc_probe.cc`와의 coverage 차이.
- 예비등록 임계값: Task 3에서 표본수와 함께 정한다.

### a안 — KB 다양성 (Task 8, RQ1/RQ3보다 먼저)

- 검증된 사용 예시 3–5개를 준비한다(예: `checkOpenEXRFile` 단일 호출, `InputFile` 스트리밍,
  `MultiPartInputFile`). 각각 사람이 손으로 컴파일 확인한 것만 KB에 넣는다(모델이 만든 스니펫을
  KB에 넣지 않는다 — 순환 검증 방지).
- 각 예시당 8 seed, 공유 KB(예시 전부 노출) 대 예시-단독 KB(하나만 노출)로 비교.
- 측정: (1) 컴파일률이 예시 종류에 관계없이 유지되는가, (2) 출력이 자신이 가장 가까운 예시를
  거의 그대로 복사하는가(edit distance/AST 호출 패턴), (3) 서로 다른 예시로 만든 출력들이
  서로 다른가(다양성이 있는가) 대 reference와 얼마나 다른가.
- 이 결과가 부정적이면(항상 복사, 다양성 없음) RQ3의 "무작위보다 낫다"는 결과가 나와도
  실질적 의미가 제한된다는 것을 §1로 다시 문서화해야 한다 — 미리 이 경우의 서술을 준비해둔다.

## 6. Task 목록

- [ ] **Task 0 (완료, 기록만)** — 타당성 시뮬레이션. 산출물:
  `tools/harness_synthesis_feasibility/`, `docs/assets/harness-feasibility-*.json`,
  `docs/harness-synthesis-feasibility-20260920.md`. 커밋 `0f49934`.
- [x] **Task 1 — Dream 실행 환경을 저장소에 고정.** `tools/dream_runtime/requirements.txt`
  (transformers==4.46.2, tokenizers<0.21, numpy<2 — **accelerate는 뺀다**: torch를 자기
  의존성으로 선언해서 uv가 `.pth`로 연결한 프로젝트 torch를 못 보고 새 torch+nvidia 휠을
  받으려다 멈춘다. `AutoModel.from_pretrained(...).to("cuda")`는 accelerate 없이도 동작하고
  이 저장소의 Dream 사용은 그게 전부다) + `tools/dream_runtime/setup.sh`(`.pth`를 pip install
  **전에** 써서 torch 재다운로드를 막고, 프로젝트 torch 2.11.0+cu128 위에 얹는 venv 생성).
  GPU 0에서 로드 검증 완료(VRAM 15.3 GB).
- [x] **Task 1b — `api_knowledge.py` 정적 KB 추출.** `fb_sim.py`의 `STATIC_KB`/`hint_for`를
  production 모듈로 승격: 헤더 목록 추출(`extract_header_index`), 선언 추출(`find_declaration`),
  프롬프트 렌더링(`render_prompt_text`), 컴파일 오류→힌트(`hint_for_missing_headers`,
  `suggest_existing_headers`). Docker fake로 8개 유닛 테스트, ruff/basedpyright clean.
- [x] **Task 2 — `token_trace.py`.** `TokenTrace` dataclass + `permanence_step()`/
  `commit_confidence()`/`position_traces()`. 다중 remask 사이클(commit→remask→recommit)에서
  마지막으로 "고정된" commit을 고르는 로직이 핵심 — 기존 `TrajectoryStep`은 이 패턴을 구조적으로
  못 만든다(spec §4). fake trajectory로 10개 유닛 테스트.
- [x] **Task 3 (경계 탐지만 완료, power calc는 별도) — `regions.py`.** 정규식 기반 statement/
  `#include` 경계 탐지 + `region_at_line`(컴파일 오류 줄 매핑). 9개 유닛 테스트. **power
  calculation 스크립트는 아직 작성 안 함** — Task 8 결과가 나온 뒤 그 분산으로 계산.
- [x] **Task 4 — `trajectory_hook.py` + `harness_reference.py`.** `src/`는 torch를 직접
  import하지 않는다는 이 저장소 관례(`runtime/ar_model_loader.py`) 그대로 따라, 실제 텐서 연산
  (softmax/entropy/margin)은 `tools/harness_synthesis/_dream_generate.py`(Dream venv에서 실행,
  `dllm_agent` import 불가 — venv 분리의 실제 대가)가 계산해 JSON으로 넘기고, `src/`의
  `TrajectoryRecorder`는 그 결과를 `token_ids`와 대조·검증·조립만 한다(불일치 시
  `TrajectoryRecorderError`). `tools/harness_synthesis/dream_backend_smoke.py`로 GPU 0에서
  실제 Dream-Coder 생성 → production 모듈(`TrajectoryRecorder`/`statement_regions`/
  `select_lowest_confidence_region`) 전체 재생 검증 완료: 256개 commit이 bookkeeping 오류 없이
  기록됐고, KB(검증된 사용 예시 1개, `<cstdint>` 포함— 처음엔 빠뜨려서 `uint8_t` 컴파일 오류로
  실패, OpenEXR 헤더가 표준 헤더를 항상 transitively include하진 않는다는 실제 발견) 기반
  harness가 `harness-compile-only` 이미지로 실제 컴파일 성공까지 확인. 9개 유닛 테스트
  (fake `CommitObservation` 기반) + 2개 `harness_reference.py` 테스트.
- [x] **Task 5a — compile-only Docker 이미지.** `tools/openexr_runtime/harness-compile-only/`
  (`Dockerfile` + `build-harness`). `dllm-openexr-runtime`을 베이스로 재사용해 OpenEXR
  재컴파일(10~20분) 없이 harness만 volume mount로 컴파일 — `/work/harness.cc` → coverage 모드는
  `/work/coverage_target`, asan 모드는 `/work/fuzz_target`(기존 `build-coverage`/`build-fuzz`와
  같은 이름, 하위 도구가 모드별 분기 없이 재사용 가능). 실측 검증: reference `poc_probe.cc` 컴파일
  후 11초 동안 216,381 runs(19,671 exec/s); 존재하지 않는 헤더(`ImfStream.h`)는 예상한
  `'ImfStream.h' file not found` 오류로 깨끗이 실패(`api_knowledge.hint_for_missing_headers`가
  파싱하는 형식과 정확히 일치).
- [x] **Task 5b — `remask.py`.** 영역-토큰 위치 매핑(`region_token_positions`), 영역 confidence
  채점(`region_confidence`), 3개 선택 정책(저confidence/무작위/컴파일오류 위치), remask 시퀀스
  구성(`remask_from_position_to_end`). 실제 backend 의존 없음(순수 함수) — Dream 연동은
  Task 4가 담당. 17개 유닛 테스트.
- [ ] **Task 6 — 감사 원장 설계.** `docs/harness-synthesis-audit-ledger.md`: 반복마다
  harness SHA-256 + KB 버전 + 사용 예시 id를 append-only JSONL에 기록하는 스키마.
  `qualification.py`의 evidence-tree 패턴 참고. 스키마 유닛 테스트.
- [ ] **Task 7 — `fitness.py`.** §8 가중합, `w_bug=0, w_diversity=0`로 시작. 유닛 테스트.
- [ ] **Task 8 — a안: KB 다양성 실험 (GPU 필요).** §5 "a안" 절 그대로 실행, 결과를
  `docs/harness-synthesis-diversity-<date>.md`로 기록.
- [ ] **Task 9 — RQ1 재측정 (GPU 필요).** Task 3의 표본수로 관측 상관관계 측정, 사전등록
  임계값과 비교.
- [ ] **Task 10 — RQ3 측정 (GPU 필요).** Task 8이 긍정적일 때만 진행. confidence-국소화 대
  무작위 remask, 사전등록 임계값과 비교.
- [ ] **Task 11 — 종합 보고서.** `docs/research-report.md`의 `[MEASURED]`/`[INFERRED]`/
  `[FUTURE EXPERIMENT]` 태깅 규칙을 따라 RQ1/RQ3/a안 결과를 종합. 취약점 발견이나 dLLM
  우월성을 주장하지 않는다(기존 문서들의 원칙과 동일).

## 7. 이 계획이 하지 않는 것

- 기존 `openexr_one_day` 50-cell 캠페인이나 `revision/` 파이프라인을 수정하지 않는다.
- RL 기반 remask 정책, 다중 타겟, 4개 모델 전부 지원, 풀 AST 파서는 spec §5 그대로 기각.
- "모델이 harness를 완전히 새로 발명한다"는 주장은 하지 않는다 — Task 8이 보여주는 것은
  최선의 경우에도 "검증된 예시를 다양하게 변형"이다.
