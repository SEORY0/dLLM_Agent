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
├── harness_compile.py      # Task 5a — 볼륨 마운트 컴파일 실행 (실물은 tools/openexr_runtime/harness-compile-only/)
├── fitness.py              # Task 7 — compile/target-reach/coverage 채점(§8 그대로)
└── audit_ledger.py         # Task 6 — [신규, spec에 없던 모듈] 반복별 harness identity
                            #   append-only JSONL 원장(docs/harness-synthesis-audit-ledger.md)

src/dllm_agent/targets/openexr/
└── harness_reference.py    # Task 4 — 기존 poc_probe.cc 파싱(diff 비교용)

tools/openexr_runtime/harness-compile-only/   # Task 5a — Dockerfile 변형
tools/dream_runtime/requirements.txt          # Task 1 — Dream venv 고정
docs/harness-synthesis-audit-ledger.md        # Task 6 — 원장 스키마 설계 문서
configs/openexr-harness-synthesis-smoke.json  # Task 8 — MVP pilot config
```

`api_knowledge.py`와 `audit_ledger.py`가 스펙 대비 추가 모듈이다(각각 Task 1b, Task 6에서
필요성이 드러났다). 나머지는 spec §7과 동일.

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
- [x] **Task 3 — `regions.py` + `power_calc.py`.** 정규식 기반 statement/`#include` 경계
  탐지 + `region_at_line`(컴파일 오류 줄 매핑), 9개 유닛 테스트. Power calc:
  `power_calc.required_n_from_discordant_pairs`(Connor 1987 McNemar 공식, 교과서 예시
  `pi_10=0.5, pi_01=0.1 → n=27` 그대로 재현) + `required_n`(마진 확률만으로 계산하는 편의
  함수, 두 detector 간 독립을 가정 — feasibility pilot이 정확한 pairing을 보존하지 않아서
  마진 확률(11/16, 3/16)만 남아있기 때문. 두 detector가 양의 상관관계일 실제 상황에서는
  이 가정이 필요 N을 과대추정하는 쪽으로만 틀린다). 계산된 N: **17**(alpha=0.05,
  power=0.80). 7개 유닛 테스트.
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
- [x] **Task 6 — 감사 원장 설계.** `docs/harness-synthesis-audit-ledger.md` + 구현
  `src/dllm_agent/harness_synthesis/audit_ledger.py`(`HarnessSynthesisLedgerRecord`,
  `load`/`append`/`append_many`/`records_sha256`). 새 패턴을 만들지 않고
  `revision/failure_registry.py`의 append-only JSONL + 배타적 파일 락 구조를 그대로 재사용—
  evidence-tree(한 번 고정된 산출물 증명)와는 목적이 다르다(반복마다 달라지는 값의 이력).
  동시-writer 테스트 포함 10개 유닛 테스트. 남은 결정(KB 버전 고정의 정확한 대상, evidence
  바이트 저장 경로)은 문서에 명시하고 Task 8로 미룸.
- [x] **Task 7 — `fitness.py`.** §8 가중합 4개 유닛 테스트. `diversity`/`bug` 가중치만 기본값
  0(spec §8이 명시적으로 이 둘만 "측정기가 아직 없다"고 지목); `compile`/`reach`/`edge_coverage`는
  필수 인자로 남겨 호출자가 config에서 명시하도록 강제.
- [x] **Task 8 — a안: KB 다양성 실험 (GPU 0·1 사용, 완료).** 검증된 사용 예시 3개
  (`checkOpenEXRFile`, `InputFile`+메모리 `IStream`, `MultiPartInputFile`+같은 `IStream`,
  전부 harness-compile-only 이미지로 개별 컴파일·10초 퍼징까지 사전 검증) × 2 조건(`solo`:
  예시 1개만 노출, `shared`: 3개 전부 노출) × seed 8개 = 48회, GPU 0·1에 24개씩 분산 실행.
  Dream-Coder 실제 생성 → `TrajectoryRecorder` 재생 검증 → `harness-compile-only` 컴파일 →
  `audit_ledger`에 반복마다 기록(첫 시도의 실패 레코드까지 포함해 96줄 append-only,
  `docs/assets/harness-kb-diversity-ledger-20260929.jsonl`). 원시 결과:
  `docs/assets/harness-kb-diversity-results-20260929.json`.

  **인프라 버그 하나 발견·수정**: 처음엔 48/48 전부 컴파일 실패했는데, 원인이 두 가지였다.
  (1) `harness-compile-only` 이미지가 이전 정리 작업 때 지워진 채였다 — `_compile.py`에
  `_ensure_image()`를 추가해 없으면 자동 빌드하도록 고쳤다. (2) 고치고 다시 돌려도 InputFile/
  MultiPartInputFile 예시는 여전히 0/8이었다 — `IStream` 클래스 본문이 `checkOpenEXRFile`
  한 줄짜리 예시보다 훨씬 길어서, 기존 `new_tokens=256` 예산이 함수 서명 중간에서 생성을
  끊어버렸다(`const uint8`에서 잘림). `new_tokens=768, steps=384`로 늘려 재실행.

  **결과 (버그 수정 후, 실제 48/48)**:

  | example | condition | 컴파일 | self_sim 평균 | reference_sim 평균 |
  |---|---|---|---|---|
  | checkOpenEXRFile | solo | 8/8 | 1.000 | 0.837 |
  | checkOpenEXRFile | shared | 8/8 | 0.333 | 0.209 |
  | InputFile | solo | 8/8 | 1.000 | 0.209 |
  | InputFile | shared | 8/8 | 1.000 | 0.209 |
  | MultiPartInputFile | solo | 8/8 | 1.000 | 0.206 |
  | MultiPartInputFile | shared | 8/8 | 0.992 | 0.209 |

  **핵심 발견**: 컴파일률은 예시 종류·조건에 관계없이 48/48로 완벽했지만(예시가 실제로
  검증된 것이면 신뢰성 있게 컴파일된다는 게 재확인됐다), **다양성은 없다**. `shared`
  조건에서 어느 예시를 "의도했든" 상관없이 생성물은 항상 `InputFile` 예시를 사실상 글자
  그대로 복제한다(`checkOpenEXRFile-shared`도 `MultiPartInputFile-shared`도 코드 끝부분이
  `InputFile`과 완전히 동일: `OPENEXR_IMF_NAMESPACE::InputFile file(stream);`). 다만 이
  "의도"는 프롬프트에 실제로 명시된 적이 없다 — `solo`/`shared` 모두 프롬프트 구조가
  동일하고, "이번엔 이 예시를 써라"라는 지시는 없다. 그래서 정확한 결론은 "여러 예시를
  구분 없이 한 KB에 다 보여주면, 모델이 맥락과 무관하게 예시 하나(`InputFile`)로 수렴한다"
  이고, "모델이 의도된 예시를 못 골랐다"는 아니다.

  **Task 8 후속 (완료): `directed` 조건.** 위 구분을 바로 검증했다. `shared`와 똑같이
  예시 3개를 전부 KB에 노출하되, 프롬프트 끝에 "이번 task는 '{example.name}' 예시를
  참고 구현으로 써라, 다른 예시가 아니라"를 명시적으로 덧붙인다(`_kb_and_examples`는
  `directed`도 `shared`와 동일하게 예시 3개를 전부 보여준다 — 차이는 오직 이 지시문뿐).
  3 예시 × seed 4개 × GPU 2장 = 24회.

  | example | 컴파일 | 지시 준수 | self_sim 평균 |
  |---|---|---|---|
  | checkOpenEXRFile | 8/8 | 8/8 | 1.000 |
  | InputFile | 8/8 | 8/8 | 1.000 |
  | MultiPartInputFile | 8/8 | 8/8 | 1.000 |

  **결론**: `shared`의 붕괴는 모델의 능력 한계가 아니라 순수한 프롬프트 설계 문제였다.
  반복마다 "이번엔 이 예시를 참고하라"는 지시 한 줄만 추가하면 24/24 전부 지시된 예시를
  정확히 따르고 24/24 전부 컴파일된다 — `solo`(1.000)와 완전히 동일한 신뢰도다. 즉 **공유
  KB에 여러 예시를 두는 것 자체는 다양성을 해치지 않는다; 어떤 예시를 쓸지 매 반복
  명시적으로 지정하지 않는 것이 문제였다.** Task 9/10은 이 `directed` 프레이밍을
  기본값으로 채택해야 한다("undirected shared"는 null/대조군으로만 남긴다).

  원시 결과: `docs/assets/harness-kb-diversity-directed-results-20260929.json`. 원장은
  기존 파일에 이어 append(총 120줄, `docs/assets/harness-kb-diversity-ledger-20260929.jsonl`).

  **사후 주의(Task 9에서 발견)**: 이 실험은 전부 `alg_temp=0.0`으로 돌았다 — 즉 seed가
  거의 결과를 안 바꾼다(아래 Task 9 참고). 하지만 여기서 측정한 것("이 프롬프트가 컴파일
  가능/지시 준수를 안정적으로 재현하는가")은 애초에 표본 간 독립성을 요구하지 않으므로
  결론은 그대로 유효하다 — 다만 "8 seed"는 통계적 의미의 8개 독립 시행이 아니라 "8번
  반복해도 항상 같은 결과"에 더 가깝다는 점은 밝혀둔다.
- [x] **Task 9 — RQ1 재측정 (GPU 0·1 사용, 완료. 결과: pilot 미재현).** `tools/harness_synthesis/rq1_measurement.py`,
  KB 없이 프롬프트만으로 생성 → 컴파일 실패 시 첫 오류 줄이 속한 region을 정답으로 삼아
  (1) `select_lowest_confidence_region`이 맞히는가, (2) 매칭된 `select_random_region` 추첨이
  맞히는가를 페어로 기록. 목표 N=17(Task 3), GPU 2장에 9개씩 분산 = 18 valid trial.

  **방법론 버그 발견·수정(진행 중 실제로 겪음)**: 첫 시도는 `alg_temp=0.0`(이전 모든
  실험과 동일한 설정)으로 돌렸는데, GPU0의 9개 시도 중 distinct 코드가 4개, GPU1은
  5개뿐이었다 — seed를 바꿔도 대부분 똑같은 출력이 나왔다는 뜻. 원인을 추적하니
  Dream의 `alg_temp=0.0`은 reveal 순서 선택을 `torch.topk`(결정론적)로 만들고, 토큰
  선택 자체도 `sample_tokens()`가 `probs.max(dim=-1)`(argmax)를 쓰기 때문에 `seed`가
  실제로 출력에 거의 영향을 못 준다 — 순수 torch RNG 자체는 seed에 정상 반응함을
  별도로 확인했다(`torch.manual_seed`는 정상 동작, 문제는 Dream의 디코딩 알고리즘
  자체가 이 설정에서 결정론적이라는 것). `_dream_generate.py`에 `--alg-temp` 옵션을
  노출해 RQ1에서만 `alg_temp=0.5`로 재실행(`torch.multinomial`을 타게 만들어 seed가
  실제로 결과를 바꾸게 함) — 재검증 결과 18/18 전부 서로 다른 코드.

  **이 발견이 미치는 영향**: 원래 feasibility pilot(11/16 대 3/16, Result 1)도 같은
  `alg_temp=0.0` 설정으로 돌렸었다 — 즉 그 16개 샘플도 진짜 16개의 독립 표본이 아니었을
  가능성이 높다(정확히 몇 개가 중복이었는지는 그 실행에서 개별 코드를 보존하지 않아
  사후에 확인 불가). Task 8/8b의 결론("이 프롬프트가 X를 안정적으로 재현하는가")은
  독립 표본을 요구하지 않으므로 영향 없지만, RQ1처럼 표본 간 변동 자체가 신호인
  실험에서는 `alg_temp=0.0`이 근본적으로 잘못된 선택이었다.

  **최종 결과 (재검증된 18개 진짜 독립 표본)**:

  | | confidence 정답 | random 정답 |
  |---|---|---|
  | 18개 중 | 1/18 (5.6%) | 3/18 (16.7%) |

  discordant n10(confidence만 맞음)=1, n01(random만 맞음)=3, exact two-sided sign-test
  p=0.625. **pilot의 방향(confidence > random)이 재현되지 않았다** — 점추정은 오히려
  반대 방향이고, 유의하지도 않다. N=18은 pilot 크기의 효과(11/16 대 3/16)를 검출하도록
  설계된 표본수라, 그보다 작은 실제 효과가 있어도 이 실험은 그것을 검출할 검정력이 없다
  — "효과 없음을 증명"이 아니라 "pilot이 시사한 크기의 효과는 재현되지 않았다"까지만
  말할 수 있다. `docs/research-report.md`의 `[MEASURED]` 태그로 종합 보고서(Task 11)에
  이 결과를 pilot의 `[INFERRED]`(재현 실패로 신뢰도 하향) 수치와 나란히 병기해야 한다.

  원시 결과: `docs/assets/harness-rq1-results-20260929.json`(18개 valid trial만).
  원장은 `docs/assets/harness-rq1-ledger-20260929.jsonl`(총 36줄 — 앞 18줄은
  `alg_temp=0.0`으로 돌린 무효 시도, 뒤 18줄이 `alg_temp=0.5`로 재검증한 유효 결과;
  append-only 원칙대로 무효 시도도 지우지 않고 남겼다).
- [ ] **Task 10 — RQ3 측정 (GPU 필요).** confidence-국소화 대 무작위 remask, 사전등록
  임계값과 비교. baseline은 Task 8 후속의 `directed` 프레이밍(예시별 명시적 참조 지시,
  24/24 컴파일·24/24 지시 준수)으로 확보됐다 — `undirected shared`는 쓰지 않는다.
- [ ] **Task 11 — 종합 보고서.** `docs/research-report.md`의 `[MEASURED]`/`[INFERRED]`/
  `[FUTURE EXPERIMENT]` 태깅 규칙을 따라 RQ1/RQ3/a안 결과를 종합. 취약점 발견이나 dLLM
  우월성을 주장하지 않는다(기존 문서들의 원칙과 동일).

## 7. 이 계획이 하지 않는 것

- 기존 `openexr_one_day` 50-cell 캠페인이나 `revision/` 파이프라인을 수정하지 않는다.
- RL 기반 remask 정책, 다중 타겟, 4개 모델 전부 지원, 풀 AST 파서는 spec §5 그대로 기각.
- "모델이 harness를 완전히 새로 발명한다"는 주장은 하지 않는다 — Task 8이 보여주는 것은
  최선의 경우에도 "검증된 예시를 다양하게 변형"이다.
