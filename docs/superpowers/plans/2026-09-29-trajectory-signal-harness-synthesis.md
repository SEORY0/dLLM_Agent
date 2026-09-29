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

> **2026-09-29 갱신**: 이 섹션이 1차 주장으로 삼은 RQ1(trajectory confidence가 harness
> correctness와 상관관계가 있다)은 이후 실측(Task 9, 224개 독립 표본 누적)에서
> 지지되지 않았다 — §6-1 참고. 이 섹션은 계획 수립 당시의 프레이밍을 그대로 남겨둔
> 기록이다.

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

  원시 결과(N=18): `docs/assets/harness-rq1-results-20260929.json`.

  **확장 재측정(N=112, 완료)**: 사용자가 "a안"(표본 늘려서 재시도)을 택해 이어서 진행했다.
  N=18의 pilot 효과 크기(11/16 대 3/16)를 검출하는 검정력은 이미 있었지만, "그래도 뭔가
  작은 효과는 있을 수 있다"는 여지를 좁히기 위해 더 작지만 여전히 실용적으로 의미 있는
  효과(`p_confidence=0.33` 대 `p_random=0.167`, N=18에서 관측된 random 기저율의 약 2배)를
  검출하도록 `power_calc.required_n(0.33, 0.167)`로 **N=112**를 계산했다(이 효과 크기
  자체는 N=18 데이터를 보고 사후에 고른 것이라 엄밀한 사전등록은 아니고, 탐색적 2차
  분석으로 취급해야 한다). 같은 생성 설정(`alg_temp=0.5, new_tokens=384, steps=192`)으로
  기존 18개에 94개(GPU 2장에 47개씩)를 더해 총 112개 진짜 독립 표본(112개 코드 전부
  서로 다름)을 확보했다.

  | N=112 최종 | confidence 정답 | random 정답 |
  |---|---|---|
  | | 7/112 (6.2%) | 14/112 (12.5%) |

  discordant n10=7, n01=14, exact two-sided sign-test **p=0.189**. 표본을 6배로
  늘려도 유의하지 않고, 점추정은 여전히 random이 confidence를 앞선다(방향이 pilot과
  반대). 이 정도 검정력에서도 재현되지 않는다는 것은 pilot의 11/16 대 3/16이 alg_temp=0.0
  중복 표본에서 나온 우연한(비독립) 결과였을 가능성에 무게를 싣는다. **RQ1(top-1
  commit confidence가 컴파일 오류 위치와 상관관계가 있다)은 이 신호·이 region 정의·이
  실험 설계로는 기각에 가깝다** — 다만 entropy/margin 등 다른 신호나 더 정교한 region
  정의는 시도하지 않았으므로 "top-1 confidence는 통하지 않는다"까지만 말할 수 있다.

  원시 결과(N=112): `docs/assets/harness-rq1-results-n112-20260929.json`.

  **옵션 (c) — entropy/margin 신호(N=112, 완료, 답 없음)**: 사용자가 "c 진행 후에도 답이
  없으면 b"를 지시해 이어서 진행했다. `token_trace.py`에 `commit_entropy`/`commit_margin`,
  `remask.py`에 `region_entropy`/`region_margin`/`select_highest_entropy_region`/
  `select_lowest_margin_region`을 추가(기존 confidence API는 동작 변경 없이 공유 헬퍼로
  일반화, 91개 유닛 테스트 통과). 같은 파이프라인으로 confidence와 같은 N(112, 완전히
  새로운 독립 표본 112개)에서 세 신호를 한 번에 측정(같은 생성을 재사용하므로 추가 GPU
  비용 없음).

  | 신호 | 정답 | random 정답 | n10 | n01 | sign-test p |
  |---|---|---|---|---|---|
  | confidence | 5/112 | 11/112 | 4 | 10 | 0.180 |
  | entropy | 14/112 | 11/112 | 12 | 9 | 0.664 |
  | margin | 4/112 | 11/112 | 3 | 10 | 0.092 |

  confidence·margin은 여전히 random보다 낮은 정답률(반대 방향, 유의하지 않음).
  entropy는 random보다 약간 높지만(14 대 11) p=0.664로 **이 비교에서 우위의 증거가
  없다**. 세 신호 전부 N=112에서 유의한 상관관계를 보이지 않는다 — top-1 confidence,
  entropy, margin 어느 것도 이 region 정의·이 실험 설계에서 컴파일 오류 위치를
  무작위보다 잘 맞히지 못한다. 사용자 지시대로 (a)·(c) 모두 답이 없으므로 **(b)로
  이동**: RQ1은 이 실험 설계로는 보류하고, 이후 작업은 RQ3(개입 질문)에 집중한다.

  원시 결과: `docs/assets/harness-rq1-entropy-margin-n112-20260929.json`. 원장은
  `docs/assets/harness-rq1-ledger-20260929.jsonl`(총 247줄 — 맨 앞 18줄만 `alg_temp=0.0`
  무효 시도, 나머지 전부 `alg_temp=0.5` 유효 시도; append-only 원칙대로 무효 시도도
  지우지 않고 남겼다).

## 6-1. RQ1 최종 판정과 (b)로의 전환

**정정**: 이 절 초안이 "총 242개"라 잘못 적었다 — N=18은 첫 N=112(confidence, a안 확장)에
**포함된** 부분집합이지 별도 표본이 아니다. 실제 독립 표본은 confidence N=112 + entropy/
margin N=112 = **총 224개**다(원본 결과·원장 파일은 수정하지 않았고, 이 요약 서술만
정정한다).

두 표본수/신호 묶음(confidence N=112 — N=18을 포함, entropy/margin N=112, 총 224개 진짜
독립 표본)에서 top-1 confidence·entropy·margin 중 어느 것도 컴파일 오류
위치를 무작위보다 유의하게 잘 맞히지 못했다. RQ1(dLLM의 denoising confidence가 harness
correctness와 상관관계가 있다)은 **이 신호 집합·이 region 정의·이 프롬프트 설계**로는
지지되지 않는다. 시도하지 않은 것: (1) 더 정교한 AST 기반 region 정의(현재는 정규식
statement 단위), (2) 여러 신호의 조합(예: confidence+entropy 앙상블), (3) 다른
prompt/온도 설정. 이들은 "RQ1이 완전히 틀렸다"가 아니라 "이 MVP 설계로는 검증되지
않았다"는 의미로 남겨둔다. Task 10부터는 RQ3(개입 질문)에 집중하고, RQ1 미지지 사실을
숨기거나 RQ3 결과로 소급 정당화하지 않는다(§Task 10의 기존 주의사항 참고).

- [~] **Task 10 — RQ3 측정 (GPU 사용 중, 스모크 완료·본 실험 미착수).** confidence-국소화
  대 무작위 remask, 사전등록 임계값과 비교. baseline은 Task 8 후속의 `directed` 프레이밍
  (예시별 명시적 참조 지시)이다. **주의(Task 9 이후 갱신)**: RQ3는 "confidence로 고른
  위치를 고치면 무작위로 고친 것보다 결과가 나아지는가"라는 개입 질문이고, RQ1(순수
  관측 상관관계)이 기각됐다고 자동으로 기각되는 건 아니다 — 틀린 위치를 못 맞혀도 remask
  자체(반복 수정 기회)가 무작위보다 나은 결과를 낼 가능성은 남아있다. 다만 RQ1의 미재현은
  이 개입이 통할 사전 확률을 낮추는 증거이므로, Task 11 종합 보고서에서는 두 결과를
  분리해서 정직하게 병기해야 한다 — RQ3가 양성이어도 "그러니까 confidence localization이
  통한다"고 되짚어 주장하면 안 된다.

### Task 10 스모크 (완료)

**동결된 프로토콜**(스모크 결과를 보기 전에 `tools/harness_synthesis/rq3_smoke.py` 상단
docstring에 확정, 결과를 본 뒤 바꾸지 않음):

1. `directed` KB baseline. 매 trial마다 초기 harness 1개를 생성하고, **같은 harness**를
   confidence-remask와 random-remask 두 arm으로 분기한다. 원본(수정 없음)은 reference arm으로
   기록한다.
2. arm은 eligible region(같은 harness에서 탐지된 동일 region 집합)과 regeneration
   seed(둘 다 같은 seed — "어느 region을 고쳤는가"만 유일한 차이가 되게)로 맞춘다.
   `remask_from_position_to_end`는 시작 위치부터 끝까지 전부 remask하므로 더 이른 위치를
   고르면 무조건 더 많은 토큰을 고치게 된다 — 강제로 동일하게 만들지 않고(끝까지 remask해야
   줄 길이가 늘거나 줄 수 있다는 기존 설계 이유 유지), 대신 매 trial마다 실제 remask된
   토큰 수와 재생성 step 수를 arm별로 기록해 감사 가능하게 한다.
3. 1차 outcome은 **시도한 모든 trial**(컴파일 실패 포함) 위에서 정의: 고정 fuzz 시간 아래
   edge coverage(컴파일 실패 시 edges=0으로 처리, intent-to-treat). 컴파일 성공률은 별도
   보고. 컴파일 조건부 edge는 2차. reference 대비 delta가 핵심 비교, origin 예시별로 층화.
4. `alg_temp=0.5` 전 구간, 매 trial 코드를 sha256으로 append-only 원장에 기록.
5. 스모크는 확증 데이터로 안 쓴다 — 메커니즘·비용 대칭성·outcome 변동성·fitness 측정
   가능성만 검증하고, 그 관측 분산으로 본 실험 표본수를 계산한 뒤 성공 기준과 분석 계획을
   먼저 고정한다.

**스모크 실행**: GPU 2장, 예시 3개 × seed 4개 = 12 trial(fuzz 20초). 원시 결과
`docs/assets/harness-rq3-smoke-results-20260929.json`, 원장(36줄, trial당 3 arm)
`docs/assets/harness-rq3-smoke-ledger-20260929.jsonl`.

**스모크 발견 1 — `directed` baseline이 실제로는 훨씬 덜 안정적이다.** Task 8b는 (같은
프롬프트·같은 토큰 예산으로) seed 3000000~3000007에서 24/24 컴파일을 얻었다. 이번 스모크는
**다른** seed(5000000~5000003대)에서 reference arm(수정 없음, `directed` 그대로) 컴파일이
**5/12(42%)**에 그쳤다. 실패 양상은 다양하다: 환각 헤더(`'ImfMultiPartInputFile' file not
found`), 환각 API(`no member named 'checkOpenEXRFile' in namespace 'Imf'`), 미완성
클래스(`expected '}'`, budget 소진으로 문장이 안 끝남). **Task 8b의 24/24는 8개 seed
표본의 우연**이었을 가능성이 높다 — `directed`의 실제 신뢰도는 100%가 아니라 훨씬 낮고
seed에 따라 크게 흔들린다. 이건 RQ3의 baseline 자체가 노이즈가 크다는 뜻이라, 나중에
`directed` 자체의 컴파일률을 더 큰 N으로 재확인할 필요가 있다(이 계획에는 아직 없음).

**스모크 발견 2 — arm 간 비용 비대칭이 실제로 크다(감사 항목 2 확인됨).** 예: `MultiPartInputFile`
seed=5000001에서 confidence arm은 493 토큰/246 step, random arm은 756 토큰/378 step —
약 1.5배 차이. `InputFile` seed=5000000: confidence 646/323 대 random 752/376. 강제로
맞추지 않기로 한 설계 그대로지만, 실제로 상당한 비대칭이 있다는 걸 확인했다 — 본 실험
분석에서 이 비용을 공변량으로 보고하거나 최소한 병기해야 한다.

**스모크 발견 3 — 흥미로운 개별 사례, 결론은 아님.** `InputFile` seed=5000003: reference와
random 둘 다 컴파일 실패(0 edges)였는데 confidence arm만 컴파일에 성공해 1135 edges를
얻었다(delta +1135). 반대 방향 사례는 없었다. n=1이라 잡음일 수도 있다 — 본 실험이
있어야 재현되는지 알 수 있다.

**1차 outcome 집계(12 trial, ties=양쪽 delta 동률인 경우, 대부분 양쪽 다 컴파일 실패한
"양쪽 0" 동률)**: confidence가 random보다 나은 trial 3개, random이 나은 trial 2개, 동률
7개(그중 상당수가 "reference도 arm도 다 컴파일 실패"인 정보량 없는 동률). 컴파일률 자체는
reference 5/12, confidence 7/12, random 5/12로 세 arm이 비슷하다.

**본 실험 표본수 계산**: RQ1과 같은 방식(McNemar paired-proportions, `power_calc.
required_n_from_discordant_pairs`)을 "confidence가 이겼다/random이 이겼다" 이산 승패에
적용한다(`pi_10=3/12≈0.25`, `pi_01=2/12≈0.167` — 스모크 자체에서 관측된, 아주 작은
차이). 결과: **N=469**. 표본이 이렇게 커진 이유는 (a) 관측된 승률 차이가 원래 작고, (b)
컴파일 실패로 인한 동률이 많아 유효 표본을 희석하기 때문이다. trial당 대략
GPU-분(초기 생성+remask 2회+컴파일·fuzz 3회) 단위이므로, N=469는 GPU 2장으로도 상당한
시간이 든다 — 실행 전에 사용자 확인이 필요하다.

**동결한 것(본 실험을 실제로 시작하기 전까지 바꾸지 않음)**: 위 §1-5 프로토콜 그대로,
1차 outcome 정의(고정 fuzz 시간 edge coverage, intent-to-treat), 분석 방법(discordant-pair
sign test, RQ1과 동일 방식), 유의수준 alpha=0.05·power=0.80.

**사용자 결정(2026-09-29)**: N=469 확증 실험은 시작하지 않는다. 스모크가 baseline
substrate 자체의 불확실성(24/24 대 5/12)을 드러냈으므로, 다음 단계는 treatment effect를
추정하지 않는 순수 baseline 안정성 감사(RQ3-B0)다. 기존 스모크 결과와 위 동결 프로토콜은
그대로 보존하고 확증 증거로 재해석하지 않는다.

### RQ3-B0 — Directed Baseline Stability Audit (완료, 64/64)

**목적**: confidence 대 random을 비교하지 않는다 — 오직 `directed`/reference 생성
substrate가 RQ3 확증 실험을 지탱할 만큼 안정적인지만 본다.

**데이터셋**: 새 seed 64개, `checkOpenEXRFile`/`InputFile`/`MultiPartInputFile` 각각
22/21/21개(고정 dataset에 검증된 예시가 3개뿐이라 "8 task × 8 seed"가 아니라 이렇게
배분함 — 예시를 8개로 늘리려면 각각 독립 컴파일 검증이 필요한 별도 작업이다). seed는
`_EXAMPLES` 순서대로, 결과를 보지 않고 사전에 순서대로 배정한다(`rq3_b0_baseline_audit.
seed_assignments()`).

**정확한 fresh-seed 규칙**: 6,000,000–6,000,063(64개, 이 감사 전용). 이번 확증 RQ3와
겹치지 않도록 6,000,000–6,000,999 전체를 향후 RQ3 확증 실험에서 사용 금지 구간으로
예약한다(재실행·확장 여유분 포함). 지금까지 실제로 쓰인 다른 구간과도 안 겹친다:
`kb_diversity_experiment.py`(Task 8/8b) 3,000,000–3,000,007, `rq1_measurement.py`
(Task 9) 4,000,000대~4,010,100대, `rq3_smoke.py`(Task 10 스모크) 5,000,000–5,000,003.

**프로토콜**: RQ3 스모크의 reference arm과 완전히 동일(baseline 성공률을 올리려고
바꾸지 않는다) — 모델, `alg="entropy", temperature=0.5, top_p=0.95, alg_temp=0.5`,
`new_tokens=768, steps=384`(예시별로 조정하지 않음), `directed` 프롬프트,
harness-compile-only 이미지 coverage 모드, `compile_and_fuzz(fuzz_seconds=20.0)`, GPU
1장당 32 trial씩 2장.

**trial당 기록**: example, seed, compiled, failure_category, generation_tokens_ceiling/
generation_steps_ceiling(상수), effective_token_count(닫는 fence 전까지 실제 토큰 수),
hit_budget(닫는 fence를 budget 안에서 못 찾음), compile_error(원문 그대로 보존), edges
(미컴파일 시 0), wall_time_seconds. 원장에는 `arm="b0-audit"`로 별도 기록해 RQ3 확증
데이터와 섞이지 않게 한다.

**실패 분류(우선순위 순 — budget 소진을 언어 오류로 오분류하지 않도록 먼저 검사)**:
(1) generation-budget exhaustion/truncation, (2) hallucinated header, (3) hallucinated
API/symbol, (4) type/signature mismatch, (5) syntax/incomplete generation, (6)
linker/build failure, (7) other. `rq3_b0_baseline_audit.classify_failure()`로 구현,
7개 케이스 모두 유닛 확인 완료(GPU 불필요).

**분석(서술적 substrate 검증만)**: 전체 컴파일률 + Wilson 95% CI, 예시별 컴파일률,
실패 카테고리별 개수, hit_budget 비율, effective_token_count 분포, 컴파일된 것들의
edges. confidence 대 random 비교는 하지 않는다.

**decision gate(결과를 보기 전에 고정, 심각한 순서로 검사)**:

| 순서 | 조건 | 다음 행동 | 근거 |
|---|---|---|---|
| 4 | 전체 컴파일률 < 40% **그리고** hit_budget 비율 < 30% | `directed` substrate 자체를 재검토 | budget 문제가 아닌데도 낮으면 프롬프트/예시 설계 결함일 가능성이 크다 |
| 3 | hit_budget 비율 ≥ 30% | RQ3 전에 `new_tokens`/`steps` ceiling부터 재검토 | budget 부족이 주원인이면 모델 능력이 아니라 파라미터 설정 문제라서, ceiling만 늘려도 해결될 수 있다 |
| 2 | 예시별 컴파일률 최댓값-최솟값 ≥ 40%p | RQ3를 예시별로 층화하거나 재설계 | 한 예시만 신뢰할 수 있고 나머지는 아니라면, 3개를 하나로 묶어 분석하는 현재 설계가 틀렸다는 뜻 |
| 1 | 위 세 조건 모두 해당 없음(컴파일률 ≥ 40%, hit_budget < 30%, 격차 < 40%p) | RQ3 표본수를 이 감사의 관측 컴파일률로 재계산하고 진행 | substrate가 쓸 만하다고 판단, 다음은 N 재계산 |

숫자 기준(40%, 30%, 40%p)은 지금 제안하는 것이며, 결과를 본 뒤 유리하게 바꾸지 않는다.
40%는 "동전 던지기보다는 낫다"는 최소선이자 스모크의 5/12(42%)와 거의 같은 값으로 잡아
스모크 관측치를 우연히 정당화하는 임계값이 아니게 했다. 30%는 "budget 소진이 주된
원인이면 이건 모델 문제가 아니라 설정 문제"라는 성격 구분선이다. 40%p는 임의적이나,
3개 예시 중 하나가 극단적으로 다르면(예: 90% 대 40%) 그 차이를 무시하고 합쳐 분석하는
게 부적절하다는 상식적 기준이다.

**표본수(N=469) 처리**: 스모크 12 trial(confidence 승 3, random 승 2, 동률 7)에서 나온
역사적 pilot 추정치로만 보존한다 — 이 discordance/tie율 추정이 극히 적은 N에 기반해
민감하다는 걸 명시한다. 이 감사 이후 최종 RQ3 표본수는 (a) 미리 정한 과학적으로 의미
있는 최소 효과크기, (b) 이 감사가 준 usable/discordant trial 빈도의 방어 가능한 추정치,
(c) 독립적인 확증 데이터셋을 써서 다시 계산해야 한다. 그 재계산 결과를 사용자가 검토하기
전까지 최종 실험을 시작하지 않는다.

**1차 endpoint는 이 단계에서 바꾸지 않는다**: 고정 fuzz 시간 edge coverage, intent-to-
treat, 컴파일 실패 보존. 다만 RQ3 확증 실험을 위해 아래를 필수 2차 outcome으로 준비해
둔다: 컴파일 성공, 컴파일 성공 조건부 coverage, generation tokens, generation steps,
wall time, edges/token(또는 다른 명확한 2차 효율 지표). arm 간 실제 token/step 수를
강제로 맞출 필요는 없다 — 허용된 resource ceiling(같은 `new_tokens`/`steps` 상한)이
동일하기만 하면 된다.

**ITT 1차 endpoint에 문제가 있는가**: 통계적으로는 문제없다(컴파일 실패를 0으로 대체하는
설계라 실패율이 얼마든 유효하다). 다만 이 감사가 보여줄 실제 컴파일률이 검정력에 직접
영향을 준다 — N=469가 컸던 이유 자체가 "양쪽 다 실패해서 생기는 동률"이 많았기 때문이라,
ITT를 바꿀 필요는 없지만 이 감사로 컴파일률을 먼저 재는 게 왜 중요한지의 이유이기도 하다.

**예상 GPU/실행 비용**: trial당 생성 1회(remask 없음) + compile+fuzz 1회만 필요해 RQ3
본 실험 trial(생성 1 + remask 2 + compile+fuzz 3)보다 훨씬 싸다. 스모크 실측 기준
trial당 대략 60~90초로 추정 — GPU 2장에 32개씩이면 전체 약 30~50분.

**구현**: `tools/harness_synthesis/rq3_b0_baseline_audit.py`(frozen docstring +
`seed_assignments()`/`classify_failure()` 순수 함수, GPU 없이 유닛 검증 완료).

#### 실행 결과 (2026-09-30, GPU 2장, 64/64 완료)

원시 결과 `docs/assets/harness-rq3-b0-audit-results-20260930.json`, 원장(64줄,
`arm="b0-audit"`) `docs/assets/harness-rq3-b0-ledger-20260930.jsonl`.

| | 컴파일 | Wilson 95% CI |
|---|---|---|
| 전체 | 21/64 = 32.8% | [22.6%, 45.0%] |
| checkOpenEXRFile | 16/22 = 72.7% | [51.8%, 86.8%] |
| InputFile | 4/21 = 19.0% | [7.7%, 40.0%] |
| MultiPartInputFile | 1/21 = 4.8% | [0.8%, 22.7%] |

예시 간 격차 68.0%p. hit_budget(생성 도중 budget이 다 떨어져 코드가 안 끝난 경우)은
64개 중 **0개(0.0%)** — budget 문제가 전혀 아니다. 실패 카테고리(미컴파일 43건):
other 15, hallucinated API/symbol 13, hallucinated header 7, syntax/incomplete
generation 5, type/signature mismatch 3.

**사후 특성화(분류는 다시 안 건드림, 원문 진단을 다시 읽어서 설명만 추가)**: "other"
15건을 직접 읽어보니 두 가지 패턴이 taxonomy 정규식에 안 걸렸을 뿐이었다 — (a)
`undefined reference to 'LLVMFuzzerTestOneInput'`(linker가 진입점을 못 찾음, 3건),
(b) `variable type 'MemoryIStream' is an abstract class`/`non-virtual member
function marked 'override' hides virtual member function`(IStream 추상 클래스의
가상함수 시그니처를 정확히 못 맞춤). 이 (b) 패턴이 `InputFile`/`MultiPartInputFile`
실패의 상당수를 차지한다: 두 예시는 `MemoryIStream : public IStream`을 직접
구현해야 하는데, `tellg()`/`seekg()`의 반환형·시그니처를 조금씩 틀리거나 추상
메서드를 다 못 채워서 "추상 클래스라 인스턴스화 불가" 오류가 반복된다.
`checkOpenEXRFile`은 함수 호출 한 줄이라 이 문제가 없고, 실패도 대부분 단순
헤더/API 환각이다. **즉 68%p 격차의 원인이 명확하다: 커스텀 추상 클래스 구현
난이도.**

**decision gate 판정**: **Gate 4 발동**(전체 컴파일률 32.8% < 40% **그리고**
hit_budget 0.0% < 30%) — "`directed` baseline 자체를 재검토"가 사전에 정한 다음
행동이다. Gate 2도 독립적으로 발동(격차 68.0%p ≥ 40%p)하지만, 미리 정한 우선순위상
Gate 4가 더 심각해 그 행동이 적용된다. **Gate 1(표본수 재계산 후 진행)에 도달하지
못했다** — 즉 지금 이 substrate로는 RQ3 표본수를 다시 계산해서 바로 진행하면 안
된다.

**결론과 제안**: `directed` 프레이밍 자체가 아니라 **`InputFile`/`MultiPartInputFile`
두 예시가 요구하는 커스텀 `IStream` 구현이 모델에게 근본적으로 어렵다**는 것이
원인으로 보인다. `checkOpenEXRFile`만 놓고 보면 72.7%(CI 51.8~86.8%)로 Gate 1
기준(≥40%)을 넉넉히 통과한다. 다음 결정은 사용자가 해야 한다: (a) RQ3를
`checkOpenEXRFile` 단일 예시로 좁혀서 그 substrate로 표본수를 다시 계산, (b)
`InputFile`/`MultiPartInputFile`용 KB 예시나 프롬프트를 개선해서 재감사, (c) 이
방향(remask 개입 실험) 자체를 재검토. Task 11 종합 보고서에는 "RQ3 substrate
안정성은 예시에 따라 근본적으로 다르다"는 이 결과를 그대로 기록해야 한다.

- [ ] **Task 11 — 종합 보고서.** `docs/research-report.md`의 `[MEASURED]`/`[INFERRED]`/
  `[FUTURE EXPERIMENT]` 태깅 규칙을 따라 RQ1/RQ3/a안 결과를 종합. 취약점 발견이나 dLLM
  우월성을 주장하지 않는다(기존 문서들의 원칙과 동일).

## 7. 이 계획이 하지 않는 것

- 기존 `openexr_one_day` 50-cell 캠페인이나 `revision/` 파이프라인을 수정하지 않는다.
- RL 기반 remask 정책, 다중 타겟, 4개 모델 전부 지원, 풀 AST 파서는 spec §5 그대로 기각.
- "모델이 harness를 완전히 새로 발명한다"는 주장은 하지 않는다 — Task 8이 보여주는 것은
  최선의 경우에도 "검증된 예시를 다양하게 변형"이다.
