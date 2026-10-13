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

**결론과 제안(완화된 표현)**: `directed` 프레이밍 전체가 아니라 **`InputFile`/
`MultiPartInputFile` 두 예시가 요구하는 커스텀 `IStream` 구현 난이도가 유력한 주요
원인으로 보인다**(n=21씩이고 "other" 15건 중 일부만 직접 읽어본 것이라 단정하지는
않는다). `checkOpenEXRFile`만 놓고 보면 72.7%(CI 51.8~86.8%)로 Gate 1 기준(≥40%)을
넉넉히 통과한다.

### RQ3-B0v2 — 1회 한정 KB/프롬프트 개선 재감사 (완료, **gate 통과 실패**)

사용자가 옵션 (b)(KB/프롬프트 개선, 1회 한정)를 선택해 진행했다. 기존 v1 결과·원장·
동결 프로토콜은 전혀 건드리지 않았고, 개선판은 `kb_v2_directed_prompt.py` +
`rq3_b0_baseline_audit_v2.py`로 완전히 별도 파일에 기록했다.

**바꾼 것(딱 1가지 종류)**: 예시 코드(`MemoryIStream` 클래스, API 호출) 자체는 v1과
바이트 단위로 동일하게 유지했다 — 실제 설치된 헤더로 이미 계약대로 맞다고 재확인했기
때문이다(`/opt/openexr-coverage/include/OpenEXR/ImfIO.h`를 다시 열어 `grep
"virtual|= 0"`로 확인: `IStream`은 `read`/`tellg`/`seekg` 세 개만 pure virtual이고
나머지(`isMemoryMapped`, `clear`, `size`, `isStatelessRead` 등)는 기본 구현이 있어
오버라이드할 필요가 없다 — 기존 예시는 이미 정확했다). 바꾼 건 프롬프트 텍스트뿐이다:
(1) 위 3개 시그니처를 명시하는 "IStream 계약" 문단 추가, (2) libFuzzer 진입점 시그니처
계약 명시, (3) "이 예시를 글자 그대로 베껴라, 시그니처를 하나도 바꾸지 마라"는 명시적
지시 추가. **고정 IStream을 코드로 강제 주입하지는 않았다** — 그건 "harness를 모델이
쓴다"는 과제 범위 자체를 바꾸는 것이라 임의로 적용하지 않고 이 문서에 옵션으로만
남긴다(아래 "다음 결정" 참고).

**재감사 기준(실행 전 고정, v1과 동일 임계값 재사용 — 결과 보고 새로 안 만듦)**: Gate
4(컴파일률<40% 그리고 hit_budget<30% → substrate 재검토), Gate 2(격차≥40%p →
재설계), Gate 1(전부 해당 없음 → pilot 진행). 새 seed 8,000,000–8,000,063(64개,
8,000,000–8,000,999 전체를 이후 RQ3 확증 실험에서 사용 금지 구간으로 예약).
`checkOpenEXRFile` 8개는 회귀
확인용 대조군(프롬프트가 이 예시에도 약간 바뀌었으므로), `InputFile`/
`MultiPartInputFile` 각 28개.

**정정(예시 배분이 다름)**: v1은 예시당 21~22개, v2는 checkOpenEXRFile 8개(대조군)·
InputFile/MultiPartInputFile 각 28개로 N이 달라 "전체 컴파일률"을 v1/v2 사이에서
그대로 비교하면 가중치가 달라 오해를 부른다. 아래는 **예시별** 비교를 중심으로 본다.

| | 컴파일 | Wilson 95% CI | v1과 비교(예시별) |
|---|---|---|---|
| checkOpenEXRFile (대조군, n=8) | 8/8 — **이번 표본에서 실패 없음** | [67.6%, 100%] | v1 72.7%(n=22); n=8의 CI가 넓어 v1과 모순되지 않음 |
| InputFile (n=28) | 3/28 = 10.7% | [3.7%, 27.2%] | v1 19.0%(n=21) — 개선 안 됨(CI 겹쳐서 유의한 악화라고도 못 함) |
| MultiPartInputFile (n=28) | 3/28 = 10.7% | [3.7%, 27.2%] | v1 4.8%(n=21) — 명목상 소폭 상승, CI 겹침 |

hit_budget은 v1과 동일하게 0/64(0.0%) — 여전히 budget 문제가 아니다. 예시 간 격차는
**89.3%p**로 v1(68.0%p)보다 커졌다(checkOpenEXRFile 대조군이 n=8이라 "실패 없음"이
곧바로 100%로 계산되는 소표본 효과가 크게 작용함).

**decision gate 판정**: **Gate 4 다시 발동**(InputFile·MultiPartInputFile 둘 다
<40% 그리고 hit_budget<30%). 사전에 정한 대로, **결과를 보고 기준을 다시 조정하지
않는다** — 즉 이 1회 개선 시도는 실패로 판정하고 v3는 만들지 않는다.

**실패 사례 재확인(분류기는 안 건드리고 원문만 다시 읽음)**: v2에서 새로 나타난 오류
패턴 — `only virtual member functions can be marked 'override'`, `expected class
name` — 는 "글자 그대로 베껴라"는 지시가 오히려 클래스 선언부를 미묘하게 망가뜨리는
경우가 늘었을 가능성을 보여준다(추측이며 확정은 아니다 — 프롬프트가 더 길어진 것
자체가 요인일 수도 있다). 이 1회 시도로는 원인을 더 좁히지 못했다.

**결론**: 프롬프트 문구를 더 명시적으로 만드는 것만으로는 `InputFile`/
`MultiPartInputFile`을 고치지 못했다. 다음 결정은 사용자 몫이다: (a) RQ3를
`checkOpenEXRFile` 단일 substrate로 좁혀서 표본수 재계산(v1·v2 양쪽에서 안정적으로
검증됨), (b) 고정 IStream을 코드로 주입하는 방식으로 과제 범위를 바꿔서 재시도(별도
승인 필요), (c) `InputFile`/`MultiPartInputFile`을 RQ3에서 완전히 제외. **gate가
통과하지 못했으므로 소규모 paired remask pilot은 진행하지 않았다** — 사용자 지시대로
"재감사 통과 시에만" 진행하는 단계였다.

원시 결과 `docs/assets/harness-rq3-b0v2-audit-results-20260930.json`, 원장(64줄,
`arm="b0v2-audit"`) `docs/assets/harness-rq3-b0v2-ledger-20260930.jsonl`.

### coverage 측정 변동성 확인 (사용자 요청)

B0 v1의 `checkOpenEXRFile` 성공 16건 중 **9건이 완전히 동일한 소스 코드**였고(모델이
예시를 그대로 베낀 경우), 그 9건의 edge coverage는 **1795~1932**(스프레드 137,
평균 대비 약 7.6%)로 달랐다 — 컴파일된 바이너리가 똑같은데도 fuzz 결과가 흔들린다는
뜻이라 별도로 확인했다.

그중 하나를 고정해 같은 바이너리를 다른 libFuzzer `-seed`로 10회, 그리고 seed=1을
2회 더 반복 fuzz(전부 `fuzz_seconds=20`)했다:
- seed 1~10: 1861, 1922, 1848, 1836, 1871, 1823, 1832, 1829, 1831, 1811 (스프레드 111)
- seed=1을 3번(원래 1회 + 반복 2회): 1861, **1861, 1830** — **같은 seed조차 정확히
  재현되지 않는다.**

**유력한 기여 요인**: `-max_total_time=20`(벽시계 기준 예산)을 쓰고 있어서, 실행
횟수 자체가 시스템 타이밍에 따라 흔들릴 수 있다 — `-seed`가 같아도 몇 번 실행됐는지가
매번 다를 수 있다는 정황과 일치한다(다른 원인, 예: 시스템 부하·스케줄링도 배제하지
못했다). **관측된 범위**: 9개 동일 소스에서 1795~1932(관측 범위 137), 고정 바이너리
반복 fuzz 10회에서 1811~1922(관측 범위 111), 같은 seed 반복 3회에서 1861/1861/1830.
표본이 적어 이 범위를 "±N" 형태의 분포 모수로 일반화하지는 않는다. RQ3의 confidence
대 random 비교가 감지해야 할 실제 효과가 이 관측 범위보다 작으면 단일 fuzz 실행만으로는
구분이 어려울 수 있다 — pilot을 설계한다면 arm별로 같은 fuzz seed 묶음으로 반복
측정해 변동을 직접 추정하거나, `-runs=`(고정 실행 횟수)를 보조 분석으로 병행하는 걸
고려해야 한다. 원시 수치는 `docs/assets/harness-rq3-coverage-variance-check-20260930.json`.

### RQ3-v3 — 고정 scaffold 방식으로 과제 범위 변경 (사용자 승인, 완료: feasibility PASS + 소규모 pilot)

**과제 재정의**: v1/v2는 모델에게 "전체 harness를 작성"하도록 시켰고, 그 안에는 손으로
검증 가능하지만 자주 훼손되는 `MemoryIStream` 클래스 구현이 포함되어 있었다 —
`InputFile`/`MultiPartInputFile`의 gate 미통과 원인이 IStream 구현 난이도로
좁혀졌다(B0v2 절 참고). v3는 사용자 승인을 받아 과제를 바꾼다: includes, 검증된
`MemoryIStream` 클래스(B0v2에서 실제 ImfIO.h로 재검증된 것과 동일 소스,
`kb_diversity_experiment._MEMORY_ISTREAM`에서 그대로 재사용), libFuzzer 진입점, `try {`
까지를 고정 Python 문자열로 취급하고, 모델은 try 블록 안의 1~4문장(API 호출 본문)만
생성·수정한다. 고정 영역은 Dream의 `--prefill-suffix`로 프리필에 붙여서 애초에 모델이
생성하는 텍스트에 포함되지 않으므로 — "베끼라고 지시했는데 훼손됨"이 아니라 구조적으로
훼손이 불가능하다. 구현: `tools/harness_synthesis/kb_v3_fixed_scaffold.py`
(`scaffold_prefix`/`scaffold_suffix`/`assemble`/`body_prompt`/`extract_body` — 중괄호
깊이 0 스캔으로 모델이 생성한 본문의 끝을 찾음). 기존 v1/v2 결과·동결 프로토콜·B0/B0v2
감사 기록은 전부 보존, 변경 없음 — v3는 별도 실험 버전.

**scaffold 자체의 독립 컴파일 검증**: `placeholder_harness()`(본문에 `(void)0;`만 채운
버전)로 scaffold가 그 자체로 컴파일됨을 확인. 고정 영역은 애초에 모델 생성 대상이 아니므로
"생성 중 변경되지 않았는지 확인"은 구성상 자명하다 — 모델이 볼 수 있는 텍스트는
`body_prompt()` 지시문과 `prefill_suffix`(자기 자신이 방금 만든 고정 텍스트)뿐이고,
Dream 쪽 스크립트는 `prefill_suffix` 뒤에서부터만 새 토큰을 생성·remask한다.

**Feasibility 체크 (동결 프로토콜, `rq3_v3_feasibility.py`)** — 목적: 본문 생성이
안정적으로 끝나는지(rambling 없이), 그리고 remask 후보가 될 만한 서로 다른 영역이
충분히 생기는지 확인하는 go/no-go 체크. `InputFile`/`MultiPartInputFile` 각 12개,
seed 9,000,000–9,000,023(9,000,000–9,000,999대 예약, 향후 RQ3 확증 실험에서 제외),
`new_tokens=60, steps=30, alg_temp=0.5`, `fuzz_seconds=10`(짧게, feasibility용).
사전 동결 gate(순서대로 확인, 결과를 본 뒤 기준을 바꾸지 않음): ①추출 성공률<70%→중단
②완료된 본문 중 region_count≥2 비율<50%→중단 ③서로 다른 본문 텍스트 3개 미만→중단
④위 모두 통과→소규모 paired pilot으로 진행.

**Feasibility 결과 (2026-09-30, 24/24 완료)**:
- 추출 성공: **24/24 = 100%** (gate ① 여유 있게 통과)
- 완료된 본문 중 region_count≥2: **16/24 = 66.7%** (gate ② 통과)
- 서로 다른 본문 텍스트: **24/24 = 100%** (gate ③ 여유 있게 통과)
- 참고 지표(gate 대상 아님): 24건 중 9건 컴파일 성공(37.5%) — v1/v2보다 높지만 이 단계는
  본문 다양성·완결성만 보려는 것이므로 본 비교로 해석하지 않는다.
- **판정: PASS → 소규모 paired pilot 진행.** 원시 결과
  `docs/assets/harness-rq3-v3-feasibility-results-20260930.json`, 원장(24줄,
  `arm="v3-feasibility"`) `docs/assets/harness-rq3-v3-feasibility-ledger-20260930.jsonl`.

**소규모 paired pilot (동결 프로토콜, `rq3_v3_pilot.py`)** — 설계: 트라이얼마다 본문을
1회 생성하고, 같은 본문을 confidence-remask 팔과 random-remask 팔로 분기(같은 감지된
영역 후보, 같은 재생성 seed — remask 시작 위치만 다름, v1 스모크와 동일한 매칭 방식),
원본 본문은 reference 팔로 그대로 둔다. remask는 본문 자체(고정 scaffold는 대상이
될 수 없음)에만 적용. 주 지표는 v1과 동일하게 고정 시간 fuzz coverage, intent-to-treat
(컴파일 실패 시 0 edges로 집계, 제외하지 않음). v1과의 차이: coverage 측정 변동성
확인에서 드러난 잡음을 사후 캐비앗으로 남기지 않고 직접 추정하기 위해, 각 팔을 같은
libFuzzer seed 묶음(1, 2, 3), 각 `fuzz_seconds=20`으로 반복 fuzz하고 평균/최소/최대를
기록. 고정 실행 횟수(`-runs=100000`) 1회는 보조 지표로만 병행. 데이터셋: 예시당 6
seed(`InputFile`, `MultiPartInputFile`) = 12 트라이얼, seed 10,000,000–10,000,011
(10,000,000–10,000,999대 예약). 팔별 `tokens_remasked`/`steps`(비용)를 강제로 맞추지
않고 그대로 기록 — 이것이 사후에 핵심 교란요인으로 드러난다(아래).

**Pilot 결과 (2026-09-30, 12/12 완료) — 최초 보고 정정.** 최초 보고에서 "다른 영역
8건 + 같은 영역 3건"이라고 썼는데, 이는 12건 중 11건만 설명하는 계산 오류였다.
실제로는 **같은 영역 선택 5건, 다른 영역 선택 7건**이며, 아래 표가 12건 전부다.

| seed | example | regions | 영역선택 | conf_tok | rand_tok | ref 컴파일 | conf 컴파일 | rand 컴파일 | ref edges | conf edges | rand edges |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10000000 | InputFile | 2 | 같음 | 35 | 35 | ✗ | ✗ | ✗ | 0 | 0 | 0 |
| 10000001 | InputFile | 2 | 다름 | 35 | 60 | ✗ | ✗ | ✗ | 0 | 0 | 0 |
| 10000002 | InputFile | 1 | 같음 | 60 | 60 | ✓ | ✓ | ✓ | 1172 | 1159 | 1196 |
| 10000003 | InputFile | 1 | 같음 | 60 | 60 | ✗ | ✗ | ✗ | 0 | 0 | 0 |
| 10000004 | InputFile | 2 | 다름 | 37 | 60 | ✗ | ✗ | ✗ | 0 | 0 | 0 |
| 10000005 | InputFile | 2 | 다름 | 35 | 60 | ✓ | ✓ | ✓ | 1212 | 1183 | 1228 |
| 10000006 | MultiPartInputFile | 2 | 다름 | 38 | 60 | ✗ | ✗ | ✗ | 0 | 0 | 0 |
| 10000007 | MultiPartInputFile | 2 | 다름 | 36 | 60 | ✗ | ✓ | ✗ | 0 | 1121 | 0 |
| 10000008 | MultiPartInputFile | 1 | 같음 | 60 | 60 | ✗ | ✗ | ✗ | 0 | 0 | 0 |
| 10000009 | MultiPartInputFile | 2 | 다름 | 28 | 60 | ✗ | ✓ | ✗ | 0 | 1117 | 0 |
| 10000010 | MultiPartInputFile | 2 | 같음 | 32 | 32 | ✗ | ✓ | ✓ | 0 | 1153 | 1134 |
| 10000011 | MultiPartInputFile | 2 | 다름 | 34 | 60 | ✗ | ✗ | ✗ | 0 | 0 | 0 |

합계: 같음 5건(10000000, 10000002, 10000003, 10000008, 10000010), 다름 7건(나머지) —
5+7=12. `region_count=1`인 3건(10000002, 10000003, 10000008)은 애초에 고를 대상이
하나뿐이라 "같은 영역"이 자명하게 강제된 경우이고, `region_count=2`이면서 우연히 같은
영역을 고른 경우는 10000000, 10000010 두 건뿐이다.

컴파일 성공: reference 2/12, confidence 5/12, random 3/12.

**reference 컴파일 실패 10건 감사 (GPU 실행 전, 저장된 본문 재분석)**: `regions.py`의
`statement_regions()`를 실패한 본문에 다시 돌려 본 결과, 12건 전부에서 검출된 영역은
항상 주석이 아닌 완결된 C++ 문장(세미콜론으로 끝남)이었고, 컴파일 성공 여부와 무관하게
영역 탐지 자체는 정상 작동했다(즉 gate ②의 region_count 수치는 신뢰할 수 있다). 10건의
컴파일 실패 원인을 개별적으로 읽어 분류하면:

- **OpenEXR API 오용 — 9건**: 존재하지 않는 메서드 호출(`headerVersion()` ×2,
  `headerType()`, `readHeader()`, `fileHeader()` — 총 5건), 잘못된 생성자 인자
  (`&stream` 포인터를 넘김 — 프롬프트가 명시적으로 "&stream 아니라 stream" 이라고
  지시했음에도 위반, 3건), 존재하지 않는 타입명(`ImfHeader`, 1건), 필수 인자 누락
  (`MultiPartInputFile::header()`는 `partNumber`가 필요한데 누락, 1건) — 일부 트라이얼은
  두 문제가 겹친다(예: seed 10000004).
- **본문 추출/scaffold 결합 오류 — 0건**: 모든 본문의 중괄호 깊이가 0으로 닫히고, 고정
  suffix와 결합했을 때 생긴 구조적 불일치는 없었다. `extract_body`가 잘못 잘랐다고
  의심할 근거는 없다.
- **그 외(위 두 범주에 안 맞음) — 1건**: seed 10000009는 API 호출이 아니라 본문 자체에
  `return ;`(값 없는 return)을 써서, `int`를 반환해야 하는 `LLVMFuzzerTestOneInput` 안에서
  경고가 에러로 처리됐다. OpenEXR API를 잘못 쓴 게 아니라 모델이 자신이 함수 전체가 아니라
  본문 일부만 쓰고 있다는 제약을 어기고 제어 흐름 문장을 넣은 경우다 — 이 실패는 추출
  버그도 아니고 API 오용도 아닌 별도 범주로 남겨둔다.

또한 영역 선택 자체에 관해 한 가지 일관된 패턴이 관찰된다: `region_count=2`인 9건
전부에서 confidence는 항상 **두 번째(더 뒤의) 문장** — 즉 실제 OpenEXR API를 호출하는
문장 — 을 선택했고, 첫 번째(생성자) 문장을 고른 적은 없다. 이 자체는 "confidence가
의미적으로 그럴듯한 지점(실제 버그가 있을 만한 API 호출부)을 가리킨다"는 것과도
부합하지만, 같은 본문에서 두 번째 문장은 구조상 항상 remask 범위가 더 짧다
(`remask_from_position_to_end`가 뒤쪽 위치에서 시작할수록 남은 토큰 수가 적어지므로).
**따라서 "confidence가 어느 문장을 짚었는가"와 "그 문장을 고름으로써 수정량이 얼마나
줄었는가"가 이 실험 설계상 구조적으로 분리되지 않는다** — 둘 중 어느 쪽이 컴파일률
차이를 만들었는지 이 데이터만으로는 가릴 수 없다. 이전 보고에서 이를 "confidence가 더
쉬운 과제를 받아서"라고 단정한 것은 과도한 해석이었다: 정확히는 **위치 선택 효과와
수정량 효과가 이 설계에서는 분리되지 않는다**는 것이 사실이고, 그 이상을 주장할 수 없다.

같은 영역을 고른 5건 중 "region_count=2이면서 우연히 일치"한 2건(10000000, 10000010)과
"애초에 후보가 하나뿐"인 3건(10000002, 10000003, 10000008)은 confidence/random이
정확히 같은 시작 위치·같은 재생성 seed로 돌아갔으므로, 둘 사이에 나는 coverage 차이는
위치 선택 효과가 아니라 **동일 조건 재실행 시의 실행 변동(같은 B0v2 coverage-variance
확인에서 이미 관측된 종류의 잡음)을 보여주는 표본**으로만 취급해야 한다 — confidence가
"위치를 잘 골랐다/못 골랐다"는 증거로 쓰지 않는다. 참고로 이 조건에서 둘 다 컴파일된
쌍은 10000002(-37)와 10000010(+19)이며, 이 수치를 과거에 측정한 잡음 범위(111~137
edges)와 비교해 "크다/작다"를 판정하는 것도 하지 않는다 — 그 범위는 다른 조건(같은
소스를 여러 seed로 반복 fuzz)에서 나온 것이라 이 비교의 유의성 판정 기준으로 쓸 근거가
없다. 단지 관측된 숫자로만 남겨둔다.

**결론(정정)**: 이 pilot에서는 confidence 팔의 표면적 컴파일률 우위(5/12 vs 2/12,
3/12)가 관찰되지만, 실험 설계상 위치 선택과 수정량이 얽혀 있어 그 우위가 "위치를 잘
짚어서"인지 "더 적게 고쳐서"인지 분리할 수 없다. 같은 영역/조건을 고른 5건은 위치
선택 효과의 증거가 아니라 실행 변동 참고 자료로만 남긴다. n=12는 어느 쪽으로도 결론을
낼 수 있는 크기가 아니며, 이 설계로는 표본을 더 모아도 이 confound가 해소되지 않는다.

**다음 실험 버전(설계 변경 필요, 아래 절)**: confound를 없애려면 선택 규칙과 무관하게
수정량 자체를 고정해야 한다 — confidence/random이 어느 위치를 고르든 동일한 길이의
token window만 remask하도록 재설계한다.

원시 결과 `docs/assets/harness-rq3-v3-pilot-results-20260930.json`, 원장(36줄,
`arm="{example}-{arm}"`) `docs/assets/harness-rq3-v3-pilot-ledger-20260930.jsonl`.

### RQ3-v3-window — 수정량을 고정한 재설계 (완료, 12/12)

**구현**: `src/dllm_agent/harness_synthesis/remask.py`에 `TokenWindow`(고정 길이
후보), `fixed_length_windows()`(본문 토큰 수를 겹치지 않는 동일 길이 window로 분할,
나머지는 버림), `select_lowest_confidence_window`/`select_random_window`(같은
window 후보 집합에서 선택), `remask_window()`(선택된 window만 마스킹, 그 외 위치는
그대로 유지)를 추가했다(테스트 `tests/test_harness_synthesis_remask.py`, 34개 전부
통과). 기존 `remask_from_position_to_end`와 `select_*_region` 계열은 그대로 두고
보존했다 — 이 재설계는 새 함수 추가이지 기존 함수 변경이 아니다.

**동결 프로토콜** (`tools/harness_synthesis/rq3_v3_window_pilot.py`, 실행 전 작성,
결과를 본 뒤 바꾸지 않음):
- window 길이 **8 토큰**, 고정. 근거: 이미 보고된(수정하지 않은) region 기반 pilot에서
  confidence가 고른 영역의 remask 비용이 대략 28~38 토큰이었고, 짧은 C++ 문장 하나가
  대략 8~15 토큰이므로, 8은 그보다 작아 대부분의 본문이 window 2개 이상으로 나뉠
  것으로 예상했다 — 이번 실험 결과를 보고 고른 값이 아니다.
- 후보 집합: 추출된 본문(고정 scaffold 제외) 토큰 범위를 `fixed_length_windows`로
  분할한 동일 window 목록. confidence는 평균 commit confidence가 가장 낮은 window,
  random은 같은 seed로 같은 목록에서 균등 추출 — 같은 초기 코드·같은 재생성 seed
  공유(이전 스모크·pilot과 동일 매칭 방식).
  remask·재생성 비용은 이제 강제로 동일(`tokens_remasked=8`, `steps=8`, 양팔 동일)하다
  — 기록만 하는 게 아니라 설계 자체로 같다.
- 중단 기준(고정, 실행 전): window가 2개 미만인 트라이얼은 후보가 없어 건너뛰고
  기록만 한다(양팔 실행 안 함); 전체 시도의 절반 넘게 이 사유로 건너뛰면 그 배치
  이후 중단하고 infeasible로 보고, window 길이를 줄여 재시도하지 않는다.
- 검증(고정, 매 트라이얼 자동 확인, 주장이 아니라 단언): 재생성 후 **선택된 window
  밖의 모든 위치**가 재생성 전 토큰과 정확히 일치하는지, 그리고 조립된 harness의
  고정 prefix/suffix가 `scaffold_prefix`/`scaffold_suffix`의 출력과 정확히 일치하는지
  둘 다 코드로 확인(`outside_window_preserved`).
- 주 지표: 이전과 동일하게 고정 시간(20초) fuzz coverage, intent-to-treat, 팔당
  동일한 libFuzzer seed 묶음(1,2,3) 반복 측정 + 고정 실행 횟수(100,000) 보조 지표.
- 데이터셋: 예시당 6 seed = 12 트라이얼, **새 seed 블록 11,000,000–11,000,011**
  (11,000,000–11,000,999대 예약 — 기존 v3 pilot의 10,000,000대, feasibility의
  9,000,000대와 겹치지 않음).

**실행 결과 (2026-09-30, GPU 2장, 12/12 완료, 0건 스킵)**:

검증 결과 — **24/24 팔(12 트라이얼 × 2 팔) 전부 `outside_window_preserved=True`**:
window 밖 토큰과 고정 scaffold 모두 매 트라이얼에서 보존이 확인됐다.

| seed | example | body_tok | windows | 같은 window | conf 컴파일(edges) | rand 컴파일(edges) |
|---|---|---|---|---|---|---|
| 11000000 | InputFile | 36 | 4 | 다름 | ✓(1169) | ✓(1176) |
| 11000001 | InputFile | 36 | 4 | 다름 | ✗(0) | ✓(1206) |
| 11000002 | InputFile | 36 | 4 | 다름 | ✗(0) | ✗(0) |
| 11000003 | InputFile | 27 | 3 | 다름 | ✓(1174) | ✓(1224) |
| 11000004 | InputFile | 23 | 2 | 다름 | ✓(1170) | ✓(1168) |
| 11000005 | InputFile | 33 | 4 | 다름 | ✗(0) | ✓(1221) |
| 11000006 | MultiPartInputFile | 37 | 4 | 다름 | ✗(0) | ✗(0) |
| 11000007 | MultiPartInputFile | 36 | 4 | **같음** | ✗(0) | ✗(0) |
| 11000008 | MultiPartInputFile | 37 | 4 | 다름 | ✓(1144) | ✓(1118) |
| 11000009 | MultiPartInputFile | 38 | 4 | 다름 | ✓(1149) | ✓(1123) |
| 11000010 | MultiPartInputFile | 60 | 7 | 다름 | ✓(1146) | ✓(1133) |
| 11000011 | MultiPartInputFile | 35 | 4 | 다름 | ✓(1128) | ✓(1117) |

reference 컴파일 10/12, **confidence 7/12, random 9/12** — 비용을 맞추자 confidence의
컴파일률이 random보다 오히려 낮다(수치상 우위가 사라졌을 뿐 아니라 반대 방향).
seed 11000007(같은 window 선택)은 confidence/random이 완전히 같은 마스크·같은
재생성 seed로 돌아갔으므로 두 팔의 결과가 동일(둘 다 실패)한 것이 설계대로다 —
이는 위치 선택 효과의 증거가 아니라 매칭이 의도대로 작동했다는 확인일 뿐이다.

ITT paired diff(confidence − random, 0 포함 12건): -7, -1206, 0, -50, +2, -1221, 0,
0, +26, +26, +13, +11. **승(conf>rand) 5, 패 4, 동률 3.** 동률을 제외한 9건에 대한
정확 이항 부호검정: p=1.0(양측) — 완전히 대칭이라 방향성의 증거가 전혀 없다. 이전
region 기반 pilot에서 보였던 "confidence가 컴파일을 더 잘 살린다"는 인상은, 수정량을
동일하게 맞추자 사라졌다 — 오히려 이 표본에서는 random 쪽이 컴파일률이 조금 더
높다(둘 다 n=12라 이 차이도 단정할 근거는 못 된다).

**결론**: 위치 선택과 수정량을 분리한 재설계에서는 confidence-remask가 random보다
낫다는 증거가 없다(컴파일률도, coverage도) — RQ1의 null 결과 및 region 기반
pilot을 비용까지 통제해 다시 본 결론과 일관된다. 검증 항목(window 밖 보존, scaffold
보존)은 설계대로 100% 통과해, 이 결과가 구현 결함이 아니라 confidence 신호 자체가
이 개입에서 이점을 주지 못한다는 쪽에 더 가깝다는 근거가 늘었다.

**권고**: 추가 프롬프트 튜닝, 표본 확대, N=469 본 확증 실험 모두 진행하지 않는다.
n=12(비동률 9)로는 애초에 작은 효과를 감지할 힘이 없으므로 "효과가 없다"를 확정하는
것도 아니다 — 다만 지금까지 나온 모든 신호(RQ1 null, region 기반 pilot의 비용-통제
부분, 이번 window pilot)가 같은 방향(이점 없음 또는 판정 불가)을 가리킨다는 점에서,
이 설계로 표본을 더 늘리는 것보다는 사용자가 이 결과를 검토한 뒤 RQ3 전체의 지속
여부를 결정하는 것이 우선이라고 본다.

원시 결과 `docs/assets/harness-rq3-v3-window-pilot-results-20260930.json`, 원장(36줄,
`arm="{example}-window-{arm}"`) `docs/assets/harness-rq3-v3-window-pilot-ledger-20260930.jsonl`.

**N=469 본 확증 실험은 계속 보류.** 추가 프롬프트 튜닝도 진행하지 않았다 — 사용자
지시대로 이번 라운드는 감사 + 재설계 + 소규모 pilot 보고까지만.

- [x] **Task 11 — 종합 보고서.** 범위 조정: `docs/research-report.md`는 PrimeVul/CyberGym
  기반의 별도 취약점-에이전트 연구 트랙(H1–H5 factorial, jsoncons/OpenEXR native
  tournament)을 다루는 문서이고 이 harness-synthesis RQ1/RQ3와는 서로 다른 연구
  질문·데이터라, 거기에 종합하면 두 트랙이 섞인다. 대신 아래 §8에 이 문서 자체 안에서
  전체 파이프라인·정정된 표본수·판정 근거·한계를 종합했다. 취약점 발견이나 dLLM
  우월성은 주장하지 않는다(기존 문서들의 원칙과 동일).

## 7. 이 계획이 하지 않는 것

- 기존 `openexr_one_day` 50-cell 캠페인이나 `revision/` 파이프라인을 수정하지 않는다.
- RL 기반 remask 정책, 다중 타겟, 4개 모델 전부 지원, 풀 AST 파서는 spec §5 그대로 기각.
- "모델이 harness를 완전히 새로 발명한다"는 주장은 하지 않는다 — Task 8이 보여주는 것은
  최선의 경우에도 "검증된 예시를 다양하게 변형"이다.

## 8. 종합 결론과 RQ3 종료 (2026-09-30)

**범위**: 이 절은 confidence 기반 remask 개입(RQ3)과 그 전제였던 RQ1(confidence
localization) 연구를 **종료**한다. dLLM 기반 퍼징 에이전트 프로젝트 전체를 종료하는
것이 아니다 — `docs/README.md`의 다른 트랙(온라인 포트폴리오 revision, jsoncons/OpenEXR
native tournament, 결정론적 replay QA)은 이 문서와 무관하게 계속된다.

### 8-1. 전체 파이프라인 재구성 (분기점부터 HEAD까지)

각 단계의 목적·주요 변경·핵심 수치·판단 근거·한계. 원시 결과와 동결 프로토콜은
모두 원래 위치에 그대로 보존했고, 이 절은 그것들을 다시 쓰는 것이 아니라 종합만 한다.

| # | 단계 | 목적 | 핵심 수치 | 한계 |
|---|---|---|---|---|
| 0 | 문헌 조사 + feasibility 시뮬레이션 (`tools/harness_synthesis_feasibility/`) | 실제 GPU를 쓰기 전에, "compile-error feedback + 공유 지식베이스" 아이디어가 시뮬레이션 수준에서 말이 되는지 확인 | 시뮬레이션 상에서 confidence 조건이 두드러짐(실제 모델 아님) | 실제 Dream-Coder 출력이 아니라 손으로 구성한 시뮬레이션 — 이후 모든 단계의 전제가 아니라 "해볼 가치가 있는지"의 사전 점검일 뿐 |
| 1 | Dream 런타임 격리 + 핵심 모듈 (Task 1–7: `token_trace`/`regions`/`remask`/`api_knowledge`, compile-only 이미지, `TrajectoryRecorder`, ledger) | 실제 Dream-Coder-7B를 GPU에서 돌리고, 그 결과를 torch 비의존 순수 모듈로 재생(replay)할 수 있는 프로덕션 경로 구축 | 실제 GPU smoke 통과 | 이 자체는 결과가 아니라 인프라 — RQ1/RQ3의 전제 조건 |
| 2 | KB 다양성 실험 (Task 8/8b) | "모델이 검증된 예시를 KB로 주면 harness를 재현하는가" 확인 | `solo`/`shared` 조건에서 붕괴 관찰, `directed`(명시적 참조 지시) 조건에서 회복 | 단순 예시(`checkOpenEXRFile`)에서만 확인, 복잡한 예시는 이후 단계(B0/B0v2)에서 계속 실패로 드러남 |
| 3 | RQ1 측정 (Task 3/9, option c) | confidence(및 entropy/margin)가 컴파일 오류 위치를 무작위보다 잘 짚는지 | 1차 N=18(파일럿, `alg_temp=0.0`이라 seed 간 독립성 결함 발견) → 재측정 N=112(confidence) → N=112(entropy/margin, 겹치지 않는 새 seed) = **독립 표본 총 224건**. 3개 신호 모두 무작위 대비 우위의 증거 없음 | 표본이 국소적(단일 헤더/단일 오류 유형), 오류 위치 정의가 statement 단위로 거칠음. **정정**: 이전 보고의 N=242는 계산 오류였고 실제로는 224 |
| 4 | RQ3 스모크 (Task 10) | RQ1이 기각된 상태에서, remask 개입 자체(위치가 아니라 "고치는 행위")가 coverage를 바꾸는지 아주 작은 규모로 먼저 확인 | 12 trial, directed KB 기반. confidence 승 3/random 승 2/동률 7 — 방향성 자체보다 **동률이 압도적으로 많다는 것**(양쪽 다 컴파일 실패)이 더 중요한 관찰이었음 | 표본이 매우 작음; 이 결과에서 사전에 필요한 표본수를 역산하면 N=469(§8-2)가 나오지만, 그 전에 baseline이 안정적인지부터 의심하게 됨 |
| 5 | RQ3-B0 baseline 안정성 감사 (완료, 64/64) | 본 실험 전에 "directed KB 기반이 충분히 안정적으로 컴파일되는가"만 서술적으로 확인(confidence-vs-random 비교 아님) | 컴파일률 32.8%(Wilson CI), budget-hit 0% — 사전 동결한 4단계 gate 중 **Gate 4(불안정) 발동** | `InputFile`/`MultiPartInputFile`의 `MemoryIStream` 구현 난이도가 원인으로 좁혀짐(예시별 실패 원인 분해로 확인) |
| 6 | RQ3-B0v2 (완료, 재감사 실패) | KB/프롬프트를 한 차례만 명시적으로 개선(실제 이미지로 재검증한 IStream 계약 명시)한 뒤 재감사 | 새 seed 64건, 컴파일률 소폭 변화했으나 **Gate 4 재발동** | 프롬프트 문구 개선만으로는 부족 — "직접 전체 harness를 베끼게 하는" 접근 자체의 한계로 판단, 사용자가 과제 범위 변경(고정 scaffold)을 승인 |
| 7 | RQ3-v3 고정 scaffold (완료, feasibility PASS) | 과제를 "전체 harness 생성"에서 "고정된 scaffold 위 1~4문장 본문 생성"으로 재정의해 IStream 구현 난이도를 구조적으로 제거 | feasibility 24/24: 추출 성공 100%, 다중 영역 66.7%, 본문 다양성 100% → PASS | 이 단계 자체는 "생성이 안정적으로 끝나는가"만 보는 것으로, confidence 비교가 아님 |
| 8 | RQ3-v3 region 기반 pilot (완료, 표기 오류 정정됨) | 고정 scaffold 위에서 confidence-remask vs random-remask 소규모 비교 | 12 trial(같은 영역 5, 다른 영역 7 — **최초 보고 "8+3=11"은 계산 오류였고 이번에 정정**), 컴파일 reference 2/12, confidence 5/12, random 3/12 | **위치 선택과 수정량(remask 비용)이 설계상 분리되지 않음** — `region_count=2`인 9건 전부에서 confidence가 항상 더 뒤쪽(항상 더 싼) 문장을 골랐기 때문에, 컴파일률 차이가 "위치를 잘 짚어서"인지 "더 적게 고쳐서"인지 이 데이터만으로 가릴 수 없음(§RQ3-v3 절에서 상세 표·감사 포함) |
| 9 | RQ3-v3-window 재설계 (완료) | 위치 선택과 수정량을 분리하기 위해, 고정 길이(8 토큰) window 후보로 재설계 | 12 trial(새 seed), 24/24 팔에서 window 밖 보존·scaffold 보존을 코드로 검증. 컴파일 reference 10/12, **confidence 7/12, random 9/12**(우위가 사라짐, 방향도 반대). ITT paired 부호검정 승5/패4/동률3, **p=1.0** | n=12로는 애초에 작은 효과를 감지할 힘이 없음 — "효과 없음의 증명"이 아니라 "이 표본에서 방향성의 증거가 전혀 없음" |

### 8-2. 표본수 관련 정정

- **N=469는 폐기된 과거 추정치다.** RQ3 스모크(12 trial, directed KB 기반)의 승/패/동률
  비율에서 McNemar 공식으로 역산한 수치이며, 그 직후 B0 감사가 baseline 자체의
  불안정성(Gate 4)을 드러내 **한 번도 사용되지 않았다**. 이후 B0v2·v3(고정 scaffold,
  region 기반·window 기반)로 설계가 두 번 바뀌었으므로, N=469는 지금의 설계·모집단과
  무관한 수치다. 앞으로 이 개입을 다시 검토한다면 표본수는 그때의 설계로 새로 계산해야
  하며, N=469를 재사용하거나 참고 기준으로 삼지 않는다.
- RQ1의 표본수는 224(N=18+N=112+N=112가 아니라 N=18이 첫 N=112에 포함되므로
  18을 포함한 112 + 새 112 = 224)로 이미 정정되어 있다(§6-1).
- RQ3-v3 pilot(region 기반, window 기반) 각각 n=12는 사전 검정력 계산 없이 고정한
  탐색적 규모다 — 이 크기에서 "효과 없음"을 통계적으로 확정할 수 없고, 그렇게 서술하지
  않는다.

### 8-3. 결론 — 구분해서 읽을 것

- **관측 (dLLM 실행 가능성)**: Dream-Coder-7B로 OpenEXR 퍼징 하네스(전체 또는 고정
  scaffold 위 본문)를 생성하고, 실제로 컴파일·libFuzzer 실행까지 반복적으로 수행할 수
  있다는 것은 여러 단계에 걸쳐 실제 GPU에서 확인됐다(수백 건의 생성·컴파일 시도,
  성공·실패 모두 원장에 기록).
- **결과 (RQ1/RQ3 판정)**: 지금까지 테스트한 조건(이 모델, 이 프롬프트·scaffold, confidence/
  entropy/margin 신호 정의, RQ1의 224 표본과 RQ3의 여러 소규모 pilot)에서, confidence 기반
  위치 선택/remask가 무작위보다 낫다는 **우위가 확인되지 않았다**. 이는 "confidence
  신호에 효과가 없다"는 확정된 결론이 아니라 — 표본이 대체로 작고 일부는 검정력 계산
  없이 진행된 탐색 단계였으므로 — "이 실험들에서 우위의 증거를 찾지 못했다"는 서술이다.
- **미판정 (dLLM 대 AR)**: dLLM과 자기회귀(AR) 모델 사이의 퍼징 성능·버그 발견·계산
  비용 우위는 이 연구에서 다루지 않았다 — 비교 대상이 되는 AR 실행 자체가 이 파이프라인
  안에서 설계·수행된 적이 없다. 이 질문은 여전히 미판정으로 남는다.

**쓰지 않을 표현**: "confidence 신호는 효과가 없다"(확정형), "dLLM은 퍼징을 못한다",
"수정량(remask 비용) 차이 때문에 기존의 컴파일률 우위가 생겼다"(§8-1 8단계의 컴파일률
차이는 위치·비용이 얽혀 있어 원인을 가를 수 없다고만 말할 수 있고, "비용 때문"이라고
원인을 확정할 근거는 없다) — 이 문서와 이후 요약 어디서도 이런 데이터보다 강한 표현은
쓰지 않는다.

### 8-4. 종료 범위

- confidence 기반 위치 선택/remask에 대한 추가 프롬프트 튜닝, 추가 pilot, N=469 또는
  새로 계산한 어떤 N의 확증 실험도 진행하지 않는다.
- `tools/harness_synthesis/`의 모든 스크립트와 `src/dllm_agent/harness_synthesis/`의
  모든 모듈, 원시 결과·원장(`docs/assets/harness-*`)은 삭제하지 않고 그대로 보존한다 —
  향후 다른 신호나 다른 설계로 이 질문을 다시 열 때의 출발점이자, 이번 탐색의 감사
  기록이다.
- dLLM 기반 퍼징 에이전트 프로젝트 자체, 그리고 `docs/README.md`에 기록된 다른
  트랙들(온라인 포트폴리오 revision, native tournament, 결정론적 replay QA, AR-parent
  control sampler 등)은 이 종료의 대상이 아니다.

### 8-5. 후속 (2026-10-13)

팀 리뷰에서 "위치는 컴파일러나 퍼저가 알려주고, dLLM은 양방향 문맥으로 그 구간만 다시
생성"하는 방향이 제시됐다. 이것은 위 종료 범위(모델 자신의 신호로 위치 선택)를 다시 여는 것이
아니라, 위치 결정을 외부 oracle로 넘긴 별도 질문(RQ4)이다 —
[`2026-10-13-compiler-localized-span-infill.md`](../specs/2026-10-13-compiler-localized-span-infill.md).
이 문서의 결과·원장은 그대로 두고, RQ4는 이 문서의 v3 scaffold와 `remask_window` 검증
결과를 출발점으로만 재사용한다.
