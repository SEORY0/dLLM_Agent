# Trajectory-Guided Harness Synthesis — Design & Feasibility

> Status: **설계안(design), 미구현**. 이 문서는 착수 전 근거 조사와 스코프 확정을 위한 것이다.
> 구현 계획(task breakdown)은 이 문서가 확정된 뒤 `superpowers:writing-plans` 형식으로 별도 작성한다.

## 1. 배경

사용자가 제시한 아이디어(GPT 초안 포함)의 핵심: dLLM은 AR과 달리 왼쪽부터 순차 생성하지
않고 전체 시퀀스를 병렬로 복원한다. 따라서 "먼저 나온 토큰"이 아니라 "먼저 **영구
안정화**된 토큰 위치"가 의미 있는 신호다. 이 신호로 libFuzzer harness 소스코드의 어느
부분이 모델이 확신하는 boilerplate/lifecycle이고 어느 부분이 불확실한 input-decoding/
target-invocation 로직인지 구분해, 불확실한 부분만 골라 remask·재생성하면서 fuzzing
피드백(coverage, target reach)으로 유도하자는 것이다.

**이것은 이 저장소의 기존 작업과 축이 다르다.** 지금까지 `dllm_agent`가 하는 일은:

- `targets/*/particle.py` + `experiments/multitarget/slot_sampler.py`: **퍼징 대상 입력
  바이트**를 고정 grammar 위 categorical slot-fill로 생성(구조적 헤더는 애초에 템플릿,
  모델은 10개 이하의 정수 selector만 채운다).
- `revision/`: 그 particle의 selector를 selective하게 remask·재생성(PoC 개선).
- harness(`tools/openexr_runtime/poc_probe.cc` 등) 자체는 **사람이 작성하고 SHA-256으로
  pin한 고정 소스**다. 모델이 손대는 대상이 아니다.

이 제안은 그 반대다 — **harness 소스코드 자체**를 모델이 생성/개선하는 대상으로 삼는다.
따라서 새 연구 축이며, 기존 openexr_one_day 50-cell 캠페인이나 revision 파이프라인을
건드리지 않는 독립 모듈로 설계해야 한다(`docs/superpowers/specs/2026-09-14-ar-parent-control-arm.md`가
AR 비교군을 기존 캠페인 밖에 둔 것과 같은 원칙).

## 2. 핵심 연구 질문 (RQ)

GPT 초안의 세 질문을 그대로 채택한다 — 이미 잘 분해되어 있고 비교군 설계까지 포함한다:

- **RQ1 (상관관계)**: 일찍 영구 안정화된(permanence-early) 토큰 위치가 실제 harness
  correctness(컴파일 성공, target reach, API 오용 없음)와 상관관계가 있는가?
- **RQ2 (효율)**: 고신뢰 영역을 고정하고 저신뢰 영역만 재생성하면, 동일 NFE(forward call)
  예산에서 전체 재생성 대비 성공률/coverage-per-NFE가 개선되는가?
- **RQ3 (피드백 유도)**: fuzzing 피드백(coverage, target reach)으로 remask 위치를 고르면
  random remasking보다 나은가?

## 3. 선행연구 근거

`docs/research-report.md`가 이미 지키는 원칙(claim마다 `[MEASURED]`/`[INFERRED]`/
`[FUTURE EXPERIMENT]` 태그 + 번호 인용)을 그대로 따른다. 조사한 문헌:

**dLLM confidence/trajectory 신호가 실제로 의미 있다는 근거**

- [1] Ye *et al.*, "Dream 7B: Diffusion Large Language Models," arXiv:2508.15487, 2026.
  https://arxiv.org/abs/2508.15487 (accessed 2026-09-15) — Dream이 이미 `maskgit_plus`
  (top-1 confidence), `topk_margin`(top1−top2), `entropy` 세 가지 confidence 기반
  remasking 전략을 native로 지원함을 확인. GPT 초안의 "재학습 없이 hook만으로 가능하다"는
  전제가 이 논문 기준으로는 맞다.
- [2] "Dream-Coder 7B: An Open Diffusion Language Model for Code," arXiv:2509.01142, 2026.
  https://arxiv.org/html/2509.01142v1 (accessed 2026-09-15) — 실제 코드 생성 대상 모델.
- [3] "Measuring Temporal Linguistic Emergence in Diffusion Language Models,"
  arXiv:2604.23235, 2026. https://arxiv.org/pdf/2604.23235 (accessed 2026-09-15) —
  **RQ1의 직접 근거**: 매 denoising step마다 위치별 top-1 confidence/entropy/hidden state를
  기록해 분석한 결과, 최종적으로 옳은 토큰은 평균 confidence 0.877, 최종적으로 틀린 토큰은
  0.819로 유의미하게 갈린다고 측정. "denoising clock"이 수렴 여부를 구분한다는 것도 이
  아이디어의 permanence_step 개념과 정확히 같은 축.
- [4] "Read the Trace, Steer the Path: Trajectory-Aware Reinforcement Learning for Diffusion
  Language Models," arXiv:2606.04396, 2026. https://arxiv.org/pdf/2606.04396
  (accessed 2026-09-15) — confidence/entropy/"top 토큰이 step 간 안정적이었는가"를 feature
  vector로 묶는 선례. TokenTrace 설계와 사실상 동일한 feature set.
- [5] "The Path Matters: Learning a Token-Commitment Policy for Diffusion Language Models"
  (TraceLock), arXiv:2605.24697, 2026. https://arxiv.org/html/2605.24697v1
  (accessed 2026-09-15) — trajectory를 "어느 토큰을 언제 커밋할지" 정책 학습 문제로 정식화.
- [6] "Efficient Diffusion LLMs via Temporal-Spatial Parallel Decoding and Confidence
  Extrapolation," arXiv:2605.30753, 2026. (accessed 2026-09-15) — confidence/entropy/momentum
  기반 correctness sensor로 디코딩을 제어.
- [7] "The Bitter Lesson of Diffusion Language Models for Agentic Workflows: A Comprehensive
  Reality Check," arXiv:2601.12979, 2026. https://arxiv.org/pdf/2601.12979
  (accessed 2026-09-15) — **균형을 위한 회의적 인용**: dLLM의 agentic/코드 작업 우월성에
  의문을 제기하는 논문. `research-report.md`의 "우월성을 전제하지 않는다" 원칙과 맞춰
  최종 보고서에서 반드시 같이 인용해야 한다.

**harness 생성 품질/AST/coverage-feedback 근거**

- [8] "Quality-Assured Fuzz Harness Generation via the Four Principles Framework"
  (QuartetFuzz), arXiv:2605.21824, 2026. https://arxiv.org/html/2605.21824v1
  (accessed 2026-09-15) — **F(n)/fitness 함수의 직접 근거**: "컴파일만 되는 harness"의
  false-positive율이 최대 94%까지 나온다고 측정하고, Logic Correctness/API Protocol
  Compliance/Security Boundary Respect/Entry Point Adequacy 4원칙의 소스-레벨 정의를 제시.
  QuartetFuzz 자체는 4.8% false-positive까지 낮춤. 이 저장소가 "컴파일 성공만으로 좋은
  harness라 주장하지 않는다"는 GPT 초안의 지적을 문헌으로 뒷받침한다.
- [9] Google, "Fuzz target generation using LLMs," OSS-Fuzz research page.
  https://google.github.io/oss-fuzz/research/llms/target_generation/
  (accessed 2026-09-15) — 실전 baseline(OSS-Fuzz-Gen). CVE-2024-9143(OpenSSL) 발견 사례
  포함, 하지만 "generic하고 잘못된 입력 제약으로 false positive가 잦다"는 한계도 명시.
- [10] "HarnessAgent: Scaling Automatic Fuzzing Harness Construction," arXiv:2512.03420,
  2026. https://arxiv.org/pdf/2512.03420 (accessed 2026-09-15) — GPT 초안이 이미 인용한
  "논리적으로 쓸모없는 harness" 문제의 원 출처 확인.
- [11] "Thinking More, Harnessing Better: Automatic Harness Generation with Dataflow
  Aggregation and Workflow Decomposition," arXiv:2607.07007, 2026.
  https://arxiv.org/html/2607.07007 (accessed 2026-09-15) — AST/dataflow 기반 harness
  워크플로 분해 선례. `rank_mutable_regions`/AST 영역 분리 설계의 참고선례로 인용.

**태그**: [1]–[6]은 [MEASURED]/[INFERRED] — confidence-trajectory 신호가 존재하고 유의미
하다는 근거이지, "이 신호가 fuzzing harness 품질을 예측한다"는 것 자체는 아직
[FUTURE EXPERIMENT]다(RQ1이 검증 대상). [7]은 그 반대 방향 근거로 병기해야 한다. [8]–[11]은
harness 품질 정의·기존 baseline의 근거이며, dLLM과 무관하게 성립하는 사실이다.

## 4. 저장소 대조 — 이미 있는 것 / 없는 것

| 구성요소 | 이미 있음 | 상태 |
|---|---|---|
| 위치별 confidence를 담는 해시-바운드 trajectory 레코드 | ✅ `experiments/multitarget/slot_sampler_contracts.py`의 `TrajectoryStep`(`step_index`, `committed_positions`, `confidences`, `state_sha256`) | **다른 용도용**: `GenericSlotSampler`는 10개 이하 categorical selector를 한 번씩만 커밋하고 절대 재방문하지 않는다(`_fill`이 `remaining`에서 지운 position은 다시 안 돌아옴) — "permanence"가 자명하게 항상 1이라 이 아이디어가 필요로 하는 재커밋/흔들림 이력이 없다. |
| free-form 코드 토큰에 대한 다단계 remask(같은 생성 내에서 토큰이 여러 번 바뀜) | ❌ 없음 | `GenericSlotSampler`도 `ar_slot_sampler.py`(AR, 순차 1회 생성)도 이 패턴이 아니다. **새로 만들어야 하는 핵심 메커니즘.** |
| Dream 계열 모델의 step-level logits/token hook | ❌ 없음 | `runtime/revision_process.py`의 `load_torch_runtime`/`load_transformers_imports`가 lazy-import 관례를 제공하지만, 실제 model.generate 루프에 hook을 꽂는 코드는 없다(`tools/stable_diffcoder_generation.py`는 완성된 output만 받는다). |
| AST 파싱/역할 태깅 | ❌ 없음 | tree-sitter/Clang AST 의존성이 이 저장소에 전혀 없다(`pyproject.toml` 확인). |
| compile+libFuzzer 실행+coverage 측정 인프라 | ✅ 부분적으로 있음 | `targets/openexr/probe.py`/`probe_coverage.py`의 Docker 기반 compile/run/coverage 파이프라인이 있지만 **고정 harness 전제**다 — harness 소스 자체가 매 iteration 바뀌는 시나리오는 새로 배선해야 한다(이미지 재빌드 없이 harness .cc만 바꿔 컴파일하는 경로가 없음). |
| Fitness/피트니스 다항목 채점 | ✅ 유사 패턴 있음 | `revision/trajectory_policy.py`의 `_field_score` 휴리스틱이 있으나 particle selector용. harness 채점(compile/target-reach/coverage/diversity/crash)은 새로 설계해야 한다. |

**결론**: "trajectory 기록"이라는 *패턴*은 이 저장소의 확립된 관용구지만, 이 제안이
요구하는 *메커니즘*(free-form 코드에 대한 다단계 confidence-guided remasking + AST 역할
매핑 + harness별 재컴파일)은 전부 새로 만들어야 한다. Docker/해시-pin/append-only ledger
철학은 재사용 가능하지만 코드 재사용은 거의 없다.

## 5. 기각한 항목 (지금 단계에서 만들지 않음)

`docs/plans/2026-denofuzz-scaling-implementation-plan.md`의 선례(오버엔지니어링 기각
테이블)를 따른다:

| 항목 | 기각 사유 |
|---|---|
| RL 기반 remask 정책 학습([5] TraceLock류) | MVP 단계에서 학습 신호(harness 실행 횟수)가 턱없이 부족하다. GPT 초안도 "RL로 갈 필요 없음, hill climbing/beam search면 충분"이라 명시 — 동의. |
| 다중 타겟(libpng/libtiff/libxml2 등) 동시 지원 | MVP는 단일 타겟(OpenEXR, 기존 harness 소스가 이미 있어 "정답"과 diff 비교 가능)으로 좁힌다. |
| 4개 모델(Stable-DiffCoder/Dream-Coder/Seed-Coder/Qwen2.5-Coder) 전부 지원 | GPT 초안 본인이 "Dream-Coder 한 모델만"을 MVP로 제안 — 동의. Dream-Coder는 step hook을 네이티브로 지원하는 유일한 후보([1]); Stable-DiffCoder/Seed-Coder는 자체 sampler 내부 hook 삽입이 필요해 2단계로 미룬다. |
| tree-sitter/Clang 풀 AST 파서 도입 | MVP는 `LLVMFuzzerTestOneInput` 본문 하나만 다루므로, 정규식/괄호 매칭 기반 statement 경계 탐지로 충분 — 풀 AST는 다중 영역 세분화가 필요해지는 2단계에서. |
| 논문급 통계(공정한 N, power analysis) | `research-report.md` §5 원칙 그대로: 표본수는 이 MVP의 관측 분산에서 나온 power calculation으로 정해야 하며, 지금 임의로 정하지 않는다. |

## 6. MVP 스코프

GPT 초안의 "가장 현실적인 MVP" 절을 그대로 채택하되, 이 저장소의 기존 인프라에 맞춰
구체화한다:

- 모델: Dream-Coder-v0-Instruct-7B 1개만(`docs/superpowers/specs/2026-09-14-ar-parent-control-arm.md`가
  이미 이 모델을 "2번째 dLLM"으로 다운로드 대상에 넣어뒀다 — 재사용).
- 타겟: OpenEXR 1개, `LLVMFuzzerTestOneInput` 본문만(기존 `tools/openexr_runtime/poc_probe.cc`가
  "정답" 참조로 이미 있어 diff·plateau 비교가 가능하다는 이점).
  - **주의**: `tools/openexr_runtime/Dockerfile`은 `poc_probe.cc`를 `COPY`로 이미지에
    고정 굽는다 — harness가 iteration마다 바뀌는 MVP는 이 이미지를 재사용할 수 없다.
    별도의 "harness가 볼륨 마운트로 주입되는" 경량 컴파일 컨테이너가 필요하다(§7).
- 저신뢰 영역 1개만 remask(§5의 "다중 영역" 기각과 일관).
- 5분 libFuzzer coverage 비교(기존 `qualification-smoke`가 이미 쓰는 300초 예산과 동일 —
  새 예산 상수를 만들지 않고 재사용).
- 비교군은 GPT 초안의 8개 중 **3개로 좁힌다**(§5 원칙): Full regeneration, Confidence
  freezing(margin/entropy 없이 top-1만), Coverage-guided remasking(제안). Random remasking은
  통계적 null이라 결국 필요하지만 1단계 스모크에서는 생략 — 3-arm 결과가 방향성을 보이면
  2단계에서 Left-prefix/Stability/AST-aware/AR-baseline까지 확장.

## 7. 모듈 설계

기존 저장소 관례(도메인별 최상위 패키지: `revision/`, `native/`, `hunt/`)를 따라 새
최상위 패키지 하나로 넣는다. GPT 초안의 `dllm_agent/fuzzing/`은 **이미 다른 용도로 존재**
(`src/dllm_agent/fuzzing/coverage/` 등, coverage-instrumentation 관련)하므로 이름
충돌을 피해야 한다:

```
src/dllm_agent/harness_synthesis/          # 신규 최상위 패키지
├── __init__.py
├── trajectory_hook.py     # Dream generate() 루프에 step별 logits hook 삽입
├── token_trace.py         # TokenTrace dataclass + permanence_step() (기존 slot_sampler_contracts.py의
│                           # TrajectoryStep과 자매 타입 — 필드명 스타일 맞춤, 상속하지 않음:
│                           # 용도가 근본적으로 다름(§4))
├── regions.py             # 정규식/괄호-매칭 기반 statement 경계 탐지(MVP는 AST 아님, §5)
├── remask.py              # 저신뢰 영역 선택 + mask 삽입 + Dream 재생성 호출
├── harness_compile.py     # 볼륨 마운트 기반 harness 재컴파일 Docker 실행(§6의 신규 컨테이너)
└── fitness.py             # compile/target-reach/coverage/diversity 다항목 채점

src/dllm_agent/targets/openexr/
└── harness_reference.py   # 기존 poc_probe.cc 파싱·경계 추출(diff 비교용, 신규 파일)

tools/openexr_runtime/
└── harness-compile-only/  # Dockerfile 변형: poc_probe.cc를 COPY가 아니라 볼륨 마운트로 받는
                            # 경량 컴파일 전용 이미지(신규)

docs/superpowers/specs/
└── 2026-09-15-trajectory-guided-harness-synthesis.md   # 이 문서

configs/
└── openexr-harness-synthesis-smoke.json  # MVP pilot config(신규)
```

`DiffusionBackend` Protocol은 GPT 초안 그대로 채택하되, 기존 `revision/controller_models.py`의
`RevisionSampler` Protocol과 이름·스타일을 맞춘다(새 프로토콜 이름: `TraceableGenerationBackend`).

## 8. Fitness 함수

GPT 초안의 가중합을 초기값으로 채택하되, 가중치 자체는 **하드코드하지 않고 config로
뺀다**(이 저장소의 다른 모든 임계값이 config-driven인 것과 일관):

```
R = w_compile * Compile
  + w_reach    * TargetReach
  + w_edge     * EdgeCoverage
  + w_diversity* PathDiversity
  - w_timeout  * Timeout
  - w_bug      * HarnessBug
```

MVP 3-arm 비교에서는 `HarnessBug`/`PathDiversity` 측정기가 아직 없으므로(§5, §6) 초기
가중치는 `w_bug=0, w_diversity=0`으로 시작하고 0단계 결과에서 필요성을 재평가한다 — 값을
추측해서 넣지 않는다.

## 9. 검증 계획

- 신규 코드는 이 저장소 관례대로 fake/protocol 기반 유닛 테스트 먼저(`ar_slot_sampler.py`
  선례와 동일) — 실제 Dream-Coder 가중치 없이 `permanence_step()`/`regions.py`/`fitness.py`
  로직을 검증.
- 실제 Dream-Coder 로드 검증은 워크스테이션에서 별도 스모크(이미 두 세션째 진행 중인 모델
  다운로드에 포함되어 있음).
- RQ1 상관관계는 순수 관측 통계(실행 개입 없음) → RQ2/RQ3는 3-arm 개입 비교.
- `docs/research-report.md` §5 원칙대로 표본수는 이 MVP의 관측 분산에서 power calculation.

## 10. 남은 리스크

- **AST 없는 statement 경계 탐지의 견고성**: 괄호 매칭이 C 매크로/문자열 리터럴에서
  깨질 수 있다. MVP 타겟 harness가 단순해 실무적으로는 괜찮을 가능성이 높지만 확인 안 됨.
- **harness 볼륨 마운트 컴파일 경로의 해시 감사 체인**: 기존 파이프라인은 harness가
  이미지에 `COPY`돼 있어 이미지 digest 하나로 harness identity가 고정된다. harness가
  매 iteration 바뀌면 감사 원장에 harness SHA-256을 매 iteration 새로 기록해야 하고,
  `docs/paper-abstract-and-evaluation-claims.md`가 요구하는 "append-only JSONL audit
  ledger" 원칙과 맞춰 설계해야 한다 — 이 문서는 그 설계를 아직 안 했다(다음 단계).
  - 이건 `qualification.py`가 이미 겪은 문제(고정 harness 가정)를 다시 겪는 것이므로,
    구현 착수 전에 별도 검토가 필요하다.
- **Dream-Coder의 hook 삽입 지점이 문서화돼 있는지 불확실** — [1]/[2] 논문은 remasking
  전략을 설명하지만, `transformers`의 실제 `generate()` 구현에서 step별 logits를 어떻게
  가로채는지는 실제 코드(`modeling_dream.py`)를 받은 뒤 확인해야 한다(아직 다운로드 중).

## 11. 다음 단계

1. Dream-Coder 다운로드 완료 → `modeling_dream.py` 실제 확인, hook 삽입 지점 검증.
2. §10 감사-체인 리스크에 대한 결정(harness identity를 어떻게 pin할지) — 사용자 확인 필요.
3. 확정되면 `superpowers:writing-plans` 형식의 task-by-task 구현 계획 별도 작성.
