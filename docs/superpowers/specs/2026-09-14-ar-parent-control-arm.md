# AR-parent control 샘플러 (RQ1 최소 비교기)

**상태:** 설계 확정, 구현 완료(`ar_slot_sampler.py`/`ar_model_loader.py`), 실기기 효능 측정 미실행
**범위:** `research-report.md` RQ1(§ "Stable-DiffCoder의 후보별 irreversible commitment
trace가 동일 계산량 AR confidence/self-consistency를 넘어 fixed-budget
validated-hypothesis prioritization을 개선하는가?")만 다룬다. §8의 전체
backbone×scaffold×budget×information factorial(H1–H5)은 여전히
`[FUTURE EXPERIMENT]`이며 이 문서의 범위 밖이다.

## 1. 왜 50-cell one-day campaign에 얹지 않았는가

`openexr_one_day`의 `ResourceMap`/`ModelArm`/wave 스케줄은 4개 dLLM arm(같은
Stable-DiffCoder, 다른 revision 전략) + 1개 non-generative baseline으로 스키마
레벨에서 고정돼 있다(`search_cores: min_length=10`도 `2 repetitions × 5 arms`에서
유도된 상수). AR arm을 그 캠페인에 6번째 arm으로 추가하면 wave 스케줄, resource
map cardinality, qualification gate, 관련 테스트 전부를 다시 고쳐야 하는 큰
blast radius가 생긴다. 반면 RQ1은 "같은 particle format·같은 검증 예산에서 후보
순위를 개선하는가"만 묻는다 — 50-cell 실전 캠페인이 아니라 독립 비교기로도 충분히
답할 수 있다. 그래서 기존 `GenericSlotSampler`(`slot_sampler.py`)는 건드리지 않고,
같은 template/catalog 유틸(`build_template`/`slot_token_ids`/`selectors`/
`sha256_tokens`, 전부 `slot_sampler_utils.py`)을 공유하는 별도의
`AutoregressiveSlotSampler`만 추가했다.

## 2. AR parent 모델 선택

Stable-DiffCoder-8B-Instruct의 HF 모델 카드: "Built directly on the Seed-Coder
architecture, data, and training pipeline"(block diffusion continual
pretraining) — 즉 **`ByteDance-Seed/Seed-Coder-8B-Instruct`가 바로 그 AR
parent**다. 같은 아키텍처·같은 tokenizer·같은 학습 데이터 계통이므로
`research-report.md` §8.2가 요구하는 "AR parent만으로는 계통 공정성은 높다"는
조건을 정확히 만족하는, 가장 방어 가능한 첫 control이다("strong independent AR"은
계통이 다른 별도 모델이 필요하므로 별도 future 작업으로 남긴다).

- Repo: `ByteDance-Seed/Seed-Coder-8B-Instruct`
- Revision: `782290842bc3fb0c03318d6c9f3be464b665f429`
- **[MEASURED]** Hub는 `License: mit`를 표시하지만 실제 file tree에는 LICENSE
  artifact가 없다 — Stable-DiffCoder에 이미 기록된 것과 동일한 패턴(`research-
  report.md` §3.2). 따라서 이 checkpoint도 "official Hub-labelled MIT,
  artifact-level weight license unresolved"로만 기술하고, 재배포·상업적 결론은
  보류한다.
- 커스텀 remote code 불필요(표준 `AutoModelForCausalLM`) — Stable-DiffCoder
  로더(`tools/stable_diffcoder_generation.py`, `trust_remote_code=True`)를
  재사용하지 않고 별도의 작은 로더(`ar_model_loader.py`)를 둔 이유이기도 하다.

## 3. 샘플러 설계 — 왜 `GenericSlotSampler`를 재사용하지 않았는가

`GenericSlotSampler`는 매 step마다 confidence로 정렬한 뒤 예산만큼 병렬
commit하는 diffusion reveal 스케줄이다. 이 스케줄을 AR 모델의 logit으로만
바꿔 재사용하면 "decoding paradigm"과 "reveal 순서 정책"이 뒤섞여 진짜 AR
control이 아니게 된다. 그래서 `AutoregressiveSlotSampler`는:

- 항상 **slot position 오름차순**으로, 한 번에 정확히 한 slot만 decode한다.
- 매 forward call은 `token_ids[:position]`만(즉 이미 commit된 접두사만) 본다 —
  아직 결정되지 않은 위치를 절대 들여다보지 않는다(`test_each_forward_call_
  only_sees_positions_already_committed`로 회귀 고정).
- greedy(허용된 catalog 내 최대 softmax 확률)로 선택한다. `seed`는 인터페이스
  parity를 위해 받지만 greedy decoding 자체는 결정론적이라 사용하지 않는다.
- revision/remask 개념이 없다 — AR parent는 "한 번에 전체 후보를 생성"까지만
  다루는 최소 control이다. selective-revision과 직접 비교하려면 별도 treatment로
  확장해야 한다(미래 작업).

## 4. NFE/FLOP 비대칭 — 반드시 함께 보고할 것

`SlotSample.forward_calls`/`nfe`는 dLLM arm에서는 "block-parallel forward
call" 수(여러 slot을 한 번에 해결 가능)를, AR arm에서는 "slot 하나당 forward
call" 수(=항상 mutable slot 개수)를 센다. **같은 필드 이름이지만 같은 단위가
아니다.** `research-report.md`가 요구하는 대로 `actual_forward_calls`, wall
time, GPU model을 원시값 그대로 나란히 보고하고, 절대로 가공 없이
"AR보다 몇 배 빠르다/느리다"는 결론으로 직접 환산하지 않는다.

## 5. 구현

- `src/dllm_agent/experiments/multitarget/ar_slot_sampler.py` — `ArTokenModel`
  프로토콜 + `AutoregressiveSlotSampler`(순수 파이썬, torch 불필요, 테스트 가능).
- `src/dllm_agent/runtime/ar_model_loader.py` — `HuggingFaceArTokenModel`.
  `revision_process.py`의 `load_torch_runtime`/`load_transformers_imports`와
  동일한 지연 `import_module` + `getattr` 검증 패턴(정적 `import torch`를 쓰지
  않음 — `gpu` extra 없이도 이 모듈이 import 가능해야 하고 strict basedpyright를
  통과해야 하기 때문). Stable-DiffCoder 전용 `ModelLike`(`generate_block` 등
  diffusion 전용 표면)는 구조적으로 맞지 않아 재사용하지 않고 표준 causal-LM용
  프로토콜을 새로 작게 정의했다.
- `tests/test_multitarget_ar_slot_sampler.py` — 3개, 전부 순수 파이썬 fake로
  실행(GPU/torch 불필요).

## 6. 아직 하지 않은 것

- 실제 Seed-Coder-8B-Instruct 가중치로 실행 검증(현재는 fake model 단위 테스트만).
- one-day campaign 또는 다른 실전 경로에 AR arm을 실제로 배선(의도적으로 범위 밖).
- "strong independent AR"(계통이 다른 모델) control.
- selective-revision과 AR parent를 동일 후보 은행·예산에서 직접 비교하는 실행
  스크립트(다음 단계).
