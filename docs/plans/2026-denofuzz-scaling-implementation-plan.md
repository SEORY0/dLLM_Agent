# DenoFuzz Scaling Implementation Plan

**작성:** 2026-09-04, Asia/Seoul
**상태:** 승인 대기 — 체크포인트 1/2 질의응답 완료, 구현 착수 전
**전제:** 이 문서는 새 아키텍처 제안이 아니다. 저장소 조사 결과 핵심 메커니즘(non-local
committed-token 재마스킹, 2×2 factorial 실험 설계)은 이미 구현·사전등록되어 있다. 이 계획은
① 그 위에 얇게 얹는 두 개의 최소 확장(search tree 일반화, 영속 failure registry)과 ② 이미
설계된 실험을 실제로 돌려 anytime discovery curve와 crossover budget을 관측하는 실행
계획이다.

## 0. 조사 근거 요약

| 주장 | 근거 |
|---|---|
| Non-local committed-token 재마스킹은 이미 구현·검증됨 | `tools/selective_revision_sampler.py:88-124`(`revise()`), `docs/dllm-native-online-portfolio-harness.md:121-153` |
| 현재는 tree가 아니라 lineage당 active scout 1개인 hill-climbing | `src/dllm_agent/revision/archive.py`(parent∪child Pareto 후 승격), harness 문서 한계 항목 3 |
| Tri-modal 실패 신호(invalid/duplicate/proof_failure)는 이미 존재, 영속화만 안 됨 | `src/dllm_agent/revision/trajectory_policy.py:43-52`(`LineageFieldOutcome`), 한계 항목 4·6 |
| Gradient-free heuristic guidance(FUDGE류)는 이미 존재 | `trajectory_policy.py:155-175`(`_field_score`, `_outcome_pressure`) — 손으로 짠 가중치(`*2.0`, `*0.25`) |
| GPU 상주 로드 실측치 | allocated 16.50GB / reserved 16.64GB, RTX 3090 24GB (`docs/research-report.md:80`) |
| 2×2 factorial (backbone×scaffold×budget×information) 설계는 이미 사전등록됨 | `docs/research-report.md` §8.1, `docs/experiment-protocol.md` D절 |
| Docker 격리 경계(non-root, read-only, network-none, cap-drop ALL)는 이미 전역 적용됨 | `src/dllm_agent/targets/openexr/probe.py:39`, `probe_coverage.py:39`, `proof_execution.py:43`, `fuzzing/execution.py:130-133` |

## 1. Executive Summary

### 1.1 가설 H1–H5

H1–H4는 `docs/research-report.md` §2.2와 `docs/experiment-protocol.md` H절에 이미 사전등록되어
있다(candidate-bank trace priority, dLLM mechanism vs AR, locality moderation, analyzer-guided
repair). 이 계획은 **H5 하나만 신규로 추가**한다.

| ID | 가설 | 판정 |
|---|---|---|
| H5 | 동일 NFE/GPU-초 예산에서, 다중 조상 재방문이 가능한 search-tree portfolio(신규)가 현재의 lineage당 단일 active-scout hill-climbing(기존 `online-portfolio`)보다 `validated_success@1` 또는 `time-to-first-proved-cause`를 개선한다. H0: 차이 없음. | point estimate ≥ 기존 SESOI 관례(5점)와 동일 기준 적용, paired lower 95% CI > 0이면 지지. `single-scout` ablation을 comparator로 둔다(기존 필수 ablation표에 신규 행 추가). |

### 1.2 Crossover budget B*

Budget 축은 기존 §8.1 정의를 그대로 쓴다: $B = (C_{GPU}, C_{validator}, T_{wall})$. dLLM 후보
경로는 모델 상주·재마스킹 forward pass 오버헤드 때문에 저예산 구간에서는 AR-equal-budget
baseline(AR-FIM/구조화 local edit, 이미 §D 표에 정의됨)보다 뒤처질 수 있다. **B\*** 는 두 arm의
`time-to-first-proved-cause` 누적곡선(anytime discovery curve)이 교차하는 budget 지점으로
정의한다: $B^* = \min\{B : \mathbb{E}[\text{proved\_causes}_{dLLM}(B)] \geq \mathbb{E}[\text{proved\_causes}_{AR}(B)]\}$.
B\*가 pilot 규모(현재 `particle_count=4`) 안에서 관측되지 않으면 "B\* 미관측"으로 명시적으로
보고하며, 존재를 가정하지 않는다.

이 두 관측(H5, B\*)은 **explorаtory**이며 §8.3의 preregistered 통계 gate를 새로 발명하지 않고
기존 H1 paired-CI 절차를 그대로 재사용한다.

## 2. 기각된 오버엔지니어링 항목

| 항목 | 기각 사유 |
|---|---|
| Gradient/backprop 기반 Energy·Classifier Guidance (Diffusion-LM식, 외부 컴파일러 대상) | 컴파일러/fuzzer가 미분 불가능이라 surrogate 없이는 gradient가 존재하지 않는다. Surrogate 학습 인프라는 현재 런 규모(4 particle)의 데이터로 정당화 안 됨. §3.3에 gradient-free 대안만 남긴다. |
| Diffusion-LM식 hidden-state 학습 분류기 guidance | 학습 데이터 수집 루프가 새로 필요하고 현재 규모로는 과적합 위험이 label 수보다 큼. |
| FuzzGuard식 reachability 분류기 학습 | 45개 실취약점 데이터셋 규모에서 나온 결과이며, 본 프로젝트 규모(런당 4 particle, 단일 target)에서는 학습 신호가 부족하다. |
| Reflexion식 verbal(자연어) episodic memory | 이 저장소의 dLLM 경로는 대화형 루프가 아니라 고정 grammar template 위 slot-fill이다. 재주입할 프롬프트 맥락이 없어 아키텍처에 안 맞는다. |
| Trie 기반 실패 메모리 | 실패 signature 카디널리티가 필드 5개×cause class 수십 개 수준으로 작아 접두사 압축 이득이 없다. |
| In-memory Bloom filter | 중복 판정 대상 수가 프로젝트 규모에서 수천 단위를 넘지 않아 근사 자료구조가 필요 없다. 기존 `proof_archive.py`의 exact SHA-256 canonical hash로 충분하다. |
| SQLite | 저장소 전역이 이미 JSONL append-only + SHA-256 패턴을 쓴다(`run_manifest.json`, `trajectory_event` 등). 새 DB 엔진을 도입할 근거가 없다. |
| Graph DB (지식 축적 하네스) | 조회 패턴이 단순 key(target+field+cause)→집계이며 그래프 순회가 필요한 질의가 없다. |
| Magma 벤치마크 포팅 | 여러 프로젝트에 forward-ported CVE oracle을 통합하는 공수가 현재 연구질문(H1–H5, trace 기반 우선순위화 검증)에 비해 과도하다. 기존 pinned OpenEXR v3.4.14 + jsoncons CWE-121 fixture로 §H절의 executable oracle 요건을 이미 충족한다. |

## 3. 인터페이스 스키마

세 스키마 모두 **신규 하위 모듈이 아니라 `src/dllm_agent/revision/` 패키지 확장**이다
(`docs/source-layout.md`의 ownership 규칙: revision은 "approved feedback-conditioned revision
schemas ... deterministic controller orchestration"을 소유). 기존 타입(`RevisionResult`,
`LineageFieldOutcome`, archive의 Pareto 축)을 재사용하고 중복 정의하지 않는다.

### 3.1 `DenoNode` — `src/dllm_agent/revision/search_tree.py` (신규 파일)

```python
class DenoNode(FrozenModel):
    """One materialized snapshot in the revision search tree."""

    node_id: str                    # sha256(parent_id + field + selected_token_ids)
    parent_id: str | None           # None only for the protected initial snapshot
    lineage_id: str
    depth: int = Field(ge=0)
    created_epoch: int = Field(ge=0)
    snapshot_sha256: str            # existing materialized-candidate hash, not recomputed
    pareto_metrics: ParetoMetrics   # reuse archive.py's existing metric tuple, no new fields
    status: Literal["frontier", "expanded", "pruned", "archived"]
```

`node_id`는 결정적 해시로, 같은 (parent, field, 선택 token) 조합은 항상 같은 id를 낸다 — 기존
`request_hash`/`canonical hash` 관례를 그대로 따른다(`revise()` 캐시 키와 동일 패턴,
`tools/selective_revision_sampler.py:90-93`).

### 3.2 `SearchAction` — 같은 파일

```python
class SearchAction(FrozenModel):
    """One proposed remask-and-revise transition from a frontier node."""

    action_id: str
    source_node_id: str
    field: FieldName                # reuse targets/openexr/particle.FieldName
    remask_positions: tuple[int, ...]
    result_node_id: str | None = None   # filled after the revise() call executes
```

`SearchAction`은 **실행자가 아니라 순수 데이터**다. 실제 remask/denoise는 여전히
`SelectiveRevisionSampler.revise()`가 수행한다 — 새 sampler를 만들지 않는다.

### 3.3 `FailureRegistry` — `src/dllm_agent/revision/failure_registry.py` (신규 파일)

```python
class FailureRecord(FrozenModel):
    """One persisted (target, field, cause) outcome — extends LineageFieldOutcome verbatim."""

    target_pin_sha256: str          # config target.commit_sha, ties record to a fixed oracle
    field: FieldName
    cause_bucket: Literal["invalid", "duplicate", "proof_failure"]  # maps 1:1 to
                                                                     # LineageFieldOutcome's
                                                                     # existing three counters
    count: int = Field(ge=1)
    reward_delta_sum: float
    trials: int = Field(gt=0)
    last_seen_run_id: str

def load(path: Path) -> tuple[FailureRecord, ...]: ...
def append(path: Path, record: FailureRecord) -> None: ...   # JSONL append, matches
                                                                # run_manifest.json conventions
def seed_outcomes(records: tuple[FailureRecord, ...], lineage_id: str) -> tuple[LineageFieldOutcome, ...]: ...
```

`seed_outcomes`는 다음 run 시작 시 `trajectory_policy.choose_lineage_field`에 넘길
`LineageFieldOutcome` 튜플을 만든다 — **`trajectory_policy.py`의 함수 시그니처를 바꾸지 않는다**,
호출부(`revision/pipeline.py`)만 영속 레코드를 미리 로드해 outcomes 인자에 합쳐 넣는다.
gradient-free guidance 업그레이드(§2 표의 유일 생존 B 항목)는 이 registry가 런을 여러 번
거쳐 충분한 표본(`trials` 합계 임계치, 추후 pilot 분산으로 확정)을 쌓은 뒤에만 착수하고, 그
전까지는 `_single_outcome_pressure`의 수동 가중치를 그대로 둔다.

## 4. Phase 계획

### Phase 1 — 얇은 search-tree 일반화 (H5 인프라)

- `revision/search_tree.py`: `DenoNode`/`SearchAction` 정의, 결정적 `node_id` 해시 함수.
- `revision/archive.py` 확장: "child가 active scout로 승격되면 나머지는 그냥 archive"하는 현재
  로직 옆에, `archive_size` 한도 안에서 **frontier 집합**(status="frontier" node 여럿)을
  유지하는 선택지를 추가한다. 기존 단일 scout 동작은 `frontier_width=1`인 특수 케이스로 남긴다
  — 회귀 없음.
- `revision/portfolio.py`: 다음 revision 대상을 active scout뿐 아니라 frontier 중에서 Pareto
  crowding-distance로 고를 수 있게 한다(백트래킹).
- **테스트:** `tests/test_poc_revision_search_tree.py` — `node_id` 결정성, frontier 선택이
  `frontier_width=1`일 때 기존 archive 테스트(`tests/test_poc_revision_archive.py`)와 동일 결과를
  내는지(회귀 고정), 다중 조상 백트래킹 단위 테스트.
- **검증:** `pytest tests/test_poc_revision_search_tree.py tests/test_poc_revision_archive.py
  tests/test_poc_revision_portfolio.py -v`, `basedpyright src/dllm_agent/revision/search_tree.py`.

### Phase 2 — Failure Registry 영속화

- `revision/failure_registry.py`: `FailureRecord`, `load`/`append`/`seed_outcomes`.
- `revision/pipeline.py` 호출부에서 run 시작 시 registry 로드 → `seed_outcomes` →
  `trajectory_policy` outcomes 인자에 병합. Run 종료 시 관측된 `LineageFieldOutcome`을
  `FailureRecord`로 변환해 append.
- 스키마 버전 필드(`schema_version: "dllm-failure-registry/v1"`)를 붙여 기존 manifest 관례를
  따른다.
- **테스트:** `tests/test_poc_revision_failure_registry.py` — round-trip append/load, cause_bucket
  ↔ 기존 `invalid_count`/`duplicate_count`/`proof_failure_count` 매핑이 정확한지,
  cross-run seeding이 §A.1 실행 중단 조건(예: hash-identical candidate 요구)을 깨지 않는지.
- **검증:** `pytest tests/test_poc_revision_failure_registry.py tests/test_poc_revision_trajectory_policy.py -v`,
  `basedpyright src/dllm_agent/revision/failure_registry.py`.

### Phase 3 — H5/B\* 실행 (신규 아키텍처 없음, 기존 §8.1 설계의 실행)

- Phase 1+2가 적용된 `search-tree` arm과, ablation 표(harness 문서)에 이미 정의된
  `single-scout`(=현재 `online-portfolio` 그대로) 및 §D절의 AR-FIM equal-budget arm을 동일
  target/container pin, corpus, seed로 나란히 돌린다.
- Budget sweep(작은 pilot 규모: `particle_count`/`epochs` 2~3단계)으로 `time-to-first-proved-cause`
  누적곡선을 그려 B\* 관측 시도. §1.2 기준대로 미관측이면 "미관측"으로 명시 보고.
- 이 phase는 **신규 코드가 거의 없다** — Phase 1/2 산출물을 기존 `poc-revision` CLI와
  `experiments/openexr_one_day` 캠페인 오케스트레이션에 새 config 파일(`configs/
  openexr-v3.4.14-search-tree-pilot.json`)로 연결하는 배선 작업이다.
- **검증:** 기존 §A.1 실행 중단 조건과 §H절 판정 gate를 그대로 적용. 새 통계 절차를 발명하지
  않는다.

## 5. 격리·보안 경계 유지 방안

- Phase 1/2/3 모두 **Docker 샌드박스 경계를 건드리지 않는다.** `SearchAction`/`DenoNode`/
  `FailureRegistry`는 전부 host 쪽 순수 데이터·해시 연산이며, target 빌드/실행은 여전히
  기존 `--network none --read-only --cap-drop ALL --user 65534:65534 --pull never` 경계
  안에서만 일어난다(`fuzzing/execution.py:130-133`, `targets/openexr/probe.py:39`,
  `proof_execution.py:43` 변경 없음).
- `FailureRegistry`는 새로운 third-party target onboarding이 아니다 — 기존 pinned target(OpenEXR
  `777d231a...`, jsoncons)의 결과만 기록하며, manifest의 `target_pin_sha256` 필드로 pin이 바뀌면
  기존 레코드가 자동으로 다른 키로 분리되어 교차 오염을 막는다.
- Registry 파일 자체는 host 파일시스템(`artifact_root` 하위)에 쓰고, sandbox 컨테이너 안으로
  마운트하지 않는다 — probe/proof 컨테이너의 read-only 경계를 유지한다.
- D(Magma)와 B(gradient guidance) 기각으로 인해 새 외부 네트워크 접근, 새 컨테이너 권한 상승,
  새 학습 파이프라인이 이 계획에 전혀 포함되지 않는다 — 공격 표면이 늘지 않는다.

## 6. 남는 것 — 명시적 future

- H5/B\*는 §8.3의 preregistered multiplicity/power 절차를 새로 만들지 않고 기존 H1 paired-CI
  절차를 재사용하는 **exploratory** 관측이다. Powered study로 승격하려면 기존 pilot
  hash-locked 분산 산정 절차를 그대로 거쳐야 한다.
- gradient-free guidance의 fitted-weight 업그레이드(§3.3 말미)는 registry에 충분한 표본이
  쌓이기 전까지 착수하지 않는다.
- 이 계획은 stock Stable-DiffCoder의 "자가 수정" 주장을 하지 않는다 — 여전히 이 저장소가
  구현한 project-owned selective-revision 경로에 대한 이야기다(`docs/dllm-native-online-portfolio-harness.md:68-70` 원칙 유지).
