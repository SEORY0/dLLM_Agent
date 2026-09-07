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

이 두 관측(H5, B\*)은 **exploratory**이며 §8.3의 preregistered 통계 gate를 새로 발명하지 않고
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

**Cold/Warm start 격리 (2026-09-04 sign-off 보완).** `FailureRecord.load`를 호출할지 여부를
config에서 명시적으로 통제한다.

```python
type RegistryMode = Literal["cold", "warm"]
# RevisionConfig(...).registry_mode: RegistryMode = "cold"
```

`cold`(기본값)는 registry 파일이 있어도 무시하고 빈 튜플로 시작한다 — H1의 candidate-bank
trace-permutation control(§E, `docs/experiment-protocol.md`)이나 H5의 `single-scout` ablation처럼
"이전 run의 지식이 이번 run 결과를 오염시키지 않았다"를 보장해야 하는 벤치마크/판정 run은
`cold`를 강제한다. `warm`은 §3.3의 `seed_outcomes`를 실제로 적용하며, 운영/탐색용
online-portfolio 장기 실행에서만 쓴다. 두 모드 모두 `run_manifest.json`에
`registry_mode`를 기록해, 사후에 "이 run이 이전 지식을 봤는가"를 hash 근거로 검증할 수 있게
한다.

## 4. Phase 계획

### Phase 1 — 얇은 search-tree 일반화 (H5 인프라) — 구현 완료 (2026-09-04)

- `revision/search_tree.py`(신규): `DenoNode`(해시 identity만 보유, snapshot/trajectory
  payload 없음), `SearchAction`/`search_action_id`(결정적 `sha256(source_key+field+positions)`),
  `LineageFrontier`/`select_lineage_frontier`.
- `revision/archive.py`: **Pareto dominance/crowding 로직은 변경하지 않았다** — 이미
  `select_parent_child`가 임의 `capacity`와 `ArchiveCandidateKey=(particle_id, snapshot_sha256)`를
  받아 같은 lineage의 parent/child가 동시에 살아남는 경우를 지원하고 있었다
  (`tests/test_poc_revision_archive.py::test_parent_and_child_with_same_particle_id_remain_distinct_by_snapshot`
  로 기존에 이미 검증됨). 재사용을 공식화하는 한 줄짜리 순수 필터 헬퍼
  `lineage_candidates(records, lineage_id)`만 추가했다.
- `revision/portfolio.py`: `choose_backtrack_key(lineage_id, frontier, epoch)` 추가 — frontier
  튜플을 epoch로 round-robin해 백트래킹 대상을 고른다. `frontier_width=1`(단일 원소 튜플)이면
  항상 그 하나를 반환해 오늘의 단일 active-scout 동작과 동일하다.
- **Frontier pruning 규칙 (2026-09-04 sign-off 보완).** `select_lineage_frontier`가
  `select_parent_child`의 `selection.dropped`를 그대로 `status="pruned"` `DenoNode`로 변환한다.
  `archive_size`(=`frontier_width`) 초과분은 기존 crowding-distance 절단(`archive.py`의
  `_crowding_truncate`)이 이미 최하위 Pareto 후보부터 떨어뜨린다 — 별도 pruning 알고리즘을
  새로 만들지 않았다. "메모리 참조 해제"는 `DenoNode`가애초에 `ArchiveCandidateKey`(문자열 두
  개) 외에는 아무 것도 들고 있지 않도록 설계해 구조적으로 달성했다 — pruned 상태로 전환된
  노드를 호출부가 더 이상 `nodes` 튜플에 담지 않는 순간 `ProbeRecord`/snapshot 참조가 자연히
  해제된다. `tests/test_poc_revision_search_tree.py::test_pruned_nodes_carry_no_snapshot_payload`가
  `DenoNode`의 필드 집합이 `{key, parent_key, depth, status}`뿐임을 고정한다.
- **테스트:** `tests/test_poc_revision_search_tree.py`(신규, 12 케이스) — action id 결정성/필드
  민감성, `frontier_width=1`이 기존 archive 테스트 fixture와 정확히 같은 active/dropped 키를
  내는 회귀 증명, `frontier_width=2`에서 병존, 다른 lineage 격리, pruned 노드 payload 없음,
  backtrack round-robin과 빈 frontier fail-closed.
- **문서:** `docs/source-layout.md`(revision 패키지 파일 목록에 `search_tree.py` 추가),
  `tests/source_layout_support.py`(`FINAL_RUNTIME_MODULES`에 모듈 추가) — 저장소의 exact-inventory
  회귀 테스트(`tests/test_source_layout.py::test_current_module_inventory_is_exact_final_layout`)가
  요구.
- **검증 결과:** 아래 §7 참조.

### Phase 2 — Failure Registry 영속화 — 구현 완료 (2026-09-04)

- `revision/failure_registry.py`(신규): `FailureRecord`(target/field/cause_bucket당 1 레코드),
  `load`/`append`(atomic temp+rename, `core/io.write_model`과 동일 관례), `failure_records()`
  (관측된 `LineageFieldOutcome`을 bucket별로 explode), `seed_outcomes()`(영속 레코드를 target
  pin으로 필터링해 특정 lineage용 `LineageFieldOutcome`으로 역변환).
- `revision/models.py`: `RevisionConfig`에 `registry_mode`("cold"/"warm", 기본 cold),
  `failure_registry_path`, `target_pin_sha256` 추가 — 전부 optional/기본값 있음, 기존 config
  JSON 파일과 하위호환. warm인데 path가 없으면 validator가 fail-closed.
- `revision/pipeline.py`: `_run_online_selective`가 `load_seed_records`(registry_mode=="warm"일
  때만 실제 로드)로 얻은 레코드를 `choose_lineage_field` 호출 시 `state.field_outcomes`와 합쳐
  넘긴다(`trajectory_policy.py` 함수 시그니처는 변경 없음). `_run_poc_revision_async` 끝에서
  `persist_failure_records`가 이번 run의 관측 결과를 append — **run 자신의 mode와 무관하게**
  path만 설정돼 있으면 항상 기록(cold run도 다음 warm run에 기여). `run_poc_revision`의 public
  시그니처("Task 4가 고정")는 그대로 — target scoping은 이미 있는 `RevisionConfig` 객체를 통해
  전달된다. `_run_online_selective`가 PLR0915(문장 수 제한)를 넘겨 promote/drop 분기를
  `_apply_archive_decision`으로 추출(순수 리팩터, 동작 변화 없음).
- **문서:** `docs/source-layout.md`, `tests/source_layout_support.py`에 새 모듈 반영 — Phase 1
  때와 동일하게 이 exact-inventory 테스트를 빠뜨려 임시로 234→235 회귀를 냈다가 즉시 고쳤다
  (교훈: 새 revision 모듈을 추가할 때마다 이 두 파일을 빠짐없이 동기화할 것).
- **테스트:** `tests/test_poc_revision_failure_registry.py`(신규, 13 케이스) — round-trip
  append/load, 손상된 JSONL fail-closed, target-pin 격리, cause_bucket↔카운터 매핑, cold에서
  기존 파일이 있어도 빈 seed, warm에서 정확히 로드, path 없으면 no-op, 그리고 동일 seed를 두 번
  주입했을 때 `choose_lineage_field`가 결정론적으로(그리고 무-seed 대비 실제로 다르게) 선택함을
  확인하는 인과 테스트.
- **검증 결과:** 아래 §7 참조.

### Phase 3 — H5/B\* 실행 배선 — 구현 완료 (2026-09-04)

- `revision/models.py`: `RevisionConfig.frontier_width: int = 1`(`ge=1`) 추가. 기본값 1로 기존
  config 전부 하위호환.
- `revision/pipeline.py`: `_run_online_selective` 루프에 `_backtrack_parent`/
  `_lineage_frontier_keys`/`_archive_record` 3개 헬퍼를 추가해 Phase 1의
  `search_tree.select_lineage_frontier`와 `portfolio.choose_backtrack_key`를 실제로 호출한다.
  - `_lineage_frontier_keys`는 `frontier_width<=1`이면 **`select_lineage_frontier`를 아예
    호출하지 않고** `(active record의 key,)` 단일 튜플만 반환한다 — 재파생이 아니라 구성 자체로
    오늘의 단일 active-scout 선택과 동일함을 보장한다. `frontier_width>1`일 때만
    `select_lineage_frontier`로 archive에서 같은 lineage의 다른 non-dominated 후보를 추가로
    가져와 active-key 뒤에 덧붙인다.
  - `_backtrack_parent`가 `choose_backtrack_key(particle_id, frontier_keys, epoch)`로 이번
    epoch의 실제 확장 지점을 고르고, `revise_unique`에 넘기는 parent, cost lookup key,
    reversion target(`parents[child_id]`)을 전부 그 선택으로 치환한다. 이게 real backtracking이다
    — 이전에는 `active[particle_id]`만 계속 확장했다.
  - 기존 `active`/archive 승격·강등 로직(`_apply_archive_decision`)은 **전혀 건드리지 않았다**
    — 백트래킹은 "이번 epoch에 누구를 확장할지"만 바꾸고, 결과 Pareto 판정은 그대로다.
- **테스트로 실제 검증:** `tests/test_poc_revision_pipeline.py`의 기존
  `test_online_archive_retained_incomparable_child_is_scheduled_next` fixture를 그대로 재사용해
  두 개 신규 테스트를 추가했다.
  - `frontier_width=1`: 2번째 revision이 여전히 promoted child에서 확장됨(기존 동작과 동일).
  - `frontier_width=2`: 같은 시나리오에서 2번째 revision이 **archived parent로 백트래킹**함
    (round-robin `epoch % len(frontier)`가 예측한 그대로) — 이게 mock-run 수준의 "에러 없이
    호출" 검증이자, backtracking이 실제로 관측 가능한 파이프라인 산출물을 바꾼다는 인과 증거다.
- **Config 2종:** `configs/openexr-v3.4.14-search-tree-pilot.json`(`frontier_width:4`,
  `registry_mode:"cold"`, `archive_size:16`=particle_count(4)×frontier_width(4)),
  `configs/openexr-v3.4.14-search-tree-warm-pilot.json`(동일 + `registry_mode:"warm"`,
  `failure_registry_path:"artifacts/openexr-failure-registry.jsonl"`). 두 config 모두
  `target_pin_sha256`은 `sha256(target.commit_sha)`로 채웠다(cli.py 자동 유도는 Phase 2에서
  보류한 결정 그대로 — config author가 명시).
- **텔레메트리 점검 결과 (신규 구현 없음, 조사만):** `PocRevisionReport`/`RunManifest`에
  `time-to-first-proved-cause`나 `validated_success@1` 같은 이름의 필드는 **없다**. 이 저장소는
  `RevisionEvent`를 의도적으로 timestamp-free하게 설계했다(`models.py` 자체 docstring: "Timestamp-free
  event row"). 대신 이미 존재하는 `RevisionEvent.sequence`/`nfe`/`forward_calls`/`gpu_ms`와
  `online_proof_status`(`OnlineProofVerdict`에도 원인 클래스 포함)를 조합하면 **NFE-indexed**
  anytime-discovery curve(wall-clock이 아니라 "누적 NFE 대비 최초 proved-cause 시점")를 사후
  계산할 수 있다 — 이게 이 저장소의 재현성 원칙과 맞는 올바른 축이다. `RuntimeAccounting`도
  `recorded_nfe`/`recorded_forward_calls`/`recorded_gpu_ms`/`probe_invocations`/
  `exact_proof_invocations`를 이미 report에 싣는다. **새 스키마 필드나 분석 스크립트는 이번
  phase에서 추가하지 않았다** — 실제 GPU pilot 데이터 없이 분석기를 먼저 만드는 건 이 저장소가
  스스로 경계하는 "미측정 주장" 패턴이기 때문이다. B\* 관측 시도는 실제 실행 뒤 이 기존 필드들로
  집계하고, sanitizer finding이 전혀 없으면(과거 OpenEXR PoC pilot의 `NO-GO`처럼) "B\* 미관측"으로
  fail-closed 보고한다 — 이 규칙은 코드가 아니라 §1.2에 이미 사전등록돼 있으므로 별도 구현이
  필요 없다.
- **검증:** 아래 §7 참조.

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
- **Failure registry 압축(compaction) 보류**: 현재 registry는 append-only라 같은
  `(target_pin_sha256, field, cause_bucket)` 조합도 실행마다 새 줄로 계속 쌓인다. 실제 GPU
  pilot을 여러 차례 돌려 파일 크기가 실질적으로 문제가 되는 걸 관측하기 전까지는 동일 키
  레코드를 합산하는 압축 로직을 만들지 않는다(YAGNI) — §8에서 배치 쓰기(`append_many`)로
  런당 파일 재작성 횟수는 이미 줄여뒀다.

## 7. 검증 결과 (Phase 1–3)

세 phase 모두 Windows 네이티브 Python이 아니라 WSL Ubuntu에서 검증했다 — 기존
`experiments/openexr_one_day/cell_runner.py`가 POSIX 전용 `resource` 모듈을 최상위에서
import하기 때문에 native Windows에서는 이 저장소의 revision 테스트 대부분이 애초에
수집(collect)조차 되지 않는다(Phase 1 이전부터 있던 제약이며, 이 계획의 범위 밖). WSL은
저장소가 실제로 겨냥하는 Linux/Docker 런타임과 일치한다.

| 항목 | Phase 1 (`4b4e27a`, `25de070`) | Phase 2 (`87fb9ce`) | Phase 3 |
|---|---|---|---|
| 신규 테스트 | 11개, 전부 통과 | 13개, 전부 통과 | 4개, 전부 통과(pipeline 2 + config 역직렬화 2) |
| 전체 `pytest` 회귀 | byte-identical(기존 무관 실패 234개, 에러 6개) | byte-identical(동일 234/6) | byte-identical(동일 234/6) — 세 번 모두 `git stash`로 diff 양방향 확인 |
| `basedpyright`(전체, `typeCheckingMode=all`) | 0 errors | 0 errors | 0 errors |
| `ruff check .` | 변경 파일 전부 통과 | 변경 파일 전부 통과 | 변경 파일 전부 통과 |
| exact-inventory (`source-layout.md`/`source_layout_support.py`) | 통과 | 한 번 놓쳤다가(234→235) 즉시 발견해 고침 | 해당 없음 — 이번 phase는 새 `.py` 모듈을 추가하지 않아 동기화 불필요, 실제로 불필요함을 확인 |

`tools/analyze_openexr_one_day.py`의 `RUF100`은 세 phase 내내 이 작업과 무관한 기존 이슈로
반복 확인됐다(파일 미변경).

### Mock-run 재현 명령

GPU 없이도 Phase 3의 backtracking 배선이 파이프라인 상에서 실제로 다른 산출물을 내는지
확인할 수 있는 mock-run은 다음과 같다(`DeterministicSampler`/`SequenceProbe` fake로
`run_poc_revision`을 end-to-end 실행):

```bash
uv run pytest tests/test_poc_revision_pipeline.py -k frontier_width -v
```

실제 GPU pilot(이 세션에서는 실행하지 않았다 — RTX 3090과 Stable-DiffCoder 로드가 필요)은
다음 명령으로 착수한다:

```bash
dllm-agent poc-revision \
  --config configs/openexr-v3.4.14-search-tree-pilot.json \
  --source <verified-openexr-checkout> \
  --runtime-python <stable-diffcoder-venv-python> \
  --cache-dir <hf-cache-dir> \
  --artifact-dir <artifact-root>/search-tree-pilot \
  --output <artifact-root>/search-tree-pilot/report.json
```

warm ablation은 `--config configs/openexr-v3.4.14-search-tree-warm-pilot.json`으로 같은
명령을 반복하되, `failure_registry_path`(`artifacts/openexr-failure-registry.jsonl`)가 이전
cold run에서 이미 채워져 있어야 의미가 있다.

## 8. PR #1 리뷰 반영 — Failure Registry 동시성/추적성 수정 (2026-09-07)

레포 오너 승인 전 AI 코드 리뷰에서 지적된 3건을 검증 후 모두 수정했다. GPU 실행 전이므로
동작 자체(H1–H5 로직)는 바뀌지 않았고, Phase 2 registry의 내구성·추적성만 강화됐다.

| # | 지적 | 확인 결과 | 조치 |
|---|---|---|---|
| P1 | `append()`의 read-modify-write(`load` 후 `_write_all`)가 프로세스 간 동기화 없이 파일 전체를 교체해, 동시 저장 시 한쪽 기록이 유실됨 | 재현됨 — 20개 스레드가 같은 registry 파일에 동시에 `append`하면 잠금 없이는 레코드가 사라짐 | `failure_registry.py`에 배타적 파일 생성(`Path.open("x")`, POSIX `O_EXCL`/Windows `CREATE_NEW` 양쪽에서 원자적)을 이용한 크로스 프로세스 락(`_locked`)을 추가하고 `append()`의 read-modify-write 구간 전체를 그 안에서 실행하도록 감쌌다. 새 의존성 없음(stdlib만 사용). 락이 30초 이상 잡혀 있으면(예: 보유 프로세스 크래시) 무한 대기 대신 `FailureRegistryError`로 fail-closed한다. |
| P2 | warm 실행이 실제로 어떤 registry 내용을 seed로 소비했는지 결과에 남지 않아 재현성 검증 불가 | 확인됨 — `load_seed_records`가 읽은 레코드가 리포트 어디에도 해시로 남지 않았음 | `failure_registry.records_sha256()`을 추가하고, `PocRevisionReport`에 `failure_registry_seed_sha256: Sha256 \| None` 필드를 추가했다. warm 모드에서는 이번 실행이 실제로 읽은 레코드 집합의 해시가, cold 모드에서는 `None`이 채워진다 — registry 파일이 이후에 바뀌거나 rotate돼도 "이 리포트를 만든 입력이 정확히 무엇이었는지" 사후 검증 가능. |
| P2 | `run_id`로 `config_hash_sha256`을 그대로 재사용해, 같은 config를 공유하지만 시드(`particle_seed_base`/`proof_seed`/...)가 다른 별도 run request가 registry에서 서로 구별되지 않음 | 확인됨 — `config_hash_sha256`은 `RevisionConfig`만의 함수라 `RevisionRunRequest`가 다른 두 실행이 같은 id를 공유함 | `pipeline._registry_run_id(config, request)`를 신설해 `config`와 `run_request`를 함께 해시하도록 바꿨다. `config_hash_sha256` 필드 자체(기존 리포트 계약)는 그대로 두고 `persist_failure_records`에 넘기는 `run_id`만 교체했다. 완전히 동일한 (config, run_request) 재실행은 여전히 같은 id를 재사용하는데, 이는 저장소의 timestamp-free 결정론 원칙과 일치하는 의도된 동작이다(무작위 UUID/wall-clock을 넣지 않음) — 이 부분은 리뷰 제안을 그대로 받지 않고 근거를 남기고 좁혀서 반영했다. |

### 8.1 배치 쓰기(`append_many`) 추가

위 3건 수정과 별개로, 락을 추가하고 나니 `persist_failure_records`가 outcome × cause bucket마다
`append()`를 개별 호출해(런 하나당 최대 수십 번) 매번 락을 잡았다 풀고 파일 전체를 다시
읽는 구조라는 게 눈에 띄었다. `failure_registry.append_many()`를 신설해 한 런의 레코드를
모아 락 한 번으로 읽고-쓰기 하도록 바꿨다(`append()`는 이제 `append_many(path, (record,))`의
얇은 wrapper). 동일 키 레코드를 합쳐서 파일 자체를 압축하는 것(§6의 보류 항목)과는 다르다 —
이건 쓰기 방식만 배치로 바꾼 것으로, registry의 append-only 시맨틱은 그대로다.

검증: 새 테스트 9개(동시성 1, `records_sha256` 2, `append_many` 2, seed-hash 리포트 2, run_id
구분 1, 나머지는 기존 registry round-trip 커버) 전부 통과, WSL 전체 회귀
`234 failed, 651 passed, 6 errors`로 병합된 #1 이후와 byte-identical, `basedpyright`(전체)
`0 errors`, `ruff check`(변경 파일) 통과.
