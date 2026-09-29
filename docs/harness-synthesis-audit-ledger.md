# Harness-synthesis 감사 원장 설계 (Task 6)

`docs/superpowers/specs/2026-09-15-trajectory-guided-harness-synthesis.md` §10이 남긴
리스크에 대한 결정과 스키마. 구현: `src/dllm_agent/harness_synthesis/audit_ledger.py`.

## 문제

기존 `openexr_one_day` 파이프라인은 harness identity를 **이미지 digest 하나**로 고정한다
(`tools/openexr_runtime/Dockerfile`이 `poc_probe.cc`를 `COPY`로 굽고, `qualification.py`의
`finalize_config_payload_from_artifacts`가 그 이미지의 실측 바이트 해시를 config에 새긴다).
캠페인 전체에서 harness가 한 번도 바뀌지 않기 때문에 이 방식이 성립한다.

harness synthesis는 반대다 — **매 iteration마다 harness 소스가 바뀐다**(Task 5a의
compile-only 이미지가 이걸 가능하게 만든 것 자체가 원인). 이미지 digest 하나로는 iteration별
identity를 구분할 수 없고, `docs/paper-abstract-and-evaluation-claims.md`가 요구하는
"append-only JSONL audit ledger" 원칙과 맞지 않게 된다.

## 결정 (사용자 확인됨)

**반복마다 harness SHA-256을 append-only JSONL 원장에 기록한다.** 이미지 digest 한 번 대신,
매 iteration의 결과를 별도 레코드로 남긴다.

## 원장 구조 — `revision/failure_registry.py`를 그대로 따름

새 패턴을 만들지 않는다. 이 저장소에 이미 있는 append-only JSONL 원장
(`revision/failure_registry.py`)과 동일한 구조를 그대로 재사용한다:

- JSONL, 한 줄에 한 레코드(`FrozenModel.model_dump_json()`).
- 배타적 파일 생성(`"x"` 모드)으로 만드는 락 파일 — POSIX(`O_EXCL`)와 Windows(`CREATE_NEW`)
  양쪽에서 원자적이라 추가 의존성이 필요 없다. 락을 쥔 프로세스가 죽어도 타임아웃 후
  fail-closed(영원히 막히지 않음).
- `load`/`append`/`append_many`(배치 append 시 lock/read/write 왕복을 한 번만) API.
- 손상된 줄은 즉시 `AuditLedgerError`로 fail-closed(그 줄 번호를 명시).

기존 evidence-tree 패턴(`qualification.py`, `tools/evidence_tree_sha256.py`)과의 차이는
목적이다: evidence-tree는 **한 번 고정된** 산출물 트리 전체의 무결성을 증명하고, 이 원장은
**반복마다 달라지는** 값들의 append-only 이력을 남긴다. 서로 대체하지 않는다.

## 레코드 스키마 — `HarnessSynthesisLedgerRecord`

| 필드 | 타입 | 의미 |
|---|---|---|
| `schema_version` | `Literal["dllm-harness-synthesis-ledger/v1"]` | 스키마 진화 대비 |
| `run_id` | `str` | 이 iteration이 속한 실행의 식별자 |
| `iteration` | `int` | 0-based 반복 번호(하나의 run 안에서) |
| `arm` | `str` | 비교 arm 이름(예: `feedback`, `random`, `kb_usage`) |
| `seed` | `int` | 이 iteration의 생성 seed |
| `harness_source_sha256` | `Sha256` | 이 iteration이 생성한 harness 소스의 정확한 바이트 해시 |
| `knowledge_base_sha256` | `Sha256` | 이 iteration에 실제로 쓰인 KB 프롬프트 텍스트의 해시(`api_knowledge.render_prompt_text`의 출력) — KB가 shared/dynamic(§ "다음 단계" 참고)이라 iteration마다 내용이 바뀔 수 있어, KB 자체도 버전 고정이 필요 |
| `usage_example_names` | `tuple[str, ...]` | 이 iteration의 프롬프트에 실제로 포함된 검증된 예시 이름들 |
| `compiled` | `bool` | compile-only 이미지(Task 5a)에서의 실제 컴파일 결과 |
| `compile_error` | `str \| None` | 컴파일 실패 시 오류 텍스트(`api_knowledge.hint_for_missing_headers`가 파싱하는 것과 같은 형식) |
| `coverage_edges` | `int \| None` | 컴파일된 경우의 커버리지 측정값(없으면 `None`, 음수 불가) |
| `recorded_at_unix_seconds` | `float` | 기록 시각(감사 추적용, 실행 순서 재구성) |

`harness_source_sha256`은 harness 소스 바이트 자체의 해시이고, 원장은 소스 텍스트 자체를
담지 않는다(evidence-tree 쪽처럼 별도 evidence 디렉터리에 실제 바이트를 저장하고 이 원장은
그 해시만 참조하는 방식을 Task 8+ 구현에서 따른다 — evidence 경로 자체의 설계는 이 문서의
범위 밖이며, Task 8 구현 시 `qualification.py`의 `persist_deep_parse_evidence` 같은
sibling-directory 패턴을 재사용할 것을 권장한다).

## 남은 설계 결정 (Task 8+에서 확정)

- **KB 버전 고정의 정확한 대상**: `api_knowledge.render_prompt_text`의 출력 문자열 해시로
  충분한가, 아니면 KB를 구성하는 각 구성요소(header_index, declarations, usage examples)를
  별도로 해시해 더 세밀하게 추적해야 하는가 — 공유 KB가 실행 중 갱신되는 설계(spec §6의
  `kb_shared` arm)가 실제로 쓰이게 되면 재검토 필요.
- **evidence 바이트 저장 경로**: 이 문서는 해시만 다룬다. 실제 harness 소스 바이트를
  어디에 sibling-directory로 영구 보존할지는 Task 8 구현에서 결정한다.
