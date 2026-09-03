# dLLM Agent

Diffusion language model(dLLM)의 denoising 궤적이 제한된 동적 검증 예산을 더 잘
배분할 수 있는지 연구하기 위한 취약점 탐지 에이전트입니다. 현재 저장소는 논문
설계와 함께 결정론적 replay MVP와 실제 Stable-DiffCoder 기반 OSS fuzzing 경로를
제공합니다.

**문서 검토 기준:** 2026-08-31 · **연구 상태:** 역사적 OpenEXR pilot은 NO-GO,
online portfolio prototype은 구현·검증 완료/실제 효능 미측정

## 현재 결론

- 주 모델은 pin된 `Stable-DiffCoder-8B-Instruct`입니다. 네 weight shard와 custom
  model source의 SHA-256을 검증하고 Python 3.12.13, Torch 2.5.1+cu121,
  Transformers 5.3.0에서 실제 checkpoint를 RTX 3090 한 장에 로드했습니다. Hub의
  MIT metadata와 별개로 exact-weight LICENSE artifact가 없는 법적 경계는 유지합니다.
- 논문의 방어 가능한 질문은 “dLLM 에이전트가 더 우월한가”가 아니라 “후보별
  commitment/denoising 궤적이 동일 후보·동일 검증 예산에서 검증 가능한 취약점
  가설의 우선순위화를 개선하는가”입니다.
- `ReplaySampler → trajectory score → candidate selection → ASan proof → hash-carrying
  JSON`은 모델 공통의 재현 가능한 최소 spine입니다. dLLM 고유 계층은 실제
  `DiffusionSampler`, trajectory-first capture, 후보별 ranking, realised-NFE 기반 budget
  routing입니다.
- Stock Stable-DiffCoder에서는 이미 확정된 토큰이 다시 열리지 않습니다. 따라서 이
  프로젝트는 해당 sampler를 “자기교정” 또는 “전역 revision”으로 설명하지 않습니다.
- 실제 경로는 `jsoncons@af497a1…` harness hunt와 OpenEXR `v3.4.14` native tournament를
  포함합니다. 모델의 token-transfer confidence와 commitment hash, 실제 fuzz·coverage
  결과를 분리해 기록하고 선택된 source만 독립 proof로 승격합니다.
- 2026-08-19 live run은 두 위치의 재현 가능한 UBSan-only zero-length copy 위반을
  관찰했고, 별도 ASan-only 121초 run은 2,011,697 units에서 ASan crash를 관찰하지
  않았습니다. 이는 zero-day 또는 dLLM 우월성 주장이 아닙니다.
- OpenEXR 단일 bring-up 비교에서 native controller는 86.20초, static baseline은 50.29초로
  더 느렸습니다. target coverage는 소폭 높았지만 후보 source·trajectory가 같고 반복이 없어
  인과 효과로 해석하지 않으며, 양쪽 모두 sanitizer finding이 없었습니다.
- feedback-conditioned OpenEXR PoC revision은 canonical 5-arm×5-rep pilot을 완료했습니다.
  artifact root는
  `/home/seory0/.cache/dllm-agent/openexr-poc-5arm-pilot-canonical-20260820-183739`입니다.
  primary `validated_success`는 모든 arm에서 false였고 paired summary는 `NO-GO`입니다.
  sanitizer signal이 없어 proof는 skipped였으며 취약점이나 zero-day는 발견하지 않았습니다.
  exploratory `new_edges`는 `selective-revision - pseudo-feedback` 평균 +11.6,
  CI `[-1.6, 25.2]`로 inconclusive입니다.

## 빠른 실행

Python 3.12와 C compiler의 AddressSanitizer 지원이 필요합니다.

```bash
uv sync --all-groups

uv run dllm-agent scan fixtures/cwe121/vuln.c \
  --fixture fixtures/replay/stable-diffcoder-cwe121.json \
  --output /tmp/dllm-scan.json

uv run dllm-agent plan configs/owned-fixture-mvp.json \
  --output /tmp/dllm-plan.json

uv run dllm-agent sample configs/jsoncons-ubjson-hunt.json \
  --runtime-python /path/to/model-venv/bin/python \
  --cache-dir /path/to/huggingface-cache \
  --output /tmp/jsoncons-candidates.json

uv run dllm-agent hunt configs/jsoncons-ubjson-hunt.json \
  --candidate-bank /tmp/jsoncons-candidates.json \
  --source /path/to/jsoncons \
  --artifact-dir /tmp/jsoncons-hunt-artifacts \
  --output /tmp/jsoncons-hunt.json

uv run dllm-agent native-hunt configs/openexr-v3.4.14-hunt.json \
  --candidate-bank /path/to/openexr-candidates.json \
  --controller configs/openexr-v3.4.14-native-controller.json \
  --source /path/to/openexr \
  --artifact-dir /tmp/openexr-native-artifacts \
  --output /tmp/openexr-native.json

uv run dllm-agent poc-revision \
  --config configs/openexr-v3.4.14-online-portfolio.json \
  --source /path/to/openexr-v3.4.14 \
  --runtime-python /path/to/model-venv/bin/python \
  --cache-dir /path/to/huggingface-cache \
  --artifact-dir /tmp/openexr-poc-revision-artifacts \
  --output /tmp/openexr-poc-revision.json

uv run dllm-agent web \
  --runtime-python /path/to/model-venv/bin/python \
  --cache-dir /path/to/huggingface-cache \
  --source-root /path/to/jsoncons
```

`openexr-v3.4.14-poc-revision.json`은 2026-08-20 5-arm 결과를 재현하기 위한 legacy
config이고, 현재 online portfolio runtime은 위 전용 config를 사용합니다.

웹 콘솔은 기본적으로 `http://127.0.0.1:8080`에만 바인딩됩니다. `/`는 저장된 run을
비교하는 최소 Runs table이고 `/run/{resource_id}`는 dLLM candidate instance,
Lines/Functions/Branches/Regions coverage, fuzz artifact를 보여주는 detail입니다.
trajectory, source, hash는 기본적으로 닫힌 evidence disclosure에 남습니다. report의
`candidate_bank_sha256`과 catalog bank SHA-256이 일치할 때만 둘을 결합하며, 브라우저는
임의 경로나 셸 명령을 전달할 수 없습니다. 화면은 생성 완료 후 저장된 증거만 표시하고
모델의 비공개 사고 과정이나 실시간 tool execution으로 표현하지 않습니다.

`scan`은 네 개의 고정 trajectory를 점수화한 뒤 가장 높은 하나만 검증합니다. CLI는
repository-owned CWE-121 target/fixture의 resolved path와 고정 SHA-256 allowlist를 먼저
검사하고, 검증된 target bytes만 임시 디렉터리에 staging해 컴파일합니다. ASan의
`stack-buffer-overflow`와 정확한 source line이 함께 관찰될 때만 `proved`가 됩니다. 결과
JSON에는 CWE, span, trajectory 순위, validator verdict, 계산량과 입력 artifact SHA-256이
포함됩니다. `synthetic_trace_rotation`은 이 fixture의 한 trace 안에서 문자열을 회전한 QA
field이며, 논문에서 제안한 후보 간 score derangement가 아닙니다.

기존 `scan` validator는 계속 allowlist를 통과한 trusted repository-owned bytes만 host에서
실행하고 외부 pair를 거부합니다. Third-party source는 별도 `hunt` 명령에서만 허용되며,
정확한 Git commit·clean worktree·origin·`SECURITY.md` hash를 검증한 뒤 network-off,
read-only, capability-dropped, non-root Docker 경계에서 실행합니다. 격리를 구성하지
못하면 host 실행으로 fallback하지 않습니다.

## 검증

```bash
uv run ruff format --check .
uv run ruff check .
uv run basedpyright
uv run pytest
```

test suite는 성공 경로, 결정론적 ranking, target/fixture allowlist·고정 hash 경계,
외부 matching pair의 무실행 거부, ASan proof, timeout·compile failure, malformed config와
missing target의 exit code 2를 포함합니다. 검증 결과는 저장된 과거 QA가 아니라 현재
작업 트리에서 위 명령을 새로 실행해 판단합니다. Owned fixture는 runner QA 전용이며 실제
취약점 발견 성능의 과학적 증거로 사용하지 않습니다.

## 문서 허브

- [문서 색인과 프로젝트 현황](docs/README.md): 구현·측정·구현 중·설계를 분리한 상태표,
  권장 읽기 순서, source of truth
- [연구 보고서](docs/research-report.md): 모델 선정, diffusion 기제, harness 결정,
  최신 관련 연구, target/benchmark, 반증 조건과 단계별 roadmap
- [사전등록 실험 프로토콜](docs/experiment-protocol.md): manifest/event schema,
  model×scaffold×budget 통제, trace permutation, leakage/dedup, 통계·resource gate
- [feedback-conditioned PoC revision 설계](docs/superpowers/specs/2026-08-20-feedback-conditioned-poc-revision-design.md):
  상태·이벤트 스키마와 검증 계약
- [feedback-conditioned OpenEXR PoC 결과](docs/poc-revision-openexr-results-20260820.md):
  canonical 5-arm pilot 결과, hash artifact, no-go 판정과 과학적 한계
- [dLLM-native online portfolio harness](docs/dllm-native-online-portfolio-harness.md):
  현재 구현 흐름, exact crash-artifact proof, dLLM-native 기제, 한계와 필수 ablation

실행 결과, 운영 문서, 역사적 설계 메모와 implementation plan은 문서 색인에서 상태별로
분류합니다. 날짜가 붙은 결과 문서는 당시 artifact의 snapshot이며 현재 기능 설명으로
덮어쓰지 않습니다.

## 현재 상태

- `Stable-DiffCoder-8B-Instruct` 기반 replay/hunt/native/web 경로는 구현과 검증이
  분리된 상태로 유지된다.
- `jsoncons` UBJSON live hunt, OpenEXR 공식 harness baseline, dLLM native tournament는
  각각 별도 결과 문서와 hash artifact로 분리되어 있다.
- feedback-conditioned OpenEXR PoC revision은 공개 CLI/config와 canonical 5-arm×5-rep
  pilot evidence가 있다. primary 결과는 no-go이고, exploratory coverage 차이는
  inconclusive이며, 취약점 발견 또는 dLLM 우월성으로 주장하지 않는다.
- 새 online portfolio runtime은 별도 config로 통합·자동 검증됐지만 post-fix 실제
  Stable-DiffCoder×OpenEXR 효능 실험은 아직 없으며, 과거 NO-GO는 새 runtime의 성능
  증거로 재사용하지 않는다.
- 모든 메인 문서는 Markdown/Mermaid로 유지하고, PDF/DOCX 생성물은 관리하지 않는다.

## 구현 경계

중간 denoising state는 분석 데이터일 뿐 도구를 호출할 수 없습니다. typed action은
finalized candidate에서 schema validation을 통과한 후에만 compiler·sanitizer 같은
validator로 전달됩니다. DeepSeek Harness에서는 durable event와 sandbox 원칙만
차용하며, AR `AsyncIterable<StreamChunk>` loop 안에 diffusion state를 가장하지
않습니다. 전체 실험은 MDASH의 prepare–scan–validate–dedupe–prove 형태를 따르되
독립 runner로 유지합니다.
