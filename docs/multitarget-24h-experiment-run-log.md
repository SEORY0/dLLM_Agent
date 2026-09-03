# Multi-target 24-hour experiment run log

이 문서는 `libpng`/`libTIFF`/`libxml2` 비교 실험의 사람이 읽는 append-only 실행 원장이다.
기계 판독 원장은 실험 디렉터리의 `events.jsonl`이며, 각 이벤트와 산출물은 SHA-256으로
연결한다. 시간은 별도 표기가 없으면 Asia/Seoul ISO 8601이다.

## 2026-08-21T11:16:55+09:00 - 실험 동결

- 프로토콜: `docs/multitarget-24h-experiment-protocol.md`
- 그리드: 3 targets × 5 arms × 1 repetition = 15 cells
- target별 dLLM 원시 trajectory: 정확히 32개
- unique gate: materialized SHA-256 기준 최소 16개, 미달 시 해당 target fail-closed
- 셀별 search worker budget: 정확히 86,400초
- feedback 셀: initial micro-probe 480초 + revision micro-probe 480초 + main 85,440초
- non-feedback 셀: main 86,400초
- proof/minimization 비용: search budget 밖에서 별도 기록
- 통계 해석: 1회 반복이므로 기술통계와 효과 방향만 보고하며 유의성은 주장하지 않음

### 고정 핀

| 구성요소 | 핀 |
| --- | --- |
| dLLM Agent 시작 Git HEAD | `8a245fec4da5d529bf9830ec15b98d5c5215a1b8` |
| Magma | `v1.2.1`, `1e4d8c248debacd7a5c4b157a1059d5290bd3a22` |
| libpng upstream | `a37d4836519517bdce6cb9d956092321eca3e73b` |
| libTIFF upstream | `c145a6c14978f73bb484c955eb9f84203efcb12e` |
| libxml2 upstream | `ec6e3efb06d7b15cf5a2328fabd3845acea4c815` |
| Stable-DiffCoder | `ByteDance-Seed/Stable-DiffCoder-8B-Instruct@10cdaf9b486f1cf0273ad968459dbe2d21a1482a` |
| remote model code | `298b19a464f32c00d38e43603516174a73fb4c531193369ff1aabbef32593104` |
| model venv Python | `/home/seory0/.cache/dllm-agent/model-venv/bin/python` |
| model venv pydantic | `2.13.4` |

Magma가 가져오는 upstream commit은 논문 표에 흔히 적히는 literal release tag commit과
같지 않다. 이 실험은 재현 가능한 Magma v1.2.1 fetch pin을 기준으로 한다.

Stable-DiffCoder는 stock checkpoint와 pin된 remote model source hash를 그대로 사용한다.
`pydantic`은 JSONL worker protocol과 runtime schema parsing을 위한 model venv 실행
의존성이다.

### 시작 자원 스냅샷

- CPU logical cores: 40
- RAM: 134,797,250,560 bytes
- 시작 available RAM: 88,781,365,248 bytes
- 디스크 여유: 약 83 GiB
- GPU: NVIDIA GeForce RTX 3090 24,576 MiB × 8
- NVIDIA driver: 535.230.02

### 모델 실행 환경 확인

검증 명령:

```bash
/home/seory0/.cache/dllm-agent/model-venv/bin/python -c "import pydantic, sys; print(sys.executable); print(pydantic.__version__)"
```

검증 결과:

```text
/home/seory0/.cache/dllm-agent/model-venv/bin/python
2.13.4
```

현재 완료된 것은 protocol freeze, 이미지 빌드/smoke, proof 경계 smoke, model runtime
dependency 확인이다. 15개 24-hour search cell 완료 결과는 아직 이 로그에 기록하지 않는다.

## 2026-08-21 - 빌드 시도 1 (실패, 보존)

- 대상: `libpng_read_fuzzer`
- image tag: `dllm-magma/libfuzzer_builtin/libpng:v1.2.1-asan-isancanary`
- 구성: ASan + `MAGMA_ENABLE_CANARIES` + `MAGMA_FATAL_CANARIES`
- 결과: autotools `checking whether the C compiler works... no`
- 원인: `-fsanitize=fuzzer`를 전역 `LIBS`에 넣어 configure link probe에도 libFuzzer
  `main`이 주입된 링크 경계 오류
- 원본 로그: `/home/seory0/.cache/dllm-agent/magma-build-logs-v1.2.1/libpng-builtin.log`
- 로그 SHA-256: `ecc78440df529b3ab682a580eb8cc7a111c2ea6c5909e2ac0155b4a7a5e851f9`

### 수정 시도 2

라이브러리 계측에는 `-fsanitize=fuzzer-no-link`만 적용하고, libFuzzer runtime은 최종
harness 링크 명령에만 `FUZZER_LIBS=-fsanitize=fuzzer`로 주입하도록 격리했다.

| 파일 | 수정 후 SHA-256 |
| --- | --- |
| `fuzzers/libfuzzer_builtin/instrument.sh` | `2561a2928139a01a33bb32029e8fb315d177fd2e7f8452aef5e96b3de5a3009f` |
| `targets/libpng/build.sh` | `7deed24abf8b37545b03610967fddbfb90e4c787443a0986ca956552e9b60679` |
| `targets/libtiff/build.sh` | `c0c203f4fe76c5f1cfbb11205a610ba8b017df716d9dca68adc481f284f2e5d3` |
| `targets/libxml2/build.sh` | `822d69c9b6b7021d32aead5276ea0342ac539c451007131cec95a248ec1a73de` |

네 shell script 모두 `bash -n`을 통과했다. 다음 단계는 libpng 재빌드로 링크 경계를
실제 Docker surface에서 확인하는 것이다.

## 2026-08-21T02:23:00+00:00 - libpng 빌드 및 실제 smoke 성공

- build attempt: 2
- image ID: `sha256:9e608294deff1208bc05d0b761da36e70bfe07e71f74bd164abd4fa094109c04`
- image size: 1,083,397,143 bytes
- Docker: 26.1.3
- build log SHA-256: `56bc0cebd886b21b8a9e50ef7e760b8a0b78ab3d4c34317276e8a190802ac874`
- 핵심 확인: autotools C compiler link probe 성공, 최종 libFuzzer harness 링크 성공

실제 surface 검증은 CPU 0, fuzzer seed `202608210`, 5초, 기본 corpus 4개로 실행했다.
컨테이너는 exit 0, OOM false였고 약 70,907개까지 진행된 상태 행, 최종 `cov=754`,
`ft=1547`, corpus 107을 관측했다. `PNG003_T=10`이었고 finding 16개가 생성됐다.
`canaries.raw` SHA-256은
`87e42452024f6990ead69615d0b90d03993c4d440451a5b34bf1f45574af2da3`이다.

이 smoke의 `UndefinedBehaviorSanitizer: SEGV`는 `MAGMA_FATAL_CANARIES`가 의도적으로
발생시킨 benchmark oracle crash다. 독립 non-fatal-canary ASan replay 전에는 실제
메모리 취약점 또는 제로데이로 분류하지 않는다.

## 2026-08-21T02:23:00+00:00 - 나머지 target 빌드 시작

- libTIFF image tag: `dllm-magma/libfuzzer_builtin/libtiff:v1.2.1-asan-isancanary`
- libxml2 image tag: `dllm-magma/libfuzzer_builtin/libxml2:v1.2.1-asan-isancanary`
- 양쪽 모두 동일한 Magma/fuzzer adapter/canary/ASan 빌드 계약을 사용한다.
- 독립 로그는 각각 `libtiff-builtin.log`, `libxml2-builtin.log`에 저장한다.

## 2026-08-21T02:27:08+00:00 - sanitizer 설정 감사 및 이미지 폐기

초기 v2 빌드 명령을 compile line과 대조하자 태그에는 `asan`이 있었지만 실제
`BUILD_FLAGS`에는 `-fsanitize=address`가 없었다. 다음 세 이미지는 본 실험에서
사용하지 않는 rejected build로 분류했다.

- `sha256:9e608294deff1208bc05d0b761da36e70bfe07e71f74bd164abd4fa094109c04`
- `sha256:4c88332b5b1dfd20b60700c90b63f558b75ad5ab11c68d80d98f2ab2622da423`
- `sha256:3e3a33b4d76f2aa51c6041f19622e9409ec6b75f90127e176a8ba64a1e5431d0`

또한 libTIFF attempt 1은 upstream `autogen.sh`가 Savannah의 `config.guess`를
빌드 중 다시 받으려다 실패했다. 네트워크 변동을 제거하기 위해 로컬
`autoreconf -fiv`로 교체했다. 수정 build script SHA-256은
`c0c203f4fe76c5f1cfbb11205a610ba8b017df716d9dca68adc481f284f2e5d3`이다.

## 2026-08-21T02:28:27+00:00 - 실제 ASan search 이미지 고정

공통 compile flags:

```text
-include /magma/magma/src/canary.h -DMAGMA_ENABLE_CANARIES
-DMAGMA_FATAL_CANARIES -g -O1 -fno-omit-frame-pointer -fsanitize=address
```

| Target | Search image ID | Build log SHA-256 |
| --- | --- | --- |
| libpng | `sha256:e304abee1c0d1a8366e10110b6267701eff4b761cc023ee882ebc7d46f74cf41` | `ed1b98292476ed5c250378cd3fcfdb3d43851843119f56606eeaaa2a4fbe39e5` |
| libTIFF | `sha256:6c98e35375e446c9ea08e2936f1470c1e3abf8dfcc2431d4a9c9fe47257e4894` | `7e9189e17b3cac367321867cb586bbf78bc3f9aa7f853e5da4340e46ba7b3b1e` |
| libxml2 | `sha256:76cd8aab328b4cd7cd4bf143448b6c1719ba821bd94a46567dedad8db7eb1046` | `9ccb9010da1017a426bffe3a0ce5ad7289bcaf1dd77aef4bf6075e2dbac4b9ae` |

5초 실제 smoke에서 libpng/libTIFF는 exit 0과 ASan runtime을 확인했다. libxml2의
전체 기본 corpus는 캠페인 시작 전 canary-trigger seed 정리에 오래 걸렸으므로 smoke를
수동 중단해 보존했고, 구조화 입력 1개로 다시 실행해 exit 0과 `cov=1135`를 확인했다.
본 실험은 모든 arm에 SHA-256 순서로 고른 최대 16개 compact base corpus를 동일하게
사용한다.

## 2026-08-21T02:42:30+00:00 - 독립 proof 경계 고정

Proof 이미지는 위 flags에서 `MAGMA_FATAL_CANARIES`만 제거했다.

| Target | Non-fatal proof image ID | Build log SHA-256 |
| --- | --- | --- |
| libpng | `sha256:32dc1104f9397670c9bc87ebfc72a75a2234ced77eea75435352f6d7c087c965` | `95aac151c23f808d1bc2370dc9313a878d1f4206ead1ada0860f0152cc57c5fb` |
| libTIFF | `sha256:79c3496b95953fcb6004d6e2bbcdd1d9c02a32772cbd544a803c3a44f4fc9ee4` | `fde4350dfb714f9cbce1d84de31787ac154ae9d16748502328b7bef69178f79f` |
| libxml2 | `sha256:edaba3bfc994869e79a5cac5bfca4d766fdd0a80797a08ce3c6d4993b58ae720` | `3d8fbf9e6cffabf5b4c7c22137f6435e599ccd30aabcd9a8300a4c1ce6636054` |

검색 이미지에서 생성된 `PNG003` finding
`64141496b5a66038972020cc10a4d811d359cbd35f85c876db7aae8859d1d13f`를
서로 다른 proof seed와 CPU의 새 컨테이너 두 개에서 재생했다. 둘 다 exit 0이고 ASan
signature가 없어 `not_reproduced`였다. 즉 fatal canary finding을 취약점으로 오인하지
않는 분리 경계가 실제 Docker surface에서 확인됐다.

## 2026-08-21T11:30:50+09:00 - 세 target v3 ASan/Magma 이미지 확인

최종 24h 비교 실험에는 아래 로컬 Docker 이미지를 사용한다.

| target | image tag | image ID | build log SHA-256 |
| --- | --- | --- | --- |
| libpng | `dllm-magma/libfuzzer_builtin/libpng:v1.2.1-asan-isancanary-v3` | `sha256:e304abee1c0d1a8366e10110b6267701eff4b761cc023ee882ebc7d46f74cf41` | `ed1b98292476ed5c250378cd3fcfdb3d43851843119f56606eeaaa2a4fbe39e5` |
| libTIFF | `dllm-magma/libfuzzer_builtin/libtiff:v1.2.1-asan-isancanary-v3` | `sha256:6c98e35375e446c9ea08e2936f1470c1e3abf8dfcc2431d4a9c9fe47257e4894` | `7e9189e17b3cac367321867cb586bbf78bc3f9aa7f853e5da4340e46ba7b3b1e` |
| libxml2 | `dllm-magma/libfuzzer_builtin/libxml2:v1.2.1-asan-isancanary-v3` | `sha256:76cd8aab328b4cd7cd4bf143448b6c1719ba821bd94a46567dedad8db7eb1046` | `9ccb9010da1017a426bffe3a0ce5ad7289bcaf1dd77aef4bf6075e2dbac4b9ae` |

빌드 로그 파일:

- `/home/seory0/.cache/dllm-agent/magma-build-logs-v1.2.1/libpng-builtin-v3-asan.log`
- `/home/seory0/.cache/dllm-agent/magma-build-logs-v1.2.1/libtiff-builtin-v3-asan.log`
- `/home/seory0/.cache/dllm-agent/magma-build-logs-v1.2.1/libxml2-builtin-v3-asan.log`

실제 surface smoke는 같은 Docker 런타임 경로로 target별 5초씩 실행했다.

| target | container prefix | exit | OOM | timeout | parsed coverage | canary series | findings | log SHA-256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| libpng | `2c83bf34598e` | 0 | false | false | 0 | 5 | 6 | `54c2398644f3f6b5ef27f6fe509d5c7e0ca001260638355c0787e14a5a500ba3` |
| libTIFF | `dd241ff041ad` | 0 | false | false | 0 | 2 | 0 | `e316f511922acdff12d25f28f1807e2f2043dc566ba1b808ec682b8d60061196` |
| libxml2 | `70ca4bdb93cd` | 0 | false | false | 0 | 8 | 0 | `9e58feb0071545c5330f040dd10ebb83ac990797bf76d86856b1e305bb57e011` |

5초 smoke에서 monitor/finding surface는 생성됐지만, 현재 로그 parser 기준 coverage는 0이다.
본 실험 전 libFuzzer 로그 보존/coverage parser를 추가로 확인해야 한다.

## 2026-08-21T12:24:17+09:00 - 실행 러너 고정 및 24시간 실행 직전 게이트

- 실제 full 실행 루트: `/home/seory0/dLLM_Agent/artifacts/experiments/multitarget-full-20260821T032417Z`
- qualification smoke 루트: `/home/seory0/dLLM_Agent/artifacts/experiments/multitarget-smoke-20260821T032417Z`
- 이전 빌드·proof 경계 원장: `/home/seory0/dLLM_Agent/artifacts/experiments/multitarget-24h-20260821T021655Z/events.jsonl`
- runner SHA-256: `e8be4ecde0cd1b9d4eec02925500ef534dae3e759c2cdf3c436417429d98af36`
- coordinator SHA-256: `e80c8dfc3784763d9fa4ed7dcffa93062cc26e8210858d8abdf1b1df3d8b5d05`
- schedule SHA-256: `147748c2971399277ec9a9db049f2979bee1a03cd56f87c47d63012f913f0d04`

모델 초기 particle, 모델 revision, initial probe, revision probe, main fuzzer, proof replay는
서로 다른 seed namespace를 사용한다. target 단위 물리 공유가 의도된 것은 initial particle과
initial probe뿐이다.

실행 전 검증:

```text
pytest: 84 passed
Ruff: All checks passed
BasedPyright: 0 errors, 0 warnings
uv build: sdist/wheel success
```

코드 리뷰에서 발견된 실행 전 위험 세 가지를 수정했다.

- full/smoke 실행 루트는 모두 빈 디렉터리만 허용한다.
- evidence tree SHA-256은 self-reference가 생기지 않도록 hashed root 바깥 sidecar에 쓴다.
- repository Markdown 결과 복사 전 `docs/` 부모 디렉터리를 생성한다.

실제 실행 명령은 다음과 같다. runner는 1초 probe·5초 main의 15-cell qualification smoke가
모두 정상 종료한 경우에만 정확한 full budget으로 진행한다.

```bash
PYTHONPATH=/home/seory0/dLLM_Agent/src:/home/seory0/dLLM_Agent \
  /home/seory0/.cache/dllm-agent/model-venv/bin/python \
  tools/run_multitarget_24h.py \
  --experiment-root /home/seory0/dLLM_Agent/artifacts/experiments/multitarget-full-20260821T032417Z \
  --smoke-root /home/seory0/dLLM_Agent/artifacts/experiments/multitarget-smoke-20260821T032417Z \
  --magma-root /home/seory0/.cache/dllm-agent/magma-v1.2.1 \
  --model-cache /home/seory0/.cache/dllm-agent/huggingface \
  --device cuda:0 \
  --smoke-probe-seconds 1 \
  --smoke-main-seconds 5
```

## 2026-08-21T12:33:18+09:00 - 실제 모델 unique gate 실패와 sampler 수정

첫 실제 실행은 모델과 remote code hash를 정상 검증한 뒤 `libpng`의 32개 trajectory를
생성했지만, materialized snapshot이 16개 미만으로 수렴해 unique gate에서 중단됐다.
qualification smoke와 full main container는 하나도 시작하지 않았고 search worker-second
소비는 0이다.

- 실패 실행 루트: `/home/seory0/dLLM_Agent/artifacts/experiments/multitarget-full-20260821T032417Z`
- 실패 원장 최종 stream SHA-256: `0ee0d63add362a17bfa0f45840a55ce0b15f234e323381f8b6e4e3629c0af06d`
- 실패 분류: expected fail-closed diversity gate

원인은 allowed numeric selector logits가 과도하게 뾰족해 seeded categorical sampling이
소수 조합으로 수렴한 것이다. 32 raw 계약을 늘리거나 실패를 숨기지 않고, 모델 순위를
유지한 채 selector-class logits에 고정 temperature `8.0`을 적용했다.

```text
proposal = softmax(allowed_selector_logits / 8.0)
```

이 정책은 seed 재현성, allowed-token 제약, parent selector 제외, 포맷 anchor를 모두
보존하면서 particle exploration만 넓힌다. unique gate 오류에는 실제 unique count도
포함하도록 보강했다.

- 수정 worker SHA-256: `98d41fb476f0333123c260c155bd2b75556302bb0f92cd2390b2a08f536b3e5c`
- 수정 candidate bank SHA-256: `07276d5b9a87d84204b466d2901f06245d520a3437711d4292e61d689e79c4ff`
- 관련 테스트: 9 passed, Ruff clean, BasedPyright 0 errors/0 warnings

## 2026-08-21T12:43:07+09:00 - diversity gate 통과, smoke 통과, run-scope 충돌 격리

temperature 수정 후 세 target의 실제 initial bank가 모두 gate를 통과했다.

| target | raw | unique | selected | NFE | GPU ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| libpng | 32 | 26 | 16 | 590 | 70,949 |
| libTIFF | 32 | 25 | 16 | 849 | 105,957 |
| libxml2 | 32 | 26 | 16 | 455 | 65,018 |

15-cell qualification smoke도 모두 exit 0으로 완료했다. smoke 원장 최종 stream SHA-256은
`461d0d61828fcfe90ced5705ae7819822c624dd7d7b18ef3fca65613dc406584`다.

그 다음 full initial probe의 첫 컨테이너가 종료된 smoke probe와 동일한 이름을 사용해
Docker conflict로 중단됐다. full main container는 시작되지 않았고 full search
worker-second 소비는 0이다.

- 실패 full 루트: `/home/seory0/dLLM_Agent/artifacts/experiments/multitarget-full-20260821T033400Z`
- 실패 full 원장 stream SHA-256: `731d70e5034e8298e83a7e9acb1d89334fd99e090d8f8be1b6813588eb5daf89`
- 성공 smoke 루트: `/home/seory0/dLLM_Agent/artifacts/experiments/multitarget-smoke-20260821T033400Z`

모든 initial/revision probe 이름에 experiment-root 기반 `run_label`을 포함하도록
coordinator를 수정했다. 수정 runner SHA-256은
`c2233954eadbcf43a3b107d39cf8135a2da64172f79ef62abb6b8a34ec74c4d3`, coordinator
SHA-256은 `a3f5e9e11c103759f2fcb62d96f13fd9e50d8b853e1ab84fec62c3c67d909479`다.

## 2026-08-21T13:01:04+09:00 - descendant OOM 판정 경계 수정

run-scoped 이름을 적용한 세 번째 qualification smoke는 다시 15/15 exit 0으로 통과했고,
full initial probe도 충돌 없이 시작됐다. libTIFF p02에서 Docker `OOMKilled=true`가
기록됐으나 컨테이너 supervisor는 exit 0으로 정확한 30초 예산을 완료했다. 로그는
libFuzzer fork child의 OOM/ASan 탐색을 `-ignore_ooms=1`로 계속 처리한 것을 보여준다.

기존 검증기는 `OOMKilled=true` 자체를 인프라 실패로 오판했다. 잘못된 실행을 계속하지
않기 위해 해당 run을 수동 중단했고, main container는 시작하지 않았다.

- 중단 full 루트: `/home/seory0/dLLM_Agent/artifacts/experiments/multitarget-full-20260821T034400Z`
- 중단 full 원장 stream SHA-256: `516b198336c14ceff27dfb376d034f35a788ddc0ddf849c192f4162d07fd4ed5`
- full initial probe launched: 35개, requested physical probe worker-seconds 1,050
- 최종 비교 포함 여부: false

검증 의미를 다음처럼 분리했다.

```text
supervisor clean = exit 0 AND docker status exited AND no timeout/process error
descendant OOM  = search finding/evidence로 기록, clean supervisor를 실패로 만들지 않음
nonzero exit    = fail closed
```

- 수정 runner SHA-256: `086918e48ea82363e8e70c1609f0e3717ce8bb15fb1062bc0e1a86d8c3538876`
- 수정 monitor SHA-256: `023723f780aa8e69dcd5cca9ff2eacd980116115af1d8f1b317e666527a22779`
- 수정 probe runner SHA-256: `e21fdb73346fe4996a46183321089d2455325ffb7e5db4e6f9a0ce34a77aa964`
- 관련 테스트: 17 passed, Ruff clean, BasedPyright 0 errors/0 warnings

## 2026-08-21T13:28:16+09:00 - 정식 15-cell 캠페인 시작

네 번째 실행은 세 target 모두 실제 resident Stable-DiffCoder generation과 unique gate를
통과했고, run-scoped qualification smoke 15/15, full initial probe 48/48, full revision
probe 144/144가 supervisor exit 0으로 완료된 뒤 정식 캠페인을 시작했다.

- 정식 실행 루트: `/home/seory0/dLLM_Agent/artifacts/experiments/multitarget-full-20260821T040200Z`
- qualification smoke 루트: `/home/seory0/dLLM_Agent/artifacts/experiments/multitarget-smoke-20260821T040200Z`
- run label: `full-d05566b1`
- 본 실행 반복 수: target×arm당 1회
- 비교 셀: 3 targets × 5 arms = 15
- full main 시작 범위: `2026-08-21T04:26:51.933953706Z`부터
  `2026-08-21T04:26:56.793667960Z`까지
- `main-campaigns-launched` 원장 시각: `2026-08-21T04:26:56.827205Z`
- 시작 시점 원장 최종 stream SHA-256:
  `0569b43b927afa8593f1e2bf28d4b63500b4cfcf27860f5e4d3a391bf83caa1f`
- main launch manifest SHA-256:
  `46ff9c0012ea94a0f2c06018c87f3015e262cad58383b252f60656515d993072`
- frozen treatment population manifest SHA-256:
  `521f8dfd71458fbd460af1a46943c4afb2fdeeac8cb17359d66cf73d503d45a3`

실제 initial bank는 다음과 같다. `raw`는 모델의 32개 denoising trajectory 전체이며,
`unique`는 materialized snapshot SHA-256으로 중복 제거한 수다.

| target | raw | unique | selected | NFE/forward calls | GPU ms | bank artifact SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| libpng | 32 | 26 | 16 | 590 | 70,795 | `ba59c4a8f2517d25edbc7a435fb9470975a99ee86a6a868b76435d59a560d65a` |
| libTIFF | 32 | 25 | 16 | 849 | 104,157 | `3dec2c797bd9d3c16bbbfde61269d7789a06a8ba883a07696a19fc93cb068283` |
| libxml2 | 32 | 26 | 16 | 455 | 64,120 | `3bdad740cfe337bb83b563756ecd0d384e31fbd2c642a22f8956f9befe4e0b53` |

예산 회계는 셀당 정확히 86,400 logical search worker-seconds다.

| arm class | initial probe | revision probe | main | logical total/cell |
| --- | ---: | ---: | ---: | ---: |
| `libfuzzer-only`, `structured-one-shot` | 0 | 0 | 86,400 | 86,400 |
| 세 feedback arm | 480 | 480 | 85,440 | 86,400 |

initial probe는 target별 동일한 16개 snapshot × 30초를 세 feedback arm이 공유하므로
물리적으로는 target당 480초, 논리적으로는 arm당 480초다. revision probe는 9개
target×feedback-arm 셀마다 16개 × 30초를 별도로 수행했다. 따라서 전체 search 비용은
다음과 같이 고정된다.

- logical search worker-seconds: `15 × 86,400 = 1,296,000`
- physical initial probe worker-seconds: `3 × 16 × 30 = 1,440`
- physical revision probe worker-seconds: `9 × 16 × 30 = 4,320`
- physical main worker-seconds: `6 × 86,400 + 9 × 85,440 = 1,287,360`
- physical search worker-seconds: `1,293,120`

시작된 컨테이너와 CPU 고정은 다음과 같다. 각 full container는 network none,
4 GiB memory/swap hard limit, PID 256, all capabilities dropped, no-new-privileges,
ASan enabled 상태다. exact argv, fuzzer seed, corpus file hash와 corpus tree hash는
`manifests/main-launches.json`에 보존된다.

| target | arm | CPU | main seconds | container ID |
| --- | --- | ---: | ---: | --- |
| libpng | libfuzzer-only | 0 | 86,400 | `975936ada31679ad70ab8de0297942eaa48affa6bcfb238ea98dc25bb2efa385` |
| libpng | structured-one-shot | 1 | 86,400 | `cc3d7af87fd2d3a8db9d94587782f018852a5ba3e58eb09d5cdb877f23973163` |
| libpng | feedback-no-remask | 2 | 85,440 | `408742dbf2a77c4ca64f1f69b0fbedf09d863b7b9f5d91d9685188e071910085` |
| libpng | selective-revision | 3 | 85,440 | `fa685a3fc98da2186549cff02b4bbeb9ac1265b966bb5e5d2f474329842506fc` |
| libpng | pseudo-feedback | 4 | 85,440 | `fe87f6f8551ea9179a6b78ccc306b14edb5e894cc33b02b81129125823cd0dfd` |
| libTIFF | libfuzzer-only | 5 | 86,400 | `2733fa91a6fb3691d189e213f64cbeb2e2b1635ddc3385838ba934a9b5cb068b` |
| libTIFF | structured-one-shot | 6 | 86,400 | `ce03e0e6fb98d301f8141b85d4785c83a1bace9ed0a96fbe5826eb617edab2e8` |
| libTIFF | feedback-no-remask | 7 | 85,440 | `294a5907a0cc042aafd8c62c09de0e480e283f77b53b00d3a23a20301b8d6ab6` |
| libTIFF | selective-revision | 8 | 85,440 | `15f480529ed244b65de2e26cdb9f1875a9944615dfe5c6929e8299832ba07e39` |
| libTIFF | pseudo-feedback | 9 | 85,440 | `5fc7601016c2dfd607fe9290264edfe2d3b95728d10bbbe450da418b64191f53` |
| libxml2 | libfuzzer-only | 10 | 86,400 | `c1538813653a6ea19f891f21cd244d7b482758e8c5932814baf901be00dc4829` |
| libxml2 | structured-one-shot | 11 | 86,400 | `9792a05ce0baa2e1310728404a089b74f85f09ef9b3e75cf92fe9e511078f5a3` |
| libxml2 | feedback-no-remask | 12 | 85,440 | `0d9ff762b8b1dba3d821595b287e91029fb05a2269519f47cd75807b6c7ae7f3` |
| libxml2 | selective-revision | 13 | 85,440 | `f258734b8643b5b8eecf689bf8dad32e151587211dd8d7025908b559a038cfb7` |
| libxml2 | pseudo-feedback | 14 | 85,440 | `33ff354f70b69d5220e97ee2b94859cdd1d9d5328b7955c34069f49daf371f26` |

이 시점의 15개 Docker state는 모두 `running`, `OOMKilled=false`다. 모델 생성은 이미
끝났으며 본 캠페인 동안 GPU와 CPU 퍼징을 중첩하지 않는다. 종료 후 각 container의
실제 시작/종료 시각, exit/OOM 상태, 요청·관측 worker-seconds, corpus/findings hash를
추가 기록하고, 발견 입력은 별도의 nonfatal ASan proof image에서 두 차례 독립 재현한다.

## 2026-08-21T19:31:14+09:00 - 6시간 운영 체크포인트

UTC `2026-08-21T10:31:14Z`에 15개 full container를 직접 `docker inspect`했다.
15/15 supervisor가 모두 `running`이고 조기 종료는 0개였다. 실행 원장은 launch 뒤
추가 기록 없이 그대로였으며, 해당 파일 SHA-256은
`9a2be178c4bcf42acddb851c29f16df448b95cac664fb7705a77938724c1a634`였다.
가용 디스크는 정확히 `84,604,239,872` bytes였다.

Docker cgroup OOM event flag는 다음 세 libTIFF 셀에서 `true`로 바뀌었다.

- `libtiff/structured-one-shot/rep-0`
- `libtiff/feedback-no-remask/rep-0`
- `libtiff/selective-revision/rep-0`

세 셀의 supervisor는 모두 계속 `running`이었다. 이는 앞서 검증한 Magma/libFuzzer
fork child OOM과 동일한 관측 경계다. 따라서 이 시점에는 인프라 실패나 sanitizer
증명으로 분류하지 않고 raw container state로만 보존한다. 최종 판정은 supervisor의
실제 종료 코드·종료 시각·timeout/process error와 독립 proof replay를 함께 사용한다.

## 2026-08-23 - 종료, 복구 수집, 독립 ASan 증명

Codex 실행 세션 중단으로 parent Python runner만 종료됐지만, detached Docker main
campaign은 계속 실행되어 15/15 셀이 모두 supervisor exit code `0`으로 종료됐다.
검색은 다시 실행하지 않았다. 동결된 `main-launches.json`에서 컨테이너 ID와 셀 배치를
복원해 기존 corpus, finding, log를 수집하고 해시로 결속했다.

- 86,400초 main 셀 6개의 실제 경과시간: `86,400.629–86,401.337초`
- 85,440초 main 셀 9개의 실제 경과시간: `85,464.834–85,525.370초`
- feedback 셀은 main 외에 initial/revision probe 각 480 logical seconds를 포함하므로
  15/15 셀 모두 동결된 `86,400 logical search worker-seconds` 예산을 충족했다.
- libTIFF의 `structured-one-shot`, `feedback-no-remask`, `selective-revision`은 Docker
  `OOMKilled=true`가 남았지만 supervisor는 exit code `0`이고 요구 시간을 모두 채웠다.
  이는 fork child OOM 관측이며, 해당 자체만 sanitizer 증명으로 세지 않았다.

### 최종 퍼징 지표

주 비교인 `selective-revision` 대 `pseudo-feedback` 결과는 다음과 같다.

| target | coverage 변화 | feature 변화 | executed units 변화 | coverage / 1M executions 변화 |
| --- | ---: | ---: | ---: | ---: |
| libpng | -5.28% | -11.11% | -53.63% | +104.25% |
| libTIFF | +31.51% | +26.33% | -62.41% | +249.87% |
| libxml2 | +7.47% | +18.84% | -30.92% | +55.56% |
| 기하평균 | +10.21% | - | -50.62% | +123.18% |

이는 이 1회 실험에서 selective revision이 적은 실행 횟수로 더 높은 coverage를 얻은
효율 신호다. 다만 libpng에서는 최종 coverage가 낮았고, 반복 수가 `n=1`이므로 통계적
우월성이나 일반화를 주장하지 않는다.

벤치마크 canary는 총 9개(`PNG003`, `PNG006`, `PNG007`, `TIF007`, `TIF012`,
`TIF014`, `XML009`, `XML012`, `XML017`)가 도달됐다. selective와 pseudo의 최초
논리 trigger time 비교는 libpng `975s 대 980s`, libTIFF `975s 대 1,915s`,
libxml2 `1,280s 대 39,830s`였다. canary 도달은 benchmark reachability 증거이며,
그 자체를 취약점 증명으로 세지 않았다.

### 모든 crash 후보의 독립 proof

`crash-*` 입력을 target과 콘텐츠 SHA-256으로 중복 제거해 52,158개를 proof 대상으로
고정했다. 12개의 독립 CPU pair(`15–38`)와 target별 nonfatal ASan image에서 각 입력을
새 Docker container로 정확히 두 번 재생했다.

- proof 대상: `52,158`
- 실제 기록된 독립 replay: `104,316` (`2 × 52,158`)
- proof wall time: `5,224.864716초`
- `proved`: `7`
- `not_reproduced`: `52,151`
- `failed`: `0`
- `crash_without_sanitizer`: `0`

7개 입력은 스택 signature 기준 2개 원인군이다.

| target | 발견 arm | 입력 수 | 두 번 일치한 sanitizer signature |
| --- | --- | ---: | --- |
| libpng | `libfuzzer-only` | 1 | `SEGV`, `png_do_expand_palette` (`pngrtran.c:4367:40`) |
| libxml2 | `structured-one-shot` | 6 | `heap-use-after-free`, `xmlDictComputeFastKey` (`dict.c:455:14`) |

`selective-revision`, `feedback-no-remask`, `pseudo-feedback`에서는 독립 ASan proof를
통과한 입력이 없었다. 따라서 이번 결과는 dLLM-native selective revision의
coverage/실행 효율 신호는 지지하지만, 검증된 취약점 발견 우월성은 지지하지 않는다.
또한 대상은 Magma bug patch/canary가 포함된 고정 benchmark build이므로 이 2개 원인군을
제로데이나 현재 upstream의 신규 취약점으로 주장하지 않는다.

### 최종 증거 결속

- 결과 보고서 SHA-256: `c3113bfe017ee0fcbfae58fc1db58ab938867015690e2de06c8c4c080e978605`
- proof batch index SHA-256: `a44d9ce9b0fd91e5b1412073d8e1748e1b20c109a8b5803c0fdc1ba26f18885a`
- append-only ledger file SHA-256: `f2f7ce534b4d704f93b81709f547a129a0691ae57927d33b3304f04098f95a2f`
- ledger final stream SHA-256: `f6861b25cb86f60bdc8ed4e40f0adf24d1bcca5ff3f2db7a3775fa2bf1dd95ab`
- immutable experiment evidence-tree SHA-256:
  `ebd3f469d2685a36a45cedd067909169268ec9746d9209a7eb7e1621e74b5992`
