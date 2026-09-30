# 단일 GPU Blackwell 워크스테이션: OpenEXR 스모크 실행 결과와 harness 합성 타당성 (2026-09-20)

브랜치 `feat/single-gpu-blackwell-hardware-readiness`. 이 문서는 (1) 실행 환경 구성, (2) OpenEXR
one-day 파이프라인의 qualification smoke를 실제로 돌린 과정과 성능, (3) trajectory-guided harness
합성 타당성 시뮬레이션, (4) 세 가지를 합친 결론을 정리한다. 모든 수치는 이 워크스테이션의 실측이며,
표본이 작아 통계적 주장은 하지 않는다.

## 1. 환경 구성

| 항목 | 값 |
|---|---|
| CPU / 메모리 | AMD EPYC 9554 64코어(128 스레드), RAM 251 GB |
| GPU | RTX PRO 6000 Blackwell Max-Q 96 GB × 2 (드라이버 570.211.01). campaign은 물리 GPU 1만 사용(`CUDA_VISIBLE_DEVICES=1`), GPU 0은 다른 작업이 점유 |
| OS / 도구 | Ubuntu 24.04(커널 6.8), Docker 29.1.3, uv 0.12.13, Python 3.12 |
| 프로젝트 venv | torch 2.11.0+cu128(sm_120 Blackwell 지원), transformers 5.17.0 |
| 검색 모델 | Stable-DiffCoder-8B-Instruct, revision `10cdaf9…`(BF16, 약 16.5 GB VRAM), 읽기 전용 캐시 `/data/dllm-agent/cache` |
| Dream-Coder-7B | transformers==4.46.2 필요 → 프로젝트 venv와 분리(아래 3절). torch 2.11+cu128은 프로젝트 것을 그대로 공유 |
| 대상 | OpenEXR v3.4.14(커밋 `777d231…`), 이미지 `dllm-openexr-runtime`(clang 18.1.8, ASan/UBSan 검색 빌드 + 커버리지 빌드, 이미지 digest `sha256:a2450af…`) |
| 캠페인 설정 | 1-GPU 파일럿(`configs/openexr-v3.4.14-one-day-1gpu.json`): 4개 model arm이 GPU 하나를 time-multiplex, candidate_steps=8, 입력 상한 1 MiB, 타깃 timeout 10 s, RSS 한도 2 GiB |

환경에서 해결한 문제:
- Blackwell은 CUDA 12.8+ / torch ≥ 2.7이 필요해 cu128 인덱스의 torch를 pin했다(README에 기록).
- Dream-Coder 원격 코드가 transformers 4.x 전용이라 프로젝트 venv를 건드리지 않고, 4.46.2를 얹은
  별도 venv를 프로젝트 torch 위에 겹쳐 사용했다(`pypi.nvidia.com`이 타임아웃이라 새 torch 설치는 불가).

## 2. OpenEXR one-day 파이프라인: 실제 실행과 성능

### 2.1 무엇을 돌렸나

`tools/start_smoke.sh`가 시작 시각을 찍고 `smoke-run`을 실행한다. 구성은 5개 arm × 2회 반복 = 10개 셀이고,
각 셀은 300초 고정 검색 창이다.

arm: `libfuzzer-only`(모델 없음), `final-only`, `structured-one-shot`, `selective-revision`, `full-regenerate`.

단계: 모델 prep(후보 bank 생성) → 셀별 300 s libFuzzer 검색 → 사후 처리(proof 재현, deep-parse 재생,
t=0/t=300 커버리지 시리즈 재계산) → 완료 후 재시작(resume) 검증.

### 2.2 처음 실패한 원인과 수정

이전 시도는 사후 처리의 deep-parse에서 `requested build provenance mismatch`로 중단됐다. config가
템플릿에 얼려 있던 checker 해시(`621ca0…`)를 쓰는데, 이 워크스테이션에서 다시 빌드한 이미지의 checker는
바이트가 달라 `326bf9…`였다. `finalize-config`가 빌드 manifest의 `deep_parse_checker_sha256`를 그대로
config에 새기도록 고쳤고(`be59a45`) 테스트를 추가했다. 이후 config를 재생성하고 재실행했다.

### 2.3 결과

- **통과**: `passed: true`, 재시작 검증 pair 0·1 모두 성공, 셀 10개 모두 proof·deep-parse 결과 산출.
- **소요 시간**: 18:18 시작 → 02:42 종료, 약 8시간 24분. 검색 자체(300 s × 10 셀, 모델 prep 포함)는 시작 후
  약 1시간 이내에 끝났고, 나머지 대부분은 사후 처리의 커버리지 재생(셀당 약 4,600~4,800개 입력을 profraw로
  재실행, 셀 하나에 약 30~50분)이 차지했다.
- **모델 prep**(bank 공유, arm별 GPU 사용): bank 준비 17.9 s. arm별 forward 호출 160~290회, GPU 시간
  2.8~5.1 s로 모델 자체 비용은 매우 작다.
- **디스크**: 스모크 산출물 51 GB(대부분 커버리지 profraw/로그).
- **크래시 0건**(모든 셀).

셀별 커버리지(전체 58,441 라인 중 라인 커버리지, t=300 s):

| arm | rep 0 | rep 1 |
|---|---|---|
| libfuzzer-only | 6,078 | 6,284 |
| final-only | 6,040 | 6,500 |
| structured-one-shot | 6,091 | 6,129 |
| selective-revision | 6,058 | 6,423 |
| full-regenerate | 6,143 | 6,729 |

라인 커버리지는 약 10.3~11.5%. 리전 커버리지도 같은 순서(3,383~3,692 / 37,717).

### 2.4 해석의 한계

이 실행은 **파이프라인 qualification**이다(모든 단계가 끝까지 동작하고 재현·재개가 되는가).
효능 실험이 아니다. n=2, 300 s에서는 arm 간 차이를 주장할 수 없다(같은 arm의 rep 0과 rep 1 차이,
예: full-regenerate 6,143 vs 6,729가 arm 간 차이보다 크다). 취약점 발견이나 dLLM 우월성을 주장하지 않는다.
알려진 잔여 문제: `test_write_record_backed_qualification_writes_canonical_evidence` 하나는 내 변경과
무관하게 실패하며 원인을 아직 조사하지 않았다.

## 3. Harness 합성 타당성 시뮬레이션

상세는 `docs/harness-synthesis-feasibility-20260920.md`. 요약:

- 메커니즘: Dream `diffusion_generate`의 `generation_tokens_hook_func(step, x, logits)`로 step별 trace
  기록, 중간 remask, 초기 상태 infill이 모두 가능하다(생성 1초, VRAM 15.8 GB). 러닝타임 이미지가 이미
  `/work/harness.cc` 볼륨 마운트 컴파일(`build-coverage`)을 지원해 새 컴파일 컨테이너가 필요 없다.
- 신호: 최저 confidence 영역이 첫 컴파일 오류 줄과 일치한 비율 **11/16**(무작위 영역 3/16).
- 한계: 프롬프트만으로 생성한 harness는 **0/48 컴파일**(환각 API/헤더).
- 컴파일 오류 피드백 + 지식베이스(arm당 8 seed, 최대 3회 수정):
  resample 0/8, 오류 국소화 remask 0/8, +헤더 KB 0/8(헤더 환각은 사라지나 C++ 문법 오류로 실패),
  +공유 KB 0/8(실제로는 한 번도 갱신되지 않음), **+검증된 사용 예시 8/8**(전부 첫 시도).
  예시 기반 harness는 예시 호출을 거의 그대로 복제하고, 커버리지는 1,793 edge로 reference의 1,921보다 낮다.

## 4. 종합

1. **하드웨어·파이프라인 준비는 끝났다.** Blackwell 한 장에서 모델 로드, 검색, 사후 처리, 재개 검증까지
   전체 스모크가 통과했다. 병목은 GPU가 아니라 사후 처리의 커버리지 재생(디스크·CPU)이며, 모델 비용은
   셀당 GPU 수 초 수준이다. 본 실험(one-day, 5 arm × 반복)을 돌릴 때는 이 사후 처리 시간(셀당 30~50분)과
   산출물 용량(스모크만 51 GB)을 일정·디스크 예산에 반영해야 한다.
2. **harness 합성은 가능하지만 명세를 바꿔야 한다.** 트레이스와 remask 메커니즘은 동작하고, confidence는
   오류 위치를 무작위보다 훨씬 잘 가리킨다. 그러나 이 모델은 컴파일 오류 피드백만으로는 고치지 못하며,
   검증된 API 사용 예시가 KB에 있어야 컴파일된다. 다만 그 경우 harness가 예시로 수렴해 다양성이 사라진다.
3. **두 실험은 아직 연결되지 않았다.** 스모크의 harness는 고정된 `poc_probe.cc` 하나이고, 합성 실험은
   harness를 바꾸는 쪽이다. 연결 지점은 이미 있다(볼륨 마운트 컴파일, 반복마다 harness SHA-256을
   append-only 원장에 기록하는 결정 사항).
4. **다음 결정**: (a) 여러 사용 예시를 가진 공유 KB가 서로 다르고 reference와도 다른 harness를 만드는지
   검증하는 소규모 실험을 먼저 할지, (b) 곧바로 harness 실험 setup 구현 계획으로 갈지.

산출물: `tools/harness_synthesis_feasibility/`, `docs/assets/harness-feasibility-*.json`,
스모크 원본 `/data/dllm-agent/openexr-one-day-1gpu/smoke/`.
