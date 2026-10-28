# Compiler-Localized Bidirectional Span Infill (RQ4) — Design

> 상태: **설계 + offline step 0 완료, 1-round GPU pilot 완료(2026-10-26, §8).** 동결
> 프로토콜은 `tools/harness_synthesis/rq4_localized_infill_pilot.py` docstring.

## 1. 출발점 — 팀원 피드백

PR #3(RQ1/RQ3 종료)를 검토한 연구 동료의 방향 제시:

> "위치는 컴파일러나 퍼저가 알려주고, dLLM은 양방향 문맥으로 그 구간만 다시 생성하는
> 방향으로 가면 다시 시도해볼만 하긴 합니다"

이것은 RQ1/RQ3을 다시 여는 것이 아니다. RQ1/RQ3이 한 질문("**어디를** 고칠지 모델 자신의
신호로 고를 수 있는가")을 버리고, 역할을 나눈다:

| 역할 | RQ1/RQ3 | RQ4 |
|---|---|---|
| 위치 결정 | 모델의 commit confidence/entropy/margin | **외부 oracle** — 컴파일러 진단(1단계), 퍼저 리포트(3단계) |
| 수정 | 같은 모델, remask | 같은 모델, **그 구간만** 양방향 문맥으로 infill |
| dLLM 고유성이 걸린 곳 | 위치 신호(trajectory) | 구간 재생성 시 **오른쪽 문맥**을 조건으로 쓸 수 있다는 점 |

## 2. PR #3 결과를 이 방향에서 다시 읽기

`docs/superpowers/plans/2026-09-29-trajectory-signal-harness-synthesis.md` §8의 결론은
그대로 유효하다. 이 방향과 관련해 그 데이터가 이미 말해주는 것은 세 가지다:

1. **모델 자신의 위치 신호는 실패했다.** RQ1 224 표본에서 confidence가 컴파일 오류 region을
   맞힌 비율은 5–7/112로 random(11–14/112)보다도 낮았다. 위치를 외부에서 받는 것은 이
   null 결과를 직접 우회한다.
2. **"그 구간만" 재생성하는 기제는 이미 작동한다.** RQ3-v3-window pilot에서
   `remask_window`(window 밖 토큰 고정, 양쪽 문맥 유지)는 24/24 arm에서 window 밖 보존이
   코드로 검증됐다. 그 pilot이 실패한 이유는 기제가 아니라 *window를 confidence로 골랐기
   때문*이다.
3. **compile 실패 대부분은 한 줄짜리 API 오용이다.** v3 pilot reference 실패 10건 중 9건이
   존재하지 않는 메서드(`headerVersion()`, `fileHeader()` …), `&stream` 포인터 전달,
   잘못된 타입명이었다 — 국소적이고, 한 줄만 고치면 되는 형태.

## 3. Step 0 — offline localizability (완료, GPU 불필요)

`tools/harness_synthesis/rq4_offline_localizability.py`가 이미 저장된 RQ3-v3 결과 3개
(feasibility, region pilot, window pilot)의 **모든 컴파일 실패 reference 본문**의 clang
오류를 production `localize` 모듈로 재분석했다. 결과
`docs/assets/harness-rq4-offline-localizability-20261013.json`:

| | 값 |
|---|---|
| 컴파일 실패 reference | 27 |
| 첫 오류가 모델이 쓴 본문 줄에 정확히 매핑 | **27/27** |
| 모든 진단이 본문 안 | 27/27 |
| 위치 줄 뒤에 모델이 쓴 코드가 있음(오른쪽 문맥) | **14/27** |
| 위치 줄 앞에 모델이 쓴 코드가 있음(왼쪽 문맥) | 16/27 |

해석:
- 위치 문제는 이 substrate에서 **compiler가 사실상 완벽하게 푼다**(confidence 5–7/112와
  대조). 고정 scaffold가 모델 쓰기 영역을 본문으로 제한했기 때문에 가능한 수치이며, 전체
  harness 생성(v1/v2)으로 일반화되지 않는다.
- 하지만 **양방향 문맥이 의미 있는 경우는 절반뿐**이다. 나머지 13건은 오류가 본문 마지막
  줄이라 "그 줄만 infill"과 "그 줄부터 끝까지 다시 생성(left-to-right)"이 사실상 같은
  개입이다. 그래서 H-bi는 오른쪽 문맥 stratum에서만 1차 분석한다(§4).
- 표본이 같은 프롬프트·같은 모델의 27건이고 사후 분석이다. pilot 게이트 (1)이 이 수치가
  새 seed에서도 유지되는지 확인한다.

## 4. RQ4 pilot 설계 (동결, 미실행)

전체 프로토콜은 스크립트 docstring이 원본이다. 요약:

- **모집단**: RQ3-v3 고정 scaffold, 같은 프롬프트·생성 설정. reference가 컴파일 실패하고
  첫 오류가 본문 줄로 매핑되는 trial만 eligible. 예시당 eligible 24건 또는 96 시도.
- **arm** (같은 실패 본문, 같은 재생성 seed):
  1. `located-infill` — compiler가 지목한 줄의 토큰 window를 `M`개 mask로 교체, 앞뒤 유지
  2. `random-infill` — 같은 `M`, 본문의 **다른** 줄 하나(무작위)
  3. `located-suffix` — 지목한 줄부터 끝까지 remask(같은 모델의 left-to-right 수리)
  4. `resample` — 처음부터 다시 생성(새 seed)
- **비교**: H-loc(1 vs 2), H-bi(1 vs 3, 오른쪽 문맥 stratum), anchor(1 vs 4). 각각
  discordant-pair exact sign test. pilot이므로 확증 주장 없음 — 이 pilot의 discordance율로
  확증 N을 `power_calc.required_n_from_discordant_pairs`로 계산한다(기존 원칙).
- **1차 지표**: 컴파일 성공. 2차: 첫 오류가 지목 줄을 벗어났는가, ITT coverage(fuzz seed
  1·2·3 × 20 s), 마스크 토큰 수.
- **seed**: 12,000,000대 예약.

### 새로 필요했던 기제

- `localize.py`(신규): clang 진단 파싱 → 본문 줄 매핑 → 그 줄을 덮는 연속 토큰 window.
- `remask.infill_window`(신규): window를 **다른 길이**의 mask로 교체. `remask_window`는
  길이를 유지해야 해서 고친 줄이 길어지거나 짧아질 수 없었다. infill은 `M = 줄 길이 + 8`
  slack을 주고, 쓰지 않은 slack 자리에 모델이 낸 special token은 span 안에서만 제거한다.

## 5. 알려진 위험과 미결정

1. **고정 길이 canvas.** Dream은 mask 수만큼 정확히 토큰을 채운다. slack + special-token
   제거가 실제로 깨끗하게 동작하는지(모델이 slack을 pad로 채우는지, 공백·주석으로 채우는지)는
   pilot 전에는 모른다 — 실패 시 기록되는 `preserved`/컴파일 오류로 드러난다. 가변 길이
   infill 자체를 다루는 관련 연구(Dream 계열의 DreamOn — expand/delete 토큰)는 실행 전에
   확인할 가치가 있다.
2. **위치만으로는 부족할 수 있다.** 실패 대부분이 "존재하지 않는 메서드"라서, 그 줄을
   다시 생성해도 같은 환각을 반복할 수 있다. 진단 메시지(`no member named 'fileHeader'`)를
   프롬프트에 넣는 것은 별도 요인이라 이번 pilot에서는 **의도적으로 뺐다**(프롬프트를 arm
   간 동일하게 유지). pilot에서 `located-infill`이 `resample`도 못 이기면 2단계 첫 후보다.
3. **dLLM vs AR 주장 불가.** `located-suffix`는 같은 Dream 모델의 left-to-right 디코딩이지
   AR 모델이 아니다. AR 모델도 fill-in-the-middle(FIM)로 양방향 infill을 할 수 있으므로,
   "양방향 문맥이 dLLM의 이점"이라고 말하려면 FIM을 지원하는 AR control(예: 이미 후보로
   기록된 Qwen2.5-Coder-7B — 지원 여부·포맷은 실행 전 확인)과 같은 위치·같은 예산으로
   비교해야 한다. 이번 pilot은 그 전 단계다.
4. **퍼저 쪽 위치(3단계).** 이 substrate는 harness가 컴파일되면 대부분 바로 coverage가
   나와서, 퍼저가 harness 줄을 지목하는 경우(ASan/UBSan stack frame이 `harness.cc:LINE`인
   harness 자체 결함, timeout, 미도달 API)가 얼마나 생기는지 아직 측정한 적이 없다.
   `localize.parse_compiler_diagnostics`와 같은 형태로 sanitizer frame 파서를 추가하면 같은
   infill 경로를 재사용할 수 있다.

## 6. 단계와 게이트

| 단계 | 내용 | 진행 조건 |
|---|---|---|
| 0 | offline localizability (완료) | — 27/27, 오른쪽 문맥 14/27 |
| 1 | 1-round pilot (위 §4, **완료 — §8**) | — 게이트 (1)(2) 통과, H-bi 반대 방향 |
| 2 | 다회 루프(compile → locate → infill, 최대 3 round) + 진단 메시지 요인 | 1단계 게이트 통과, H-loc/anchor 방향 확인 |
| 3 | 퍼저 리포트 기반 위치 + FIM AR control | 2단계 결과 검토 후 별도 결정 |

## 7. 쓰지 않을 표현

"compiler-localized infill이 효과가 있다"(pilot은 확증이 아님), "dLLM이 AR보다 낫다"(AR
control 없음), "양방향 문맥이 원인이다"(오른쪽 문맥 stratum 밖에서는 H-bi가 정의되지 않음).

## 8. 1-round pilot 결과 (2026-10-26, GPU 2장, 동결 프로토콜 그대로)

원시 결과 `docs/assets/harness-rq4-pilot-{inputfile,multipart}-results-20261026.json`, 원장
`docs/assets/harness-rq4-pilot-ledger-20261026.jsonl`(260줄 = reference 68 + eligible 48 × 4
arm), 분석 `docs/assets/harness-rq4-pilot-analysis-20261026.json`
(`tools/harness_synthesis/rq4_analyze.py`). 실행 전 seed 12,900,000으로 1회 스모크(데이터셋
밖, 결과는 버림)로 스크립트만 확인했다.

### 8-1. 게이트

- 시도 68(InputFile 33, MultiPartInputFile 35), reference 컴파일 20, eligible **48**(예시당 24).
- 게이트 (1): 컴파일 실패 48건 중 본문으로 위치를 못 찾은 것 **0건** — offline 27/27이 새
  seed에서도 재현됐다. 게이트 (2): 통과.
- 오른쪽 문맥 stratum 32/48(offline 14/27과 비슷한 비율).

### 8-2. 1차 결과 (컴파일 성공, eligible 48)

| arm | 컴파일 | ITT 평균 edges | 평균 mask 토큰 |
|---|---|---|---|
| located-infill | 10/48 | 219 | 22.1 |
| random-infill | 0/48 | 0 | 22.1 |
| located-suffix | **22/48** | 535 | 48.5 |
| resample | 16/48 | 392 | 60 |

| 비교 | located-infill만 성공 | 상대만 성공 | 동률 | exact sign test p |
|---|---|---|---|---|
| H-loc (vs random-infill) | 10 | 0 | 38 | 0.002 |
| **H-bi, 오른쪽 문맥 stratum (vs located-suffix)** | 2 | **12** | 18 | **0.013** |
| H-bi, 오른쪽 문맥 없음 (별도 보고) | 2 | 4 | 10 | 0.69 |
| anchor (vs resample) | 7 | 13 | 28 | 0.26 |

예시별로도 방향은 같다(H-bi 오른쪽 문맥: InputFile 0 대 7 p=0.016, MultiPartInputFile 2 대 5
p=0.45). 프로토콜은 가설별 개별 검정만 정했고 예시 간 pooling 여부는 명시하지 않았으므로,
pooled와 예시별 수치를 함께 적는다. infill arm 96건 전부 span 밖 토큰·scaffold 보존이 검증됐다.

해석(pilot, 확증 아님):
- **H-loc는 거의 구성상 참이다.** random-infill은 오류 줄을 건드리지 않으므로 고칠 수가
  없다 — 0/48은 "위치 정보가 의미 있다"는 sanity check이지 동료 가설의 독립 증거는 아니다.
- **H-bi는 예측의 반대 방향이다.** 이 구현의 양방향 infill은 같은 위치에서 끝까지 다시 쓰는
  left-to-right 수리보다 **유의하게 나빴고**(12 대 2), 처음부터 다시 생성하는 것(resample)도
  이기지 못했다.

### 8-3. 사후 특성화 (동결 분석 밖 — 원인 설명용, 판정에 쓰지 않음)

1. **slack이 내용으로 채워진다(§5 위험 1이 실제로 발생).** located-infill의 교체 텍스트가
   원래 줄보다 1.3배 넘게 긴 경우가 **47/48**. 모델이 남는 mask 자리를 pad token으로 비우지
   않고 인자·문장·주석을 더 써서 채운다 — 예: `InputFile inputFile(&stream);` →
   `InputFile inputFile OPENEXR_IMF_NAMESPACE::InputFile(stream);`. 고정 길이 canvas 위에서
   `M = 줄 길이 + 8`을 강제로 채우게 한 설계 자체가 새 오류를 만든다. suffix arm은 원래 생성
   예산(본문 뒤 꼬리 포함)이 커서 모델이 줄을 끝내고 나머지를 pad로 흘려보낼 수 있다.
2. **컴파일 성공의 상당수가 "고친" 게 아니라 "지운" 것이다 — 모든 arm 공통.** 컴파일된 본문
   중 constructor만 남고 header 검사 호출이 사라지거나 주석 처리된 경우: located-infill
   **10/10**, located-suffix 17/22, resample 10/16. 즉 1차 지표(컴파일)는 API 오용을 올바른
   호출로 바꾼 것과 그 줄을 지운 것을 구분하지 못한다. edges는 constructor만으로도 대부분
   나오므로 ITT coverage도 이를 구분하지 못한다.
3. 같은 줄에 다시 환각(`fileHeader` → `readHeader` 등)하는 패턴은 그대로 남아 있다(§5 위험 2).

### 8-4. 결론과 다음 결정

- **관측**: 컴파일러 위치는 이 substrate에서 신뢰할 수 있다(새 seed 48/48). infill 기제는
  span 밖을 정확히 보존한다(96/96).
- **결과**: 이번 구현(고정 slack 8, 진단 메시지 없음)의 양방향 span infill은 같은 위치의
  left-to-right 수리보다 컴파일률이 낮았고 resample도 넘지 못했다. **"양방향 문맥이 수리를
  돕는다"는 이 pilot에서 지지되지 않는다** — 다만 관측된 주원인이 문맥이 아니라 고정 길이
  canvas의 slack 강제 채움이라, 이 결과는 "양방향 문맥이 무익하다"가 아니라 "이 infill
  구현으로는 이점이 드러나지 않았다"까지만 말한다.
- **지표 문제**: 다음 설계는 "검사 호출을 유지하면서 컴파일"(또는 constructor 외 API 호출
  보존)을 1차 지표에 넣어야 한다. 지금 지표로는 삭제가 수리로 집계된다.
- **확증 실험은 하지 않는다.** 다음 후보(서로 배타적이지 않음, 사용자 결정):
  (a) 가변 길이 문제 제거 — slack 0을 포함한 여러 mask 길이 후보를 만들고 컴파일러를
  verifier로 best-of-k(같은 k를 suffix/resample에도 부여), 또는 가변 길이 infill을 지원하는
  모델(DreamOn 계열) 확인; (b) 진단 메시지를 프롬프트에 넣는 요인; (c) 위 지표 수정.

### 8-5. 후속: 사후 감사와 RQ4b (2026-10-26, 위 결과는 수정하지 않음)

[사후 감사](2026-10-26-rq4-retrospective-audit.md)는 위 수치를 그대로 재현했고(재계산 일치,
재컴파일 불일치 0), 다음을 덧붙인다. 오른쪽 문맥 stratum 32건 중 17건은 뒤에 주석만 있고,
13건(전체 48건 중 13건, 저장된 앞 3줄 기준으로는 12건)은 여러 줄에 오류가 있어 한 줄 infill로는
원리상 고칠 수 없는 경우가 많다(oracle 생성자 수리 후에도 10건 잔존). 의도한 비교 대상(뒤에
실행 코드가 있고 오류가 한 줄)은 2건뿐이라, 8-4의 2 대 12는 양방향 문맥 효과에 대해 어느 쪽으로도
말하지 못한다. 8-3의 slack 강제 채움(47/48)은 독립 재측정으로 확인됐다. 이 교란을 구조적으로
제거한 후속 설계가 [RQ4b](2026-10-26-rq4b-functional-repair-benchmark.md)다(동결, 미실행).

