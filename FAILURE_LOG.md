# FAILURE_LOG.md — 실패 / 시행착오 / 막힌 지점 기록

> 본 문서는 파이프 자동배치 강화학습 v2 프로젝트의 **진행 중 발생하는 실패 사례** 를 기록한다.
> PROGRESS.md 가 "왜 결정했나" 의 archive 라면, 본 문서는 **"무엇에서 막혔나"** 의 archive 다.
> 시간이 쌓이면 본 문서가 프로젝트의 가장 가치 있는 개인 자산이 된다.
>
> **🎓 2026-06-24 졸업판 동기화**: 다른 spec 파일 (CLAUDE/SKILL/PROGRESS) 의 \_ing 졸업과 함께 cross-reference 갱신.
> 본 문서는 파일명 변경 없음 (이미 \_ing 없었음). 과거 entry 안의 "\_ing 시스템" 표현은 그 시점의 역사적 사실로 보존.
>
> **§0.1 메커니즘 (2026-06-18 신규)**: 본 archive 의 entry 는 단순 학습 자료가 아니라 **regression set 의 dynamic source**. 새 entry 추가 시 해당 실패 케이스가 수작업 seed pool 로 영구 등록됨 (CLAUDE.md §11.0.5).

---

## 0. 본 문서의 역할

```
PROGRESS.md  → WHY  (왜 이 결정을 내렸나)
FAILURE_LOG  → WHAT WENT WRONG (무엇에서 막혔나)
```

같은 실수를 두 번 하지 않기 위해, 그리고 동일 패턴의 문제를 빨리 식별하기 위해 작성한다.

### 0.1 본 문서의 확장된 역할 (2026-06-18 신규, L-A5 해제 반영)

2026-06-18 L-A5 해제로 본 문서의 역할이 **passive archive → regression set 의 dynamic source** 로 확장됨.

```
[메커니즘]
새 entry 추가 시 (학습/평가 실패):
  1. 실패 상황의 grid / start / goal / 기타 메타데이터를 YAML 로 추출
  2. 해당 Step 의 수작업 seed pool 로 영구 등록
     (위치: scenarios/stepN/manual_seeds/case_YYYYMMDD_NNN.yaml)
  3. 다음 학습/평가부터 회귀 / 최종 평가 set 에 자동 포함
  4. (선택) procedural generator 의 parameter 조정 검토 (비슷한 패턴 더 자주)

[효과]
실패한 적 있는 케이스가 영구 평가 대상으로 승격.
시간이 지날수록 수작업 seed 비율이 자연 증가 (최대 30% 권장).
본 문서는 단순 학습 자료가 아니라 **시스템의 실측 evidence base** 가 됨.

[운영 규칙]
- 수작업 seed 비율 30% 초과 시 procedural / 수작업 비율을 의식적으로 관리 (과대표집 방지)
- 수작업 seed 도 generator_version 호환성 추적 (algorithm 수정 시 영향 검토)
- 너무 trivial 한 실패 (예: 한 번의 OOM) 는 seed 등록 대상에서 제외 (운영 판단)
```

본 메커니즘으로 spec 의 §11.0.5 (Procedural 100% 시작, 수작업 점진 추가) 가 작동.

---

## 1. 작성 시작 시점

본 문서는 **Phase 2 (harness engineering 시작) 시점부터** 본격 작성된다.
현재는 placeholder 상태이며, 다음 시점에 첫 entry 가 작성된다:

- 첫 코드 작성 후 발생한 첫 버그
- 첫 학습 실패 (수렴 안 됨, OOM, 등)
- 첫 평가 metric 이상 (예상과 다른 결과)

---

## 2. 앞으로 기록할 내용 (예상 카테고리)

### 2.1 학습 관련 실패

```
- Reward sparse 문제 (학습 신호 부족)
- Reward hacking (의도 외 행동으로 reward 획득)
- 발산 (학습 중 loss 폭발)
- 수렴 정체 (특정 KPI에서 더 이상 개선 없음)
- catastrophic forgetting (이전 step 능력 상실)
- Action mask 충돌 (마스킹 후 가능 action 0개)
```

### 2.2 환경 / 인프라 실패

```
- OOM (메모리 폭발)
- Colab 세션 끊김 / 학습 중단
- Partial EDT 업데이트 버그
- 좌표계 변환 오류 (mm ↔ cell)
- 시각화 렌더링 실패
- 핸드오프 패키지 로드 실패 (state_dict key 불일치)
```

### 2.3 평가 체계 실패

```
- A* benchmark 계산 오류
- 평가 시나리오 편향 (특정 시나리오만 잘됨)
- 회귀 검증 누락 (이전 step 성능 저하 미감지)
- Pareto front 거리 계산 이상
- 물리 metric 측정 오류 (kg 단위, 길이 단위 등)
```

### 2.4 autoresearch 운영 실패

```
- 4-Gate 우회 (게이트 통과했지만 실제로 hack)
- 1차 스크리닝 오판 (좋은 변형이 짧은 학습에서 못 드러남)
- Bayesian Opt 수렴 안 함
- Sequential sweep 중 인자 간 interaction 놓침
- Claude API 변형 제안 품질 저하
```

### 2.5 Hierarchical 구조 실패

```
- Macro-Micro non-stationarity (학습 불안정)
- Selective unfreeze 후 회귀 발생
- Macro 후보 다양성 부족 (5개가 거의 같은 경로)
- Visibility graph 노드 수 폭발
- waypoint 간 구간이 너무 길거나 짧음
```

### 2.6 다중 파이프 실패 (Step 7~10)

```
- 양보(yielding) 학습 안 됨
- 우선순위 결정 모호
- 배치 순서에 따른 결과 차이
- 파이프 간 deadlock (서로 양보 기다림)
```

---

## 3. 표준 entry 양식

각 실패는 다음 양식으로 기록한다:

```markdown
### YYYY-MM-DD — [짧은 제목]

**상황:**
어떤 작업 중에 발생했는가

**증상:**
구체적으로 무엇이 잘못됐는가 (에러 메시지, 이상 결과 등)

**원인 분석:**
디버깅 결과 발견된 진짜 원인

**해결책:**
어떻게 고쳤는가 (코드 수정, config 변경, 등)

**교훈:**
다음에 같은 함정을 피하기 위해 기억할 것

**관련 파일/위치:**
이 실패와 관련된 코드/spec 위치
```

---

## 4. entry 작성 규칙

1. **즉시 작성**: 실패 발견 후 24시간 이내 (잊으면 손실)
2. **솔직하게**: 자기변호 금지, 무엇이 잘못됐는지 그대로
3. **검색 가능하게**: 키워드 명확히 (나중에 grep 가능)
4. **간결하게**: 한 entry 1페이지 이내, 길어지면 분할
5. **연관 표시**: 비슷한 패턴의 이전 실패와 연결 (`See: 2026-05-15 entry`)

---

## 5. 주기적 회고 (Quarterly Review)

3개월 / Step 완료 시점마다 본 문서를 다시 읽고:

- 반복 패턴 식별 (같은 종류의 실수가 N번 반복됐는가)
- 반복 패턴 발견 시 → SKILL.md 에 예방 규칙 추가
- 학습된 교훈을 CLAUDE.md 에 반영 (필요 시)

본 문서는 **passive archive 가 아니라 active learning resource** 다.

---

## 6. 첫 entry 작성 예시 (placeholder, 실제 아님)

```markdown
### 2026-05-15 — Step 1 reward sparse 로 학습 정체

**상황:**
Step 1 첫 학습 시도, 50,000 step 학습 후에도 success_rate 0%

**증상:**
- Episode reward 거의 0 근처에서 진동
- Goal 도달 거의 못 함
- 정책이 random 과 거의 차이 없음

**원인 분석:**
goal_bonus 가 100 으로 너무 작음. Episode 길이 200~500 step 이라
length penalty (-1 per step) 누적이 -200 ~ -500 인데
goal 도달 보상이 +100 이라 net negative. 도달해도 손해.

**해결책:**
goal_bonus: 100 → 500 으로 증가
추가로 distance-based shaping reward 도입 (CLAUDE.md §11.2 reward baseline 수정)

**교훈:**
sparse reward step 에서는 goal_bonus 가 episode 누적 penalty 보다 충분히 커야 함.
Episode length 분포를 먼저 측정하고 reward scale 결정해야 함.

**관련 파일/위치:**
- envs/step1_env.py L.142 (reward 계산)
- CLAUDE.md §11.2 (Step 1 baseline reward)
- ADR-003 (reward scale 결정 근거)
```

(위는 실제 발생한 실패가 아닌 양식 예시)

---

## §6.5 첫 entry (실제 발생, 2026-05-14 추가)

> 본 entry는 placeholder 가 아닌 실제 실패 기록. 본 문서가 active learning resource 로 전환되는 시점.

### 2026-09-06 — w1/w2/w3 가 CLI 인자·params dict 에는 있지만 실제 보상 계산에는 전달된 적이 없었음

**상황:**
Round 1(α/β) 을 var008(α=2.0, β=0.3) 고정으로 종결하고 Round 2(w1×w2×w3 sweep, §16.3.2,
`enqueue_round2_grid()` 로 이미 grid 존재)로 넘어가기 직전, "Round 2가 실제로 신호를
만드는지" 를 점검하다가 사용자가 "`train_fn` 에서 w1·w2·w3가 실제로 보상 함수에
전달되는 배선이 되어 있는지" 확인을 요청. 코드 추적 결과 배선이 아예 없었음을 발견.

**증상:**
- `training/train_step1.py::run_training()` 은 `--w1/--w2/--w3` CLI 인자를 파싱하고
  wandb config, `reward_config`, `HandoffConfig.extra`, 콘솔 출력(478행)에 전부
  기록하지만, 실제 학습에 쓰이는 `make_env_fn(args.seed+i, args.alpha, args.beta,
  args.difficulty)` 호출에는 넘기지 않았다.
- `COLAB_GUIDE.md` 의 `make_train_fn_real().train_fn` 도 `params.get("w1"/"w2"/"w3", ...)`
  로 값을 읽어 지역변수에 담기만 하고, 그 아래 `make_env_fn(...)` 호출에는 `alpha, beta`
  만 넘겼다.
- `make_env_fn()` 자체가 w1/w2/w3 를 받는 파라미터가 없었고, `Step1Env.__init__` 도
  w1/w2/w3 를 받을 방법이 없어 `_calc_reward()` 는 항상 모듈 레벨 상수
  `W1=0.1/W2=2.0/W3=50.0` 만 참조했다.
- 겉보기(로그, wandb config, 콘솔 출력)로는 값이 "정상적으로 기록"되고 있어서 배선이
  살아있다는 착시가 강했다 — 실제로 값이 쓰이는지 검증하는 절차가 없었다면 Round 2
  (36 variants, ~150시간/222 units 추정) 전체가 **w1/w2/w3 값과 무관하게 항상 동일한
  reward 함수로 학습되는 완전한 no-op** 이 됐을 것이다.

**원인 분석:**

[원인 1. "값을 읽고 기록한다" 와 "값이 실제로 쓰인다" 를 구분하는 검증 단계가 없었음]
CLI 인자 파싱(`parse_args`) → wandb config 기록 → handoff 기록까지 각 단계가 전부
정상 동작해서, 코드를 눈으로 훑을 때 "w1/w2/w3 가 파이프라인을 관통한다"는 인상을
준다. 그러나 이 모든 지점이 **읽기/기록**일 뿐, 실제 소비처(`_calc_reward()`)까지
값이 도달하는지 확인하는 단일 테스트도 없었다.

[원인 2. Round 1(α/β) 배선은 처음부터 됐던 게 오히려 함정]
`alpha, beta` 는 `Step1Env.__init__` 부터 `make_env_fn()` 까지 처음 설계 시점부터
정상적으로 관통해서, w1/w2/w3 도 "같은 패턴이니 당연히 됐겠지" 라고 가정하기 쉬운
구조였다. 실제로는 w1/w2/w3 가 CLAUDE.md §16.3.1 baseline reward 초기값으로만 먼저
구현되고(모듈 상수), 나중에 §16.3.2 Round 2 sweep 설계 시점에 CLI 인자만 추가됐을 뿐
그 인자를 소비하는 경로가 함께 추가되지 않았다 — 두 파라미터 그룹이 코드베이스에
들어온 시점이 달라서 생긴 배선 누락.

[원인 3. 2026-08-30 entry 와 동일 유형 — "spec 상수가 정의돼 있다고 실제로 쓰인다는
보장은 없다"는 교훈이 재발함]
2026-08-30 entry(screening 평가 버그)의 교훈 3번 "screening/평가 함수는 spec 을
문서 참조가 아니라 실제 호출부 코드에서 검증해야 한다"가 그대로 재발했다 — 이번엔
평가가 아니라 학습(reward 계산) 경로에서.

**해결책:**
1. `envs/step1_env.py::Step1Env.__init__` 에 `w1: float = W1, w2: float = W2,
   w3: float = W3` 파라미터 추가, `self.w1/self.w2/self.w3` 로 저장. `_calc_reward()`
   가 모듈 상수 `W1/W2/W3` 대신 `self.w1/self.w2/self.w3` 를 참조하도록 변경.
   기본값이 기존 모듈 상수와 동일해 override 하지 않으면 기존과 완전히 동일하게 동작
   (하위호환 보장).
2. `training/train_step1.py::make_env_fn()` 에 `w1=DEFAULT_W1, w2=DEFAULT_W2,
   w3=DEFAULT_W3` 파라미터 추가, `Step1Env(...)` 생성 시 전달. `run_training()` 의
   호출부에서 `args.w1/w2/w3` 를 실제로 넘기도록 수정.
3. `COLAB_GUIDE.md::make_train_fn_real().train_fn` 의 `make_env_fn(...)` 호출에
   `w1=w1, w2=w2, w3=w3` 추가.
4. **배선 검증 절차 신설** (아래 "재발 방지 절차" 참조) — 코드 수정과 별개로, 이번에
   놓쳤던 종류의 문제를 구조적으로 잡기 위한 절차.
5. 회귀 테스트 추가:
   - `tests/test_reward_baseline.py`: `test_default_w1_w2_w3_match_module_constants`
     (기본값이 모듈 상수와 정확히 같음 — 회귀 방지), `test_w1/w2/w3_override_changes_*`
     (동일 seed·동일 state 에서 w1/w2/w3 값만 바꾸면 reward 가 실제로 달라짐 — 배선이
     살아있음을 직접 확인).
   - `tests/test_train_step1_smoke.py`: `test_make_env_fn_default_w_matches_module_constants`,
     `test_make_env_fn_forwards_explicit_w_values` (`make_env_fn()` 을 거쳐 실제
     `Step1Env` 인스턴스까지 값이 도달하는지 확인).
   - 기존 `tests/test_reward_baseline.py` 의 ~15개 assertion(모두 override 없이 모듈
     상수 W1/W2/W3 를 직접 참조)이 수정 후에도 그대로 통과함을 확인 — 기본값 동작이
     바뀌지 않았다는 회귀 증거.

**실측 재현(Round 1 결과와의 비교) 관련 한계**: 이번 CLI 환경(Claude Code, GPU 없음)에서는
250K 학습 자체를 재현 실행할 수 없다. 대신 위 5번 테스트로 "기본값을 쓰면 `_calc_reward()`
출력이 수정 전후로 수학적으로 동일한 함수" 임을 코드 레벨에서 증명했다 — reward 계산이
순수 함수이고 그 출력이 바뀌지 않았다면, 같은 시드/같은 환경 동역학에서 학습 궤적의
확률분포도 수정 전후로 통계적으로 동일해야 한다(이미 알려진 대로 `MaskablePPO` 가
unseeded 라 실행마다 원래도 값이 정확히 일치하진 않는다 — §10 3-seed 체크 참조). 완전한
경험적 확인(= var008 을 수정된 코드로 재학습해 캐시된 0.9322 와 0.0135 노이즈 대역
안에서 일치하는지)은 Colab GPU 가 필요해 `COLAB_GUIDE.md` 에 §11 로 추가한 별도
"배선 수정 검증" 셀에서 사용자가 직접 1회 확인하도록 안내했다 (§11 참조, 비용 ~1.23 units).

**재발 방지 절차 (사용자 요청 — "spec 에 정의된 파라미터가 실제로 코드에 연결되어
있는지 확인하는 절차"):**

autoresearch sweep 대상 파라미터(현재: alpha, beta, w1, w2, w3)를 새로 추가하거나
변경할 때마다 다음을 의무화한다:

1. **"소비처까지 값이 도달하는가" 테스트를 반드시 함께 추가한다.** CLI 파싱/wandb
   기록/handoff 저장 테스트만으로는 불충분 — 반드시 "같은 seed·같은 state 에서 파라미터
   값만 바꾸면 최종 산출값(reward, 또는 해당 파라미터가 영향을 미치는 값)이 실제로
   달라진다"는 형태의 테스트를 짝으로 둔다 (이번 `test_w1_override_changes_*` 류).
   파라미터를 "읽기만 하고 기본값이 항상 쓰인다"는 조용한 버그는 값이 달라지는지
   보는 테스트가 아니면 절대 걸러지지 않는다.
2. **호출 체인을 코드에서 한 번에 grep 으로 추적**: 새 sweep 파라미터를 추가하면
   `CLI/params dict → make_*_fn() → 실제 소비 지점(_calc_reward 등)` 전체 체인이
   한 PR 안에서 문자열 검색만으로 끊김 없이 이어지는지 확인한다
   (`grep -rn "Step1Env(" ... | grep "w1\|w2\|w3"` 같은 방식으로 이번에 실제 확인함).
   "각 단계가 개별적으로 동작한다" 는 "체인 전체가 연결돼 있다" 를 보장하지 않는다.
3. **Round(autoresearch) 시작 전 스모크 체크리스트에 "sweep 파라미터 실제 반영 확인"
   항목을 추가한다**: 이미 Round 1 dry-run(5K) 단계가 있으므로, 그 dry-run 결과에서
   sweep 축을 하나 골라 극단값 2개의 산출 metric 이 실제로 달라지는지 확인하는 절차를
   Round 착수 전 체크리스트에 넣는다 (Round 2 는 아래 Phase 1 "효과 크기 사전 추정"이
   이 역할을 겸한다 — PROGRESS.md 참조).

**교훈:**
1. **"CLI 인자로 받아서 로그에 기록한다"는 "실제로 쓰인다"의 증거가 아니다.** 값이
   config/handoff/console 에 찍히는 것과 그 값이 실제 계산 경로에 들어가는 것은
   완전히 다른 두 가지 사실이며, 후자를 검증하는 테스트가 없으면 전자만으로는 아무것도
   보장되지 않는다.
2. **파라미터가 코드베이스에 들어온 시점이 다르면 배선이 끊기기 쉽다.** alpha/beta 는
   설계 초기부터 관통 경로가 있었고, w1/w2/w3 는 나중에(§16.3.2 Round 2 설계 시점)
   CLI 인자만 추가됐다 — "기존 파라미터가 되니까 새 파라미터도 됐겠지"는 검증 없이는
   성립하지 않는 가정이다.
3. **이 실패는 2026-08-30 entry(screening 평가 버그)와 같은 계열이다** — "spec/설정에
   정의돼 있다"와 "실제 호출부에서 쓰인다" 사이의 간극. 두 번째 발생이므로 앞으로
   sweep 대상 파라미터를 추가할 때마다 위 "재발 방지 절차"를 표준 체크리스트로 삼는다.

**관련 파일/위치:**
- `envs/step1_env.py::Step1Env.__init__`, `_calc_reward()`
- `training/train_step1.py::make_env_fn()`, `run_training()`
- `COLAB_GUIDE.md::make_train_fn_real().train_fn`, 신규 §11(배선 수정 검증 셀)
- `tests/test_reward_baseline.py`, `tests/test_train_step1_smoke.py` (신규 테스트)
- 관련: 2026-08-30 entry (screening 평가 버그) — 같은 유형("정의됨 ≠ 실제로 쓰임")의
  두 번째 발생

---

### 2026-09-01 — PR #5 의 "이전 버그 캐시 정리" 코드가 진짜 Stage 1 250K 결과를 삭제

**상황:**
PR #5(screening 평가 버그 수정) 머지 후 dry-run 검증까지 마치고, Round 1 Stage 1
(12 variants × 250K) 본 실행을 시작. 7시간 동안 실행해 12개 중 8~9개 variant 의
250K 결과가 `cache/round1/`에 저장된 상태였다. 다음날 세션을 재시작해 §8 복구
절차대로 셀 1 → 3 → (끊긴 셀부터, 관행적으로 셀 4부터) 재실행했는데, 셀 4(dry-run)를
다시 실행하자마자 `cache/round1/` 폴더가 새로 생성되며 기존 8~9개 결과가 전부
사라졌다.

**증상:**
- 셀 4 실행 직후 `cache/round1/`가 빈 상태로 재생성됨.
- 전날 7시간 학습한 8~9개 variant 의 `stage1_var*.json` 결과가 전부 소실.
- 애초에 모델 가중치(`model.save()`)를 저장하는 코드가 없어서, 설령 캐시가
  안전했더라도 학습된 정책 자체는 처음부터 복구 불가능한 상태였음 (별도 확인됨).
- 학습 로그에 `Using cpu device` 가 찍혀 있었음 — GPU 가 아니라 CPU 로 7시간 학습된
  것으로 추정(250K variant 1개당 실측 ~47~50분/variant, GPU 기준 예상치 ~17~18분/variant
  대비 약 3배 — CPU 학습 정황과 일치).

**원인 분석:**

[원인 1 (핵심). PR #5 에서 추가한 "일회성 정리" 코드가 무조건 실행되는 정상 흐름(셀 4)에
영구히 박혀 있었음]
PR #5 는 screening 평가 버그 수정 직후, 이전 버그로 생성된 잘못된 캐시(전부
success_rate=0.9000)를 한 번 지우기 위한 코드를 셀 4(dry-run) 맨 앞에 추가했다:

```python
CACHE_DIR = f"{PROJECT}/cache/round1"   # 진짜 Stage 1/2 결과가 쌓이는 바로 그 경로
...
for _p in (f"{CACHE_DIR}_dry", CACHE_DIR):   # CACHE_DIR(진짜 캐시) 이 루프에 포함됨
    shutil.rmtree(_p, ignore_errors=True)
```

이 코드는 "그 시점에 한 번만 실행하면 되는 마이그레이션 스텝"이었지만, 셀 4는 세션을
재시작할 때마다 정상적으로 다시 실행하게 되는 셀이다. 그 결과 **셀 4를 실행할 때마다
매번 `cache/round1/`(진짜 결과)를 통째로 지우는 코드**가 정상 실행 흐름에 영구히
포함된 셈이 됐다. 심지어 같은 `COLAB_GUIDE.md` §8 에는 "`cache/round1/` 은 절대
지우지 말 것"이라는 경고가 이미 존재했는데, PR #5 를 작성하며 그 경고와 대조 확인을
하지 않아 스스로 어겼다.

[원인 2. 모델 체크포인트 저장 로직 자체가 없었음]
`make_train_fn_real`의 `train_fn` 은 `model.learn()` 후 평가만 하고 `model.save()`를
호출하지 않았다. 캐시 JSON(metric 숫자)만 저장되고 학습된 가중치는 애초에 디스크에
남지 않으므로, 이번처럼 캐시가 삭제되지 않았더라도 "재평가"나 "체크포인트 로드"는
원천적으로 불가능한 상태였다.

[원인 3. GPU 미사용을 알아차릴 방법이 없었음]
셀 2("GPU 확인")가 있지만 §8 복구 절차는 "셀 1 → 3 → 끊긴 셀부터"만 명시해 셀 2가
매 세션 재실행 대상에서 빠지기 쉬웠다. 셀 4/5/6 자체에는 GPU 사용 여부를 확인하는
코드가 전혀 없어서, Colab 런타임이 어떤 이유로든(런타임 유형이 CPU로 잡히는 등) GPU
없이 배정돼도 아무 경고 없이 그대로 학습이 진행됐다.

**해결책:**
1. 셀 4의 캐시 정리 대상을 dry-run 전용 캐시(`cache/round1_smoke/`, 이름 변경 — 아래
   참조)와 dry-run 전용 Optuna DB(`autoresearch_dry.db`)로만 한정. `cache/round1/`
   (진짜 결과)와 `autoresearch_round1.db`는 셀 4에서 완전히 제외.
2. `cache/round1/`를 정말 지워야 하는 경우를 위해 정상 흐름(셀 0~8) 밖에 별도
   "부록 A. 위험 작업" 섹션을 신설. `CONFIRM = "DELETE ROUND1 CACHE"` 문자열을
   정확히 입력해야만 삭제되도록 게이트를 걸어, "Run all"로 전체 실행해도 안전.
3. 캐시 폴더 이름 정리 — 기존 `cache/round1_dry`(Colab dry-run) 와
   `cache/dryrun_round1`(별도 로컬/CI 스크립트 `scripts/dryrun_round1.py`) 이 서로
   혼동하기 쉬운 이름이었던 것도 함께 정리:
   - `cache/round1/` — 진짜 결과 (유지)
   - `cache/round1_smoke/` — Colab dry-run 전용 (구 `round1_dry`)
   - `cache/round1_cli_smoke/` — `scripts/dryrun_round1.py` 전용, Colab 과 무관 (구 `dryrun_round1`)
4. 셀 5 시작 시 `"12개 중 N개 캐시 재사용, M개 신규 학습"` 을 출력해 재개/신규 실행
   여부를 즉시 알 수 있게 함.
5. 셀 4/5/6 각각에 독립적으로 `torch.cuda.is_available()` 확인 코드를 추가하고,
   GPU 가 없으면 **기본값으로 즉시 예외를 발생시켜 중단**(`REQUIRE_GPU = True`,
   셀 상단에서 override 가능)하도록 함 — 경고만으로는 로그에 묻혀 놓치기 쉽다는
   점을 반영.
6. `train_fn` 에 `model.save(f"{cache_dir}/stage{stage_n}_var{variant_id:03d}_model")`
   추가. `make_train_fn_real(n_envs, cache_dir=...)` 가 `cache_dir` 를 호출부에서
   명시적으로 받도록 시그니처를 바꿔, dry-run/본 실행의 체크포인트가 서로 다른
   폴더에 저장되어 절대 뒤섞이지 않도록 함(전역 변수 암묵 참조 방식은 이번 사고와
   같은 클래스의 실수를 다시 낼 수 있어 피함). 비용 실측: variant 당 ~0.87MB,
   ~22ms — Round 1 전체(18개) 합쳐도 ~16MB, ~0.4초로 무시 가능한 수준.
7. 위 1~3, 6 번의 안전장치는 `exec()` 로 실제 코드를 돌려 검증함 — (a) GPU 미탑재
   상태에서 하드 스톱이 실제로 발동하는지, (b) 정리 코드가 실제 `cache/round1/`
   내용을 보존하면서 smoke 캐시만 지우는지, (c) dry-run/본 실행 체크포인트 경로가
   실제로 서로 겹치지 않는지, (d) 부록 A 셀이 기본값(빈 `CONFIRM`)에서는 절대
   삭제하지 않고 정확한 문자열 입력 시에만 삭제하는지.

**손실 확인 — 복구 불가능**: 사라진 8~9개 variant 의 250K 결과(및 애초에 저장된 적
없는 모델 가중치)는 이번 수정으로도 복구되지 않는다. Stage 1 은 처음부터 다시
실행해야 한다. 이번 수정 이후로는 (a) 진짜 캐시가 자동 삭제되지 않고 (b) 체크포인트가
저장되므로 같은 손실이 재발하지 않는다.

**교훈:**
1. **"한 번만 실행하면 되는" 코드를 정상 실행 흐름(재실행되는 셀)에 넣지 말 것.**
   일회성 마이그레이션/정리 작업은 반드시 별도의, 명시적으로 트리거해야 하는
   위치에 둔다 — 정상 흐름에 섞이면 "한 번"이 "매번"이 된다.
2. **파괴적 코드를 추가할 때는 같은 문서 안의 기존 경고문과 반드시 대조 검증할 것.**
   이번 사고는 §8 에 이미 적혀 있던 "cache/round1/ 은 절대 지우지 말 것"이라는
   경고를 정면으로 어겼는데도 PR #5 리뷰에서 걸러지지 않았다.
3. **파괴적 연산은 기본값이 안전하도록(no-op) 설계할 것.** 이번에 적용한 패턴
   (명시적 확인 문자열 입력, `REQUIRE_GPU` 같은 opt-out 방식의 하드 스톱)을 앞으로
   Colab 가이드에 위험한 셀을 추가할 때 표준으로 삼는다.
4. **가중치/산출물 저장 로직은 "평가 로직을 다 만든 뒤 나중에"가 아니라 학습 파이프라인을
   처음 만들 때부터 넣는다.** 캐시가 안전했어도 체크포인트가 없었다면 이번 손실은
   똑같이 복구 불가능했다.
5. **런타임 환경(GPU 배정 등)을 매 실행마다 다시 확인할 것.** 세션이 바뀌면
   이전 세션에서 확인했던 전제(GPU 사용 가능)가 더 이상 유효하지 않을 수 있다 —
   중요한 셀은 스스로 전제를 재검증해야 한다.

**관련 파일/위치:**
- `COLAB_GUIDE.md` 셀 4/5/6, 신규 "부록 A. 위험 작업" 섹션, §8
- `scripts/dryrun_round1.py` (`CACHE_DIR` 이름 변경)
- 관련: 2026-08-30 entry (screening 평가 버그) — 같은 세션에서 그 수정 직후 발생한 사고

---

### 2026-08-30 — Round 1 screening 이 spec 을 안 따르고, 그마저도 20 episode 전부 동일 시나리오였음

**상황:**
Round 1 Stage 1(12 variants × 250K)이 완료됐으나 12개 variant 의 metric 이 전부
`0.9000` 으로 동일해 생존자 6개가 성적이 아니라 variant 번호 순으로 잘리는
현상을 사용자가 발견, 원인 분석 요청.

**증상:**
- Stage 1 12개 variant, 심지어 5K dry-run 까지 success_rate 가 정확히 0.9000 으로 일치.
- 생존자 6개가 α ∈ {0.5, 1.0} (variant 앞 6개)로, 정렬 순서에 의한 결과로 의심됨.
- wandb 에 `eval/success_rate` 가 학습 종료 후 딱 1개 점만 기록되어 학습 곡선 확인 불가.

**원인 분석:**

[원인 1. Screening 이 §11.0 spec(75개 고정, Easy:Medium:Hard=3:5:2)을 전혀 따르지 않음]
`training/train_step1.py::_evaluate_success_rate()` 를 `COLAB_GUIDE.md` 의
`make_train_fn_real` 이 Stage 1/2 screening 평가로 그대로 사용 중이었는데, 이 함수는
`mode="final_eval"` 하드코딩 + `eval_seed=130000`(§11.0.4 **Final Eval** 범위 시작값)
+ `n_episodes=20` + `difficulty="medium"` 고정이었다. §11.0 의 screening 전용
75개 seed(`STEP1_SCREENING_SEEDS`, 110000~110074)는 정의만 되어 있고 어디서도
참조되지 않았다.

[원인 2 (핵심). Step1Env 생성자 seed 와 reset(seed=) 의 역할 혼동으로 20개 "episode" 가
전부 동일 시나리오였음]
`Step1Env` 는 생성자의 `seed=` 와 `reset(seed=N)` 의 `seed=` 가 서로 다른 일을 한다:
- 생성자 `seed=`: `_episode_rng` 초기화(→ start_dir_idx/goal_dir_idx 추첨)에만 쓰임
- `reset(seed=N)`: 시나리오(occupancy/start/goal cell)를 결정하는 `_forced_scenario_seed` 설정

`_evaluate_success_rate` 는 매 episode `Step1Env(seed=eval_seed+i, ...)` 로 **새 인스턴스**를
만들고 `env.reset()` 을 **seed 없이** 호출했다. `_forced_scenario_seed=None` 이므로
`_sample_scenario()` 는 `_next_seed()` 로 폴백하는데, 매번 새 인스턴스라 pool 카운터가
항상 0부터 시작 → `STEP1_FINAL_EVAL_SEEDS[0]=130000` 이 **매번** 뽑혔다. `eval_seed+i`
는 생성자에만 들어가고 시나리오 결정에는 전혀 영향을 주지 않는 죽은 코드였다.

직접 재현 결과, 20개 seed 전부 occupancy 해시/start=[2,11,18]/goal=[10,5,15] 동일.
그 고정된 미로에서 6방향 중 정확히 1방향(+Z)만 시작점 옆이 막혀 있었고, 첫 스텝은
`action_masks()` Rule 1(§12.4 Hard Constraint)이 `start_dir_idx` 방향 하나만 강제하며
**그 방향이 막혀있는지 검증하지 않는다**. `start_dir_idx` 는 생성자 seed 로 결정되는
`_episode_rng` 추첨값이라 20개 episode 마다 달랐고, 그중 정확히 2개(seed 130000,
130017)가 막힌 방향(+Z, index 4)을 뽑아 정책과 무관하게 1스텝째 즉시 collision 종료됐다.
나머지 18개는 같은 미로를 다른 방향에서 볼 뿐이라 정책 품질이 결과에 거의 반영되지
않았다 → 12개 variant + dry-run 전부 18/20=0.9000 으로 정확히 일치한 이유가 100% 설명됨.

(참고: `training/regression_callback.py::evaluate_step_policy` 는 인스턴스 하나를
재사용하며 반복 `reset()`(seed 없이) 하므로 pool 이 정상 순환해 이 버그의 영향을
받지 않는다. 버그는 "매 episode 새 인스턴스를 만들고 생성자 seed 만 주는" 호출부
패턴에서만 발생한다.)

[부수 확인 — 학습 자체는 정상] `mode="train"` 은 `DummyVecEnv` 안에서 인스턴스가
유지된 채 반복 `reset()`(seed 없이) 되므로 `_train_seed_counter` 가 150000부터
정상 순환한다. 즉 250K 학습 자체는 실제로 진행됐으나, 평가가 고장나 그 차이가
전혀 드러나지 않았을 뿐이다. wandb 1-point 문제는 버그가 아니라 현재 설계(학습
종료 후 1회만 평가/로깅, 주기적 EvalCallback 없음)이며 이번 수정 범위에서는
의도적으로 제외했다 (아래 "다루지 않은 것" 참조).

**해결책:**
1. `_evaluate_success_rate`: `Step1Env(seed=...)` 생성자 seed 제거, `env.reset(seed=eval_seed+i)` 로 명시.
2. 신규 `_evaluate_screening(model, alpha, beta)` 추가 — §11.0 spec 대로 75개 고정
   (easy 23/medium 37/hard 15, 연속 블록 분할: `STEP1_SCREENING_SEEDS_EASY/MEDIUM/HARD`)
   을 `reset(seed=X)` 로 순회. 매 episode 마다 반드시 `reset(seed=)` 를 명시적으로 전달.
3. §11.2 length_ratio(L_rl/L_astar) 추가. A* 대신 6-connected unit-cost BFS 사용(격자가
   unit-cost 라 A*(cost=length) 최적해와 동일) — `envs/scenario_generator.py::bfs_shortest_path_length()`.
   실측 비용: 75개 시나리오 BFS 전체 1.24초 (개당 16.6ms) — 시나리오당 1회 계산이라 무시 가능.
   §12.4 강제 첫 스텝(start_dir_idx)을 반영해 "start_dir_idx 방향 이웃 cell" 에서
   BFS 시작 + 1 로 계산해 공정성을 맞췄다.
4. success_rate 와 length_ratio 를 하나의 순위로 결합: lexicographic (success_rate
   1차, length_ratio 2차 tie-break) 를 단일 float `score` 로 인코딩
   (`EPS * CAP < 1/75` 불변식으로 length_ratio 항이 success_rate 순위를 절대 못 뒤집게 보장).
5. `COLAB_GUIDE.md` 의 `make_train_fn_real` 이 `_evaluate_screening` 을 쓰도록 갱신 +
   "이 수정 이후 최초 재실행 시 이전 버그로 생긴 캐시/DB 삭제 필수" 안내 추가
   (그렇지 않으면 예전의 잘못된 0.9000 결과가 캐시 히트로 재사용됨).
6. 회귀 테스트 추가: `tests/test_env_generator_integration.py`
   (`reset(seed=)` 시나리오 다양성 확인 + 이 함정이 여전히 존재함을 문서화하는
   `test_constructor_seed_alone_does_not_pin_scenario`, BFS 정확성),
   `tests/test_screening_eval.py` (`_evaluate_screening` 난이도 gradient, score 불변식).

**다루지 않은 것 (의도적 범위 제외):**
- wandb 학습 곡선(주기적 eval callback) — 진단 결과 학습 자체는 정상 진행된 것으로
  확인되어 급하지 않음. 한 번에 두 가지를 바꾸면 다음 재실행 결과가 이상할 때
  원인 분리가 어려워지므로 별도로 미룸.
- variant 별 학습 seed 오프셋(`150000 + variant_id*100`) 무력화 — 같은 생성자
  seed 무시 패턴이 `make_train_fn_real` 의 학습용 `DummyVecEnv` 구성에도 있지만,
  모든 variant 가 동일한 학습 시나리오 순서를 쓰게 되는 효과라 공정 비교 관점에서는
  오히려 중립적이라 판단, 이번 수정에서 건드리지 않음. 사실관계로 여기 기록만 남김.
- `_get_action_mask()` Rule 1(첫 스텝 강제)이 그 방향이 막혀있는지 검증하지 않는 문제
  자체는 이번에 고치지 않았다. 이번 fix 는 "screening 이 항상 같은 미로/구도만 봤다"는
  문제를 없앴을 뿐, 새로 다양해진 75개 시나리오에서도 "무작위로 뽑힌 start_dir_idx 가
  하필 막힌 방향" 케이스는 여전히 시나리오마다 일정 확률(대략 1/6 안팎, 장애물 밀도에
  따라 다름)로 발생해 정책과 무관하게 실패로 잡힐 수 있다. Action mask 코어 로직을
  건드리는 더 큰 변경이라 별도 세션에서 논의 필요 — screening/regression/final_eval
  전반의 잠재적 노이즈 floor 로 남겨둠.

**교훈:**
1. **생성자 `seed=` 와 `reset(seed=)` 는 다른 일을 한다 — 시나리오를 고정하려면 항상
   `reset(seed=)` 를 명시하라.** 이 프로젝트의 `Step1Env`/`BaseEnv` API 고유의 함정이며,
   "매 episode 새 인스턴스를 만드는" 흔한 평가 루프 패턴과 결합하면 조용히(예외 없이)
   시나리오가 고정되어 버린다 — 에러가 안 나서 발견이 늦어짐.
2. **평가 metric 이 정확히 동일한 값으로 여러 variant 에 걸쳐 tie 나면 "우연"이 아니라
   "평가 자체가 고장났다"는 신호로 먼저 의심하라.** 이번 케이스처럼 우연한 tie 로
   치부하고 넘어갔다면 Round 1 전체가 무의미한 순위로 진행될 뻔했다.
3. **screening/평가 함수는 §11.0 spec(고정 seed pool, 난이도 분할)을 문서 참조가
   아니라 실제 호출부 코드에서 검증**해야 한다 — spec 상수(`STEP1_SCREENING_SEEDS`)가
   정의돼 있다고 해서 실제로 쓰이고 있다는 보장은 없다.
4. **버그 수정 후 재실행 시 이전(버그가 낳은) 캐시를 반드시 무효화**해야 한다 —
   `StageRunner`/Optuna 캐시는 "이미 계산된 결과"를 정확도와 무관하게 재사용하므로,
   평가 로직을 고쳐도 캐시를 안 지우면 예전의 잘못된 결과가 그대로 재사용된다.

**관련 파일/위치:**
- `training/train_step1.py::_evaluate_success_rate`, `_evaluate_screening` (신규)
- `envs/scenario_generator.py::bfs_shortest_path_length` (신규), `STEP1_SCREENING_SEEDS_EASY/MEDIUM/HARD` (신규)
- `envs/base_env.py::reset()`, `sample_start_goal_dirs()`, `Step1Env._get_action_mask()` Rule 1 (관련이지만 미수정)
- `COLAB_GUIDE.md` 셀 4/5/6 (`make_train_fn_real`)
- `docs/evaluation-spec.md` §11.0, §11.2
- `tests/test_env_generator_integration.py`, `tests/test_screening_eval.py` (신규)

---

### 2026-06-28 — CLAUDE.md §16.3.2 sweep 산술 오류 (4×3×3=27 → 실제 36)

**상황:**
Sub-단계 5.2.a (autoresearch 인프라 구현 — Optuna + wandb + Stage runner) 중,
`autoresearch/optuna_study.py` 의 `enqueue_round2_grid()` 테스트 작성 과정에서 발견.
Round 2 sweep 범위 표기 검증 중 spec 과 실제 산술의 불일치 확인.

**증상:**
CLAUDE.md §16.3.2 (L-16.3-w 해제 결과, 의사결정 19) 의 sweep 범위:
- 표기: `4 × 3 × 3 = 27 variants`
- 실제: 4 × 3 × 3 = **36** variants (산술 오류)

동일 오류가 여러 파일에 전파됨 (§16.3.2 외에 §16.5, §16.6.7, §99 Lock 해제 로그, SKILL.md §3.9, §4.9, PROGRESS.md 3개 위치, autoresearch/optuna_study.py 로그 메시지).

**원인 분석:**

[원인 1. 의사결정 작성 시 산술 미검증]
의사결정 19 (L-16.3-w 해제, 2026-06-18, Session 2026-06-18) 작성 시
Claude AI 세션에서 `4 × 3 × 3 = 27` 로 직접 계산 오류 기록.
산술 표기가 있을 때 별도 검증 단계가 없었음.

[원인 2. 사용자 confirm 의 한계]
사용자 confirm 은 큰 결정 방향에 대한 동의. 모든 세부 산술 검증 아님.
`4 × 3 × 3 = ?` 를 사용자가 암산 검증했을 것이라는 암묵적 가정이 있었음.

[원인 3. \_ing 졸업 시점 (2026-06-24) 검증 미포함]
졸업 체크리스트 (의사결정 21/24) 에 "기존 spec 의 산술 표기 일관성 검증" 항목 없었음.

[원인 4. 오류가 여러 파일에 복사 전파]
의사결정 19 결론 → CLAUDE.md §16.3.2 → SKILL.md §3.9/§4.9 → PROGRESS.md 여러 위치 → autoresearch 코드 주석까지 동일 오류 전파. 단일 오류 포인트에서 다중 파일 오염.

[원인 5. implementation 단계 이전에는 발견 경로 없었음]
spec 단계에서는 "27 개를 학습한다" 는 계획 수준이라 오류가 가시적이지 않음.
코드에서 실제 루프를 돌리는 순간 (`4×3×3 = 36` 루프 결과) 에 비로소 발견됨.

**해결책:**

1. CLAUDE.md §16.3.2: `27 variants` → `36 variants` (수정 완료)
2. CLAUDE.md §16.5 (16.5.C 변형 제안 메커니즘): `12~27` → `12~36` (수정 완료)
3. CLAUDE.md §16.6.7 Round 2: `27 variants`, Stage 1 `하위 14 제거`, Stage 2 `13` → `36 variants`, `하위 18 제거`, `18` (수정 완료)
4. CLAUDE.md §99 L-16.3-w 해제 로그: `27 variants` → `36 variants` (수정 완료)
5. SKILL.md §3.9 Round 2: `→ 27 variants` → `→ 36 variants` + 의사결정 27 footnote (수정 완료)
6. SKILL.md §4.9 Round 2: `27 variants`, Stage 1/2 수 갱신 (수정 완료)
7. PROGRESS.md 3개 위치: 모두 `36` 으로 수정 완료
8. autoresearch/optuna_study.py 로그 메시지 + docstring 주석 갱신 (수정 완료)
9. tests/test_optuna_study.py 독스트링 갱신 (수정 완료)

wall-clock 시간 재산정:
  Stage 1: 250K × 36 = 9M timestep (T4 동시 3개 기준 약 12.5 시간)
  구 산정: 250K × 27 = 6.75M timestep (약 9.4 시간)

**교훈 (5가지):**

1. **산술은 spec 작성 단계에서 별도 검증 필요**:
   `4 × 3 × 3` 같은 단순 산술도 작성자가 잘못 계산 가능.
   spec 안에 산술 표기가 있으면 작성 시 **즉시 계산기 또는 Python 으로 검증** 의무.
   향후 spec 변경 시 산술 표기는 별도 확인 단계 거침.

2. **두 세션 (Claude AI 세션 + Claude Code CLI) 역할 분리가 의도대로 작동**:
   Claude AI 세션 (spec 결정) 에서 산술 오류 → Claude Code CLI (implementation) 가 발견.
   implementation 단계가 spec 의 품질 보증 역할을 수행 — 본 \_ing 시스템 설계의 정상 작동.

3. **사용자 confirm 의 한계를 가정하지 않음**:
   confirm 은 "방향 동의" 이지 "세부 검증" 아님.
   spec 작성자 (Claude) 의 self-check 가 사용자 검증보다 먼저여야 함.
   "사용자가 확인했으니 맞겠지" 는 논리적 오류.

4. **implementation 단계가 spec 의 가장 강력한 검증 도구**:
   추상적 계획 수준에서는 `27` 이나 `36` 이나 똑같아 보임.
   코드에서 실제 루프를 돌리는 순간 숫자가 강제 검증됨.
   테스트 코드 작성 = spec 검증의 또 다른 이름.

5. **단순 오류도 archive 가치 있음**:
   `4 × 3 × 3 = 27` 같은 "단순한 오류" 일수록 향후 반복될 위험.
   패턴: 행렬 크기 계산, 조합 수 계산, 비율 계산 등 spec 내 산술 전반에 동일 위험.
   archive 화로 향후 spec 작성 시 산술 표기에 주의 기울이는 습관 형성.

**관련 파일/위치:**
- CLAUDE.md §16.3.2 (수정 완료) — sweep 범위 27 → 36
- CLAUDE.md §16.5, §16.6.7, §99 (수정 완료) — 연관 위치
- SKILL.md §3.9, §4.9 (수정 완료)
- PROGRESS.md (수정 완료 — 3개 위치)
- autoresearch/optuna_study.py (수정 완료)
- tests/test_optuna_study.py (수정 완료)

**관련 의사결정:**
- 의사결정 19 (2026-06-18) — L-16.3-w 해제 — 산술 오류 포함 원본
- 의사결정 27 신규 (2026-06-28) — 산술 오류 수정 + archive

---

### 2026-06-25 — CLI 자동 로드 skill 과 졸업판 spec 충돌 (120-dim vs 150-dim / 27-action vs 7-direction)

**상황:**
harness engineering 첫 작업 (PBS helper class 구현, `PotentialBasedShaping`) 진행 중
CLI 가 자동 로드한 서버 등록 skill 의 spec 이 졸업판 SKILL.md 와 상이함을 발견.

**증상:**
- CLI 자동 로드 skill (서버 등록 — v1 시절 등록): obs 120-dim, action 27-action
- 졸업판 SKILL.md (로컬 파일): obs 150-dim, action 7-direction
- 두 spec 이 정면 충돌 — 어느 것이 "진짜 spec" 인지 불분명한 상태에서 코드 작성 불가

**원인 분석 (진단 3단계 진화):**

[1차 추측 — 오진단]
CLI 가 SKILL.md 를 읽지 못하거나, 서버 skill 이 로컬 파일을 덮어쓰는 것으로 추측.
→ 오진단: 실제로는 두 출처가 독립적으로 동시 존재.

[2차 부분 정정]
서버 skill 과 로컬 SKILL.md 가 별도 경로임을 인식. 서버 skill 이 v1 시절 등록된 구버전일 가능성 제기.
→ 방향은 맞으나 "서버 skill 이 자동 갱신됐을 것" 이라는 가정이 남음.

[3차 디스크 증거 기반 확정]
PowerShell `dir` + 파일 직접 확인 → SKILL.md 내용 (150-dim / 7-direction) 명시.
skill list 에서 서버 등록 skill description → 120-dim / 27-action 확인.
두 출처가 **동시에 독립적으로 존재**하며 CLI 는 둘 다 자동 로드.

[확정 원인]
사용자가 v1 시절 등록한 custom skill (120-dim / 27-action) 이 서버에 영구 보존됨.
졸업 작업 (2026-06-24 의사결정 21) 에서 서버측 skill 갱신을 체크리스트에 포함하지 않은 것이 근본 원인.

**해결책:**
Option A-1 (in-place 갱신) 채택 (의사결정 23):
- 서버 등록 skill 을 졸업판 SKILL.md 내용으로 in-place 업데이트
- SKILL.md frontmatter `name` 필드 = 서버 등록 skill 이름과 정합화
- 이후 SKILL 갱신 순서: 로컬 수정 → git commit → 서버 in-place 갱신 → CLI 검증 (의무)

**교훈 (5가지):**

1. **'졸업' 의 범위 재정의 필요**:
   본 \_ing 시스템의 졸업 (의사결정 21) 은 로컬 파일 rename + 헤더 갱신만 포함.
   **서버 등록 skill 갱신은 졸업 체크리스트에 없었음**. 의사결정 24 로 체크리스트에 추가.

2. **CLI Claude 의 자기 정정 능력 신뢰 가능 + 디스크 증거 우선**:
   진단 3단계에서 CLI 가 자체 가정을 수정하는 과정이 정상 작동.
   "로컬 파일 vs 서버 등록" 구별은 디스크 직접 확인 없이는 가정에 기반할 수밖에 없음.
   **디스크 증거가 항상 가정보다 우선**.

3. **PowerShell 오류 의미 구별 필수 (캐시 비워짐 vs 파일 미존재)**:
   PowerShell `Test-Path` / `Get-Content` 오류 시 "파일 미존재" 인지 "캐시 비워짐/권한 오류" 인지 즉시 구별.
   오류 메시지 문구 (not found vs access denied) 확인이 필수.

4. **skill 시스템 권위 모호 시 한 쪽을 source 로 명시**:
   서버 등록 skill 과 로컬 SKILL.md 충돌 시 어느 쪽이 권위인지 명확히 해야 함.
   본 프로젝트 결론: **로컬 SKILL.md 가 source of truth**. 서버 skill 은 mirror.

5. **본 \_ing 시스템의 다음 진화 자료**:
   harness engineering 첫 세션에서 발생한 메타 충돌 (spec 관리 인프라 자체의 결함).
   \_ing 시스템의 "졸업" 개념을 서버 인프라까지 확장하는 계기 — 의사결정 24 의 직접 trigger.

**관련 파일/위치:**
- SKILL.md (로컬, source of truth) — 150-dim / 7-direction 정의
- 서버 등록 skill — 갱신 전 120-dim / 27-action, 갱신 후 SKILL.md 와 동일
- SKILL.md §0.5 (신설) — 서버 등록 skill 동기 의무
- CLAUDE.md §101 (갱신) — 졸업 후 운영 원칙 + 의사결정 22/23/24 추가

**관련 의사결정:**
- 의사결정 22 (2026-06-25) — PBS state-only 검증 강화
- 의사결정 23 (2026-06-25) — skill 시스템 충돌 해결 (Option A-1)
- 의사결정 24 (2026-06-25) — 졸업 범위 확장 (서버 skill 동기화 의무 추가)

---

### 2026-06-11 — spec 자체의 회귀 검증 체계 미완성 + Lock 명명 규칙 암묵화

**상황:**
사용자가 두 질문을 제기. (1) "L-D3 의 D 가 무슨 의미?" — Lock 명명 체계가 spec 어디에도 명시 안 됨. (2) "전이학습/curriculum 의 가중치 보존을 어떻게 확신? 측정 지표는? 학습 얼마나 했을 때 확인?" — 회귀 검증 체계가 임계값/지표/주기 측면에서 미완성. 본 entry 는 학습 실패가 아니라 **harness engineering 시작 전 spec 자체의 결함 발견** 사례.

**증상:**

[증상 1. Lock 명명 체계 암묵화]
- L-A / L-B / L-C / L-D 의 접두사 의미를 spec 본문에서 직접 설명한 곳 없음
- 항목 내용과 시간순 추이로 역추론만 가능
- 6개월 후 자기 자신 또는 새 협업자가 spec 만 읽고는 명명 체계 이해 불가

[증상 2. 회귀 검증 체계의 빈 곳]
CLAUDE.md §12.3 + §13.1 이 회귀 검증의 인프라는 갖추었으나, 다음 세 가지가 모든 step 에서 미정:
- **회귀 지표**: "성능 떨어지면 회귀" 라고만 적혀 있고 어떤 metric / 어떻게 계산 안 정의
- **측정 주기**: 학습 종료 후 1회? 학습 도중? 주기는 얼마?
- **트리거 임계값과 대응**: 어느 수준 이하에서 무엇을 하는지

§12.2 의 후보 A/B/C/D (전이학습 전략) 가 모두 "forgetting 방지" 가 명분인데, 정작 forgetting 을 어떻게 측정할지 spec 에 없음. **수단을 나열하면서 측정 도구를 빠뜨림.**

**원인 분석:**

[원인 1. 메타 문서화 누락]
Lock 명명 체계는 2026-04-30 의사결정 1~7 작성 시점부터 자연스럽게 사용했으나, "왜 이 알파벳을 골랐는지" 를 명시한 적 없음. 2026-05-14 에 L-B 가 전면 폐기되고 L-D 가 신설되며 알파벳이 비연속 (B 다음 D) 되었는데, 이 시점에서도 범례 부재를 알아채지 못함.

[원인 2. 평가 인프라와 측정 체계의 분리 실패]
의사결정 8 (Layer 1/2 평가 분리, 2026-05-14) 작성 시 회귀 검증의 임계값 / 주기 / 지표를 같이 정의 안 함. "회귀 시 학습 중단/조정" 만 적어두고 측정 체계는 통째로 비워둠. **"인프라가 있다 = 측정이 정의됐다" 로 착각.**

[원인 3. 가중치 보존을 "확신" 영역에 둠]
신경망 순차 학습에서 catastrophic forgetting 은 디폴트 결과인데, 본 spec 은 무의식적으로 "전이학습 = 가중치 살아남음" 으로 전제. 사용자가 "어떻게 확신하지?" 라고 물어 비로소 이 전제가 검증 영역으로 이동.

**해결책:**

[2026-06-11 의사결정 12, 13 으로 반영됨 — PROGRESS.md Session 2026-06-11 참조]

1. **CLAUDE.md §99 맨 앞에 Lock 명명 규칙 범례 표 내장** (의사결정 12)
   - L-A/B/C/D + 섹션 직참조 형식의 의미를 표로 명시
   - 폐기된 카테고리 ID 재할당 금지 명시
   - Single source of truth 는 CLAUDE.md §99 한 곳

2. **회귀 검증 체계 명시화** (의사결정 13)
   - 지표 3종 의무: 각 과거 step metric / BWT (Backward Transfer) / Forgetting measure
   - 측정 주기: wandb logging 주기와 정합, 매 K timestep
   - 트리거: 임계값 이하 → 학습 중단 → (mixed curriculum 상향 → 학습률 감소 → 체크포인트 복귀)

3. **SKILL §3.7 회귀 감시 의무 신설**

4. **L-D1 의 범위 확장**: "전이학습 전략 선택" 만이 아니라 "회귀 감시 임계값 / 주기 / 트리거를 동시 결정" 으로 의미 확장

5. **SKILL §0 차단 표 확장**: "Step 2 이상 학습 시작 → L-D1 미해제로 차단" 추가

**교훈:**

1. **spec 의 자기 설명성 (self-documenting) 검증 필요**:
   spec 작성자가 자연스럽게 쓰는 명명 체계 / 약어 / 카테고리는 시간이 지나면 의미가 잊힌다. spec 작성 후 6개월 후 자기 자신이 spec 만 읽고도 이해 가능한지 셀프 점검 필요.

2. **인프라 ≠ 측정 체계**:
   회귀 검증 파일 (`stepN_regression_report.json`) 을 핸드오프 파일에 명시한 것만으로 측정 체계가 완성됐다고 착각함. **"무엇을 측정" + "어떻게 측정" + "언제 측정" + "무엇이 임계점" 을 모두 spec 화해야 측정 체계 완성.**

3. **자명해 보이는 전제는 의심**:
   "전이학습은 가중치를 보존한다" 같은 전제는 실은 보장되지 않음. 자명해 보일수록 의심하고 측정 체계로 끌어내려야 함. Catastrophic forgetting 은 신경망 순차 학습의 디폴트 결과이며 보존이 예외 (수단으로 강제해야 발생).

4. **"수단 나열" 과 "측정 도구" 의 분리 검증**:
   §12.2 후보 A/B/C/D 처럼 수단을 나열했을 때 "이 수단들이 실제 효과를 내는지 어떻게 측정하는가?" 를 동시에 명시 안 하면, 수단만 있고 검증이 없는 spec 이 된다.

5. **\_ing 시스템의 정상 작동 (반복 확인)**:
   본 entry 역시 \_ing 시스템 (Lock 체계 + harness engineering 차단) 의 정상 작동 사례. 학습 시작 전에 spec 의 두 결함이 사용자 질문으로 발견되어, 큰 손실 없이 spec 수정 가능. 2026-05-14 entry 의 마지막 교훈 5번 ("\_ing 시스템 자체의 가치 검증") 이 한 번 더 입증됨.

**관련 파일/위치:**
- CLAUDE.md §99 (Lock 명명 규칙 범례 신설)
- SKILL.md §3.7 (회귀 감시 의무 신설), §6.4 (절대 금지 항목 추가), §6.5 (Lock 명명 규칙 참조 신설), §0 차단 표 확장
- PROGRESS.md Session 2026-06-11 (의사결정 12, 13)
- (영향) CLAUDE.md §12.2 후보 A/B/C/D — L-D1 해제 시 회귀 감시 체계 동시 결정 의무
- (영향) CLAUDE.md §12.3 + §13.1 (회귀 검증 인프라) — 측정 체계 의무 추가됨

**관련 의사결정:**
- 의사결정 12 (2026-06-11) — Lock 명명 규칙 spec 내장
- 의사결정 13 (2026-06-11) — 회귀 검증 체계 명시화 + L-D1 범위 확장
- 의사결정 8 (2026-05-14) — Layer 1/2 평가 분리 — 본 entry 의 원인 2 가 가리키는 미완성 결정 (보강됨)

---

### 2026-05-14 — Phase 1 비효율 경로 + Hierarchical 가설 오진단

**상황:**
기존 pipe-routing-r1 프로젝트 Phase 1 학습 결과가 만족 수준에 못 미침. 2026-04-30 세션에서 이 문제 해결 방향으로 Hierarchical (Macro+Micro 별도 네트워크) 구조를 채택 (의사결정 3). 그러나 후속 세션 (2026-05-14) 에서 본 결정의 근거가 오진단에 기반함을 확인.

**증상:**

Phase 1 학습 결과의 정성적 양상:
- 학습은 수렴함 (갇힘/발산 아님)
- Goal 도달은 성공
- **다만 경로 자체가 비효율** — 숙련 엔지니어 기준에 못 미치는 우회/지그재그/비최적 선택

Autoresearch 결과:
- 전 분야 (reward 가중치, hyperparameter, 부가 구조) 변형을 통한 점진적 개선은 있었음
- 그러나 만족 수준에 도달 못 함
- 천장 (plateau) 인지 search 부족인지 진단 불가

**원인 분석:**

[Layer 1. RL 학습 자체의 문제]
2026-05-14 세션에서 사용자에게 Phase 1 실패 양상을 직접 확인한 결과:

> "LOCAL VIEW단계에서 잘되었다면 GLOBAL VIEW도 잘 되었을것으로 여겨짐. LOCAL VIEW에서 부터 만족되지 않음."

이는 단순히 "global view 부재" 가 아니라 **"local 실행 능력 자체가 부족"** 임을 의미. 가능한 진짜 원인:
- Reward 신호가 sparse 하여 학습 신호 부족
- Observation 정보량 부족 (SDF + Raycast 가 충분하지 않을 가능성)
- Action space (7-direction discrete) 가 너무 미시적
- Autoresearch가 만질 수 없는 영역 (절대 변경 금지 항목) 에 천장이 있을 가능성

[Layer 2. 의사결정 3 자체의 오진단]
2026-04-30 의사결정 3에서 Claude가 진단한 원인:
> "Phase 1 실패 원인 = local 결정만 하는 구조 (global view 부재)"

이는 **"local은 잘 되는데 global이 부재"** 라는 가정을 깔고 있음. 그러나 사용자 답변에서 그 가정 자체가 사실과 다름이 확인됨. Hierarchical RL의 표준 가정 ("low-level이 reasonably well 학습된 후 high-level이 조립") 이 깨진 상태이므로 Macro+Micro 도입은 잘못된 처방.

**해결책:**

[2026-05-14 의사결정으로 반영됨]

1. **의사결정 3-r1**: Hierarchical (Macro+Micro 별도 네트워크) 폐기. 단일 에이전트 + 전이학습 구조로 전환.

2. **의사결정 10**: Dense Reward (Potential-Based Shaping) 도입.
   - HRLP (Tan & Mu 2024) 영감
   - "Local 부실" 문제의 직접 처방
   - 별도 네트워크 없이 단일 에이전트 안에서 dense reward 효과 흡수
   - PBS 3대 조건 (state-only / Φ(terminal)=0 / γΦ(s')-Φ(s)) 으로 정책 보존 이론적 보장

3. **의사결정 9**: wandb 도입.
   - autoresearch 천장 진단 도구
   - Parallel Coordinates Plot 으로 변형 간 패턴 시각화
   - Phase 1 재학습 시 진단 가능성 확보

4. **의사결정 11**: Macro-action 도입 검토 (Phase 1 후 결정).
   - HRLP의 options 정신만 차용 (별도 네트워크 없이)
   - Phase 1 결과 (dense reward 단독 도입 후) 에 따라 도입 결정

**교훈:**

1. **실패 양상의 정성적 구체화 없이 진단 금지**:
   "비효율 경로" 같은 추상적 표현으로 멈추지 말고, "어떤 종류의 비효율인가 (우회/지그재그/장애물 거리/직선 회피/bend 과다)" 를 구체화한 후 진단해야 함.

2. **Hierarchical RL 도입 전 Low-level 능력 검증 필수**:
   "low-level이 reasonably well 학습된 후" 가 Hierarchical의 표준 가정. 이 가정이 깨진 상태에서 Hierarchical을 도입하면 부실한 부하 위에 부실한 상관을 얹는 것.

3. **새 구조 결정 시 인접 문헌 사전 검색**:
   2026-04-30 의사결정 3 당시 Claude가 HRLP 같은 도메인 인접 문헌을 사전 검색했더라면, "별도 네트워크 Hierarchical" 대신 "Dense reward + Options (또는 macro-action)" 조합을 먼저 제안했을 것. SKILL.md §6.3 (인접 문헌 사전 검색 규칙) 으로 반영됨.

4. **Autoresearch 가시성 도구 없이 진단 시도 금지**:
   wandb 같은 가시성 도구 없이는 "천장 vs search 부족" 을 진단할 수 없음. 추측에 기반한 구조 변경 결정은 위험.

5. **"_ing 시스템의 정상 작동"**:
   본 사례는 \_ing 시스템 (CLAUDE_ing/SKILL_ing/PROGRESS_ing/FAILURE_LOG/§99 Lock) 의 정상 작동 사례. Lock 시스템이 vibe coding 차단을 작동시킨 결과, harness engineering 시작 전에 의사결정 3의 오진단이 발견되어 큰 손실 없이 spec 수정 가능. \_ing 시스템 자체의 가치 검증.

**관련 파일/위치:**
- CLAUDE.md §2 (전체 아키텍처 재작성), §11.5 (Layer 1/2 분리), §12 (전이학습 재작성), §16.7 (Dense Reward 신규), §16.8 (wandb 신규), §17 (폐기)
- SKILL.md §2.1 (4번째 철학 추가), §2.2 (재작성), §3.1 (Macro 관련 폐기 + PBS 안전조건 / wandb 통합 추가), §3.5 (PBS 구현 안전장치), §3.6 (wandb 의무), §6.3 (인접 문헌 사전 검색 규칙)
- PROGRESS.md Session 2026-05-14 (의사결정 3-r1, 3-r2, 8, 9, 10, 11)

**관련 의사결정:**
- 의사결정 3 (2026-04-30) → 의사결정 3-r1, 3-r2 (2026-05-14) 로 재검토
- 의사결정 7 (2026-04-30) → 자동 폐기 (Macro 자체가 사라짐)

---

## §99. 첫 entry 작성 시 본 §0~6 보존

본 문서는 작성 시작 후에도 §0~6 (역할, 작성 시점, 카테고리, 양식, 규칙, 회고 절차) 은 그대로 보존한다.
실제 entry 는 본 §99 위쪽에 시간 역순으로 추가한다 (최신 entry 가 위).

```
[FAILURE_LOG.md 최종 구조]

§0~6  : 운영 규칙 (불변)
---
[Entry N+1] (최신)
[Entry N]
...
[Entry 1] (첫 실패)
---
§99   : 안내 (불변)
```

---

**문서 버전:** v1.3 (🎓 졸업판, entry 6개)
**졸업일:** 2026-06-24
**마지막 갱신:** 2026-09-06 (2026-09-06 entry 추가 — w1/w2/w3 배선 누락 + 재발 방지 절차)
**이전 갱신:** 2026-09-01 (Stage 1 캐시 삭제 사고 entry), 2026-08-30 (screening 평가 버그 entry), 2026-06-28 (§16.3.2 산술 오류 archive, 의사결정 27), 2026-06-25 (의사결정 22/23/24), 2026-06-24 (졸업판 동기화), 2026-06-18 (§0.1 dynamic source 역할), 2026-06-11 (entry), 2026-05-14 (첫 entry)
**최신 entry:** 2026-09-06 — w1/w2/w3 가 CLI 인자·params dict 에는 있지만 실제 보상 계산에는 전달된 적이 없었음
**첫 entry:** 2026-05-14 — Phase 1 비효율 경로 + Hierarchical 가설 오진단
