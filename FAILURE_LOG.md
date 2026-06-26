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

**문서 버전:** v1.1 (🎓 졸업판, entry 3개)
**졸업일:** 2026-06-24
**마지막 갱신:** 2026-06-25 (2026-06-25 entry 추가 — skill 충돌 사례. 의사결정 22/23/24)
**이전 갱신:** 2026-06-24 (졸업판 동기화), 2026-06-18 (§0.1 dynamic source 역할), 2026-06-11 (2026-06-11 entry), 2026-05-14 (첫 entry)
**최신 entry:** 2026-06-25 — CLI 자동 로드 skill 과 졸업판 spec 충돌 (120-dim vs 150-dim / 27-action vs 7-direction)
**첫 entry:** 2026-05-14 — Phase 1 비효율 경로 + Hierarchical 가설 오진단
