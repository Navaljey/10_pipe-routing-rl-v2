# 폐기 내역 및 Lock 해제 이력 참조 문서

> CLAUDE.md에서 분리된 참조 문서. 과거 의사결정의 근거(폐기된 Macro 구조, 해제 완료된 Lock 항목 상세, Lock 해제 로그)를 찾을 때 로드한다.
> 현재 활성 Lock 4개(L-A3', L-A4, L-C4, L-D3)는 CLAUDE.md §99 본문에 남아 있다. 아래는 이미 해제/폐기되어 참조용으로만 남은 항목이다.

---

## 17. (폐기) Macro 학습 환경

> **⚠️ 2026-05-14 폐기**: 본 장은 v0.1의 Hierarchical 구조 전제로 작성되었으며, 의사결정 3-r1 (Hierarchical 폐기) 으로 전체 폐기됨.
>
> 폐기된 내용:
> - §17.1 Macro 후보 생성 (A* + Yen's K-shortest paths)
> - §17.2 Visibility Graph 구성
> - §17.3 Macro Observation/Action
> - §17.4 Macro Reward
>
> Step 7~10 다중 파이프 환경의 단일 에이전트 구조 적용 방식은 §12.3 + L-D1 (전이학습 전략) 후속에서 정의됨.

---


---

## 해제 완료된 Lock 항목 상세 (§99에서 이동)

### ~~L-A5: 평가 시나리오 구성~~ (2026-06-18 해제)

해제 내용: §11.0 시나리오 spec 신설로 결정 완료.
- 시나리오 4분할 (학습 무한대 / screening 75 / regression 150 / final eval 500)
- 난이도 분포 3:5:2 + Step별 난이도 축 차별화
- Seed range 분리 체계 (Step별 100,000 단위)
- Procedural 100% 시작 + FAILURE_LOG 누적 수작업 추가
- 회귀 검증 통과 임계값 (Stage 1 / Stage 2 분리)
- 구체 spec: §11.0.1 ~ §11.0.7 참조

### ~~L-A6: Procedural Scenario Generator 알고리즘 spec~~ (2026-06-18 해제)

해제 내용: §11.0.8 신설 (Generator v1.0.0 spec 확정).

핵심 결정:
- 장애물 배치: Poisson disk sampling + Cuboid 합성 (tank/structural/small_eq 비율 4:3:3)
- Start-Goal sampling: rejection sampling + A* reachable 검증 (학습 신호 아님, 시나리오 검증 전용)
- 난이도 파라미터 매핑: `DIFFICULTY_PARAMS` dict + `_inherits` 키로 Step 상속
- Step 4~6 sub-algorithm: 개요만 spec 화, 각 Step 진입 직전 구체화
- Step 7~10 sub-algorithm: Step 7 진입 시점 별도 세션
- Generator version 관리: semver (major.minor.patch) + git tag + hash 보관
- 구체 spec: §11.0.8.1 ~ §11.0.8.6 참조

### ~~L-C2: autoresearch 메타 구조~~ (2026-06-18 해제)

해제 내용: §16.5 Stage 1/2 timestep 명시, §16.6 종료 조건 채움, §16.6.5 Optuna 통합 spec, §16.6.6 wandb naming, §16.6.7 Phase 1 운영 순서 신설.

핵심 결정:
- 1차 스크리닝 시간 비율: 1/8 (정상 2M 의 250K)
- 1차 스크리닝 제거 비율: 50% (successive halving 표준)
- 동시 학습 가능 모델 수: T4 GPU 1개 기준 2~3개 (Pro+/A100 환경에서 4~6개)
- autoresearch Round 상한: 5 (또는 plateau 감지 시 조기 종료)
- Round 단위: Stage 1 + Stage 2 + 평가 ≒ 1.5~3 시간
- Optuna 채택: TPE + MedianPruner + sqlite storage
- wandb naming: `step{N}_round{R}_stage{S}_var{V:03d}` + tag 체계
- 구체 spec: §16.5 ~ §16.6.7 참조

### ~~L-16.3-w: baseline reward 가중치 초기값 및 sweep 범위~~ (2026-06-18 해제)

해제 내용: §16.3.1 Phase 1 초기값, §16.3.2 sweep 범위, §16.3.3 Step 2~6 상대 스케일 원칙 추가.

핵심 결정:
- Step 1 초기값: w1=0.1, w2=2.0, w3=50.0, w4=5.0 (§12.4 고정), w5=15.0 (§12.4 고정)
- Step 1 sweep: w1∈{0.05, 0.1, 0.2, 0.5} × w2∈{1, 2, 5} × w3∈{20, 50, 100} = 36 variants (구 spec "27" 오타 — 의사결정 27)
- 운영 방식: Sequential sweep (Round 1 = α/β, Round 2 = w1/w2/w3)
- Step 2~6 w: Phase 1 후 후행 결정. 상대 스케일 원칙 명시 (hard ~20 / soft ~0.5)
- 구체 spec: §16.3.1 ~ §16.3.3 참조

---

### ~~L-D1: 단일 에이전트 전이학습 전략~~ (2026-06-18 해제)

해제 내용: §12.2 (전이학습 메커니즘) + §12.3 (회귀 감시 체계) 채워짐. 의사결정 13 (2026-06-11) 의 범위 확장 (회귀 감시 임계값/주기/트리거 동시 결정) 도 반영 완료.

핵심 결정:
- 전략: 후보 A (순차 fine-tuning) + 후보 B (mixed curriculum) 혼합. C, D 비채택.
- 학습 데이터 mix: Warm-up 50/40/10 → 정상 60/30/10 (Step N 신규 / Step 1~N-1 / FAILURE_LOG)
- 학습률: 신규 학습의 50% 로 시작
- 회귀 감시 주기: 매 50K timestep
- 트리거: 3단계 escalation + Level 0 즉시 중단 (Layer 1 hard 5% 이상 위반 시)
- 회귀 임계값: §11.0.7 표 (Stage 1 / Stage 2 분리)
- 구체 spec: §12.2 ~ §12.3 참조

### ~~L-D2: Dense Reward Potential function Φ 구체 정의~~ (2026-06-18 해제)

해제 내용: §16.7 (Dense Reward) 본문 채워짐. Phase 1 적용 spec 확정.

핵심 결정:
- Φ 구성: Φ(s) = α·Φ_goal(s) + β·Φ_cong(s). Φ_future 는 Phase 1 미사용 (Phase 2+ 검토)
- Φ_goal = -d_manhattan(s, goal) / d_initial (정규화, 범위 [-1, 0])
- Φ_cong = -1 / (mean_SDF_5x5x5(s) + 1.0)
- Terminal Φ = 0 명시적 set (goal / collision / timeout / out_of_bounds)
- 초기값: α = 1.0, β = 0.1
- Stage 1 sweep: α ∈ {0.5, 1.0, 2.0, 5.0} × β ∈ {0.0, 0.1, 0.3} = 12 variants
- Stage 2: best 주변 ± 50% local grid
- `PotentialBasedShaping` helper class 로 PBS 4대 조건 코드 강제 (§16.7.5)
- wandb 진단 panel: r_shape_to_baseline_ratio, action_value_phi_corr, phi_goal_trajectory 등 (§16.7.8)
- γ_PBS == γ_PPO 강제 (4번째 안전조건 신설)

구체 spec: §16.7.1 ~ §16.7.8 참조

### (폐기된 Lock 항목 - 2026-05-14)

다음 항목은 Hierarchical 폐기로 인해 항목 자체가 폐기됨 (해제가 아님):

| 폐기 Lock | 폐기 사유 |
|----------|----------|
| L-B1 (Macro Observation/Action Space) | Macro 자체 폐기 |
| L-B2 (Macro Reward 설계) | Macro 자체 폐기 |
| L-B3 (다중 파이프 양보 학습 방식) | Macro 의존 항목, L-D1에서 단일 에이전트 방식으로 재정의 |
| L-B4 (Macro 학습용 시나리오 정의) | Macro 자체 폐기, L-A5에 통합 |
| L-12.2 (Macro 아키텍처 결정) | Macro 자체 폐기 |
| L-17.1 (Macro K값 및 다양성 threshold) | Macro 자체 폐기 |

---


## 101. Lock 해제 로그

| 일자 | 해제/폐기 항목 | 결론 요약 |
|------|----------|----------|
| 2026-04-30 | - | 초기 spec 작성, Hierarchical 구조 채택 (의사결정 1~7) |
| 2026-05-14 | **§2 폐기/재작성** | Hierarchical 폐기, 단일 에이전트 + 전이학습 (의사결정 3-r1, 3-r2) |
| 2026-05-14 | **§12.2 폐기** | Step 6→7 Macro 도입 폐기 (의사결정 7 자동 폐기) |
| 2026-05-14 | **§12.3 폐기** | Step 7~10 Macro 신규 정의 폐기 |
| 2026-05-14 | **§13.2 폐기** | Step 6→7 핸드오프 확장 4파일 폐기 |
| 2026-05-14 | **§13.3 폐기** | Macro 핸드오프 폐기 |
| 2026-05-14 | **§17 폐기** | Macro 학습 환경 전체 폐기 |
| 2026-05-14 | **§11.5 보강** | Step 4~6 평가 Layer 1/2 분리 (의사결정 8) |
| 2026-05-14 | **§16.7 신규** | Dense Reward (PBS) 도입 (의사결정 10) |
| 2026-05-14 | **§16.8 신규** | wandb 통합 (의사결정 9) |
| 2026-05-14 | **L-B1/B2/B3/B4/12.2/17.1 폐기** | Macro 관련 Lock 항목 전체 폐기 |
| 2026-05-14 | **L-D1/D2/D3 신규** | 전이학습 전략 / Dense reward Φ / Macro-action 도입 검토 |
| 2026-06-11 | **§99 Lock 명명 규칙 신설** | Lock ID 카테고리 범례 표 내장 (의사결정 12) |
| 2026-06-11 | **L-D1 범위 확장** | 회귀 검증 임계값/주기/트리거 동시 결정 의무 (의사결정 13) |
| 2026-06-18 | **§11.0 신설** | 평가 시나리오 spec — 4분할, 난이도, seed range, 임계값 (L-A5 해제, 의사결정 14) |
| 2026-06-18 | **L-A6 신규** | Procedural Scenario Generator 알고리즘 spec (의사결정 15) |
| 2026-06-18 | **§12.2~12.3 채움** | 전이학습 A+B 혼합 + 회귀 감시 체계 구체화 (L-D1 해제, 의사결정 16) |
| 2026-06-18 | **§16.7 채움** | Dense Reward Φ Phase 1 spec 확정 (L-D2 해제, 의사결정 17) |
| 2026-06-18 | **§16.5~16.6.7 채움** | autoresearch 운영 spec — Stage 1/2 timestep, Optuna, wandb naming, Round 운영 (L-C2 해제, 의사결정 18) |
| 2026-06-18 | **§16.3.1~3 채움** | baseline reward 가중치 — Step 1 초기값 + sweep 범위 + 상대 스케일 원칙 (L-16.3-w 해제, 의사결정 19) |
| 2026-06-18 | **§11.0.8 신설** | Procedural Scenario Generator v1.0.0 알고리즘 — Poisson disk + rejection sampling (L-A6 해제, 의사결정 20) |
| 2026-06-24 | **🎓 \_ing 시스템 졸업** | CLAUDE_ing/SKILL_ing/PROGRESS_ing → CLAUDE/SKILL/PROGRESS. harness engineering 시작 가능 상태. 차단 표 의미 전환 (의사결정 21) |
| 2026-06-25 | **PBS 검증 강화** | state-only closure 경고 + unit test 의무화 (의사결정 22) |
| 2026-06-25 | **skill 충돌 해결** | Option A-1 채택. 로컬 SKILL.md = source of truth. SKILL.md §0.5 신설 (의사결정 23) |
| 2026-06-25 | **졸업 체크리스트 v2** | 서버 skill 동기화 + CLI 검증 항목 추가. SKILL.md §0.5 (의사결정 24) |
| 2026-06-28 | **핸드오프 폴더 분리** | `base_dir/step{N}/` 자동 생성. §13.1 보강 (의사결정 25) |
| 2026-06-28 | **Phase 1 첫 실증** | Sub-단계 5.1 50K baseline: success_rate 94%, PBS ratio 0.631. §16.3.1/§16.7 검증 (의사결정 26) |
| 2026-06-28 | **§16.3.2 산술 오류 수정** | `4×3×3=27` → `36` 오타 수정. 관련 파일 전수 갱신 + FAILURE_LOG entry (의사결정 27) |
| 2026-08-16 | **§18 신설** | 작업 경로 원칙(A안) 추가 — GitHub 정본 / Colab 실행 전용 / 세션 시작 시 git pull 선행 / 예외 시 FAILURE_LOG 기록 (의사결정 28) |
| 2026-08-16 | **CLAUDE.md 문서 분리** | 77KB → 상시 로드 핵심(비전/아키텍처/전 Step 불변 규칙/작업 경로 원칙/활성 Lock)만 CLAUDE.md 유지, 주제별 세부 spec 을 docs/ 6개 참조 문서로 분리 (`pipe-engineering-spec.md`, `evaluation-spec.md`, `transfer-learning-and-handoff.md`, `autoresearch-ops.md`, `ops-and-viz.md`, `deprecated-and-lock-history.md`) (의사결정 29) |

> Lock **해제** 가 아니라 상위 **의사결정 재검토** 결과임에 유의.

---
