---
name: pipe-routing-rl-v2-rules
description: 파이프 자동배치 강화학습 v2 (단일 에이전트 + 전이학습 구조) 프로젝트의 구현 필수 규칙. autoresearch + Dense Reward (PBS) + wandb 통합. 2026-06-24 졸업 — Phase 1 harness engineering 진행 가능 상태. 남은 4개 활성 Lock (L-A3', L-A4, L-C4, L-D3) 은 자기 자연 시점에 해제 예정.
---

# pipe-routing-rl-v2 SKILL

> **🎓 2026-06-24 졸업**: 본 SKILL 은 `_ing` 시스템에서 졸업했다 (`SKILL_ing.md` → `SKILL.md`).
> Phase 1 학습 시작 조건 충족 — harness engineering 진행 가능.
>
> **⚠️ 2026-05-14 구조 결정**: Hierarchical (Macro+Micro 별도 네트워크) 구조 폐기, 단일 에이전트 + 전이학습 구조 채택. Macro 관련 규칙 모두 폐기됨. 자세한 근거는 PROGRESS.md Session 2026-05-14 참조.

---

## 0. 운영 상태 및 활성 Lock 안내

### 0.1 졸업 후 운영 상태 (2026-06-24)

**harness engineering 진행 가능 상태**. 다음 작업 모두 허용:

```
✅ 환경 구현 (env class)
✅ 단일 에이전트 학습 코드 작성 (Step 1 = Phase 1)
✅ Dense Reward (PBS) 구현 — CLAUDE.md §16.7.5 helper class 의무
✅ autoresearch 시스템 구현
✅ 시나리오 생성기 작성 (CLAUDE.md §11.0.8 v1.0.0)
```

### 0.2 남은 4개 활성 Lock (자기 자연 시점 해제)

| Lock | 차단 행위 | 해제 권장 시점 |
|------|----------|---------------|
| L-A3' | Step 1~3 평가 임계값 구체값 (§11.2~11.4) | Phase 1 학습 결과 분포 본 후 후행 결정 |
| L-A4 | Step 4~6 hack 방지 hard constraint (§11.5 Layer 1) | Step 4 진입 시점 |
| L-C4 | grid 좌표계 정밀도 spec | implementation 시 unit test 위임 가능 (사실상 자동 해제) |
| L-D3 | Macro-action 도입 결정 (§17 신규 또는 §16.7 Φ_future) | Phase 1 결과 후 천장 진단 시 |

**위 Lock 들은 모두 Phase 1 학습 시작에 영향 없음.** Step 2 이상 학습도 §12.2~12.3 (전이학습 + 회귀 감시) 구체화 완료로 진행 가능.

### 0.5 서버 등록 skill 과 로컬 SKILL.md 동기 의무 (2026-06-25 신규, 의사결정 23/24)

**배경**: 2026-06-25 harness engineering 첫 세션에서 서버 등록 skill (v1 시절, 120-dim/27-action) 이
졸업판 SKILL.md (150-dim/7-direction) 와 충돌. 졸업 작업 (의사결정 21) 에서 서버 skill 갱신 누락이 원인.

**원칙**: **로컬 SKILL.md 가 source of truth. 서버 등록 skill 은 로컬 파일의 mirror.**

**SKILL 갱신 순서 (절대 준수)**:

```
1. 로컬 SKILL.md 수정
2. git commit (로컬 이력 보존)
3. 서버 skill in-place 갱신 (SKILL.md 내용 그대로)
4. CLI 새 세션 진입 → 로드된 skill 검증 (아래 체크 항목 확인)
```

**충돌 감지 체크 (새 CLI 세션 진입 시 의무)**:

```
다음 항목 로컬 SKILL.md 와 일치 확인:
  ✅ obs dim: 150-dim (zero-padding, Step 1~10 고정)
  ✅ action space: 7-direction discrete
  ✅ 졸업 날짜: 2026-06-24
  ✅ 활성 Lock 목록: L-A3', L-A4, L-C4, L-D3

하나라도 불일치 → Option A-1 (in-place 갱신) 즉시 실행
```

**졸업 체크리스트 v2 (의사결정 24 반영)**:

```
[신규 \_ing 졸업 작업 또는 대규모 spec 갱신 시 체크리스트]
1. 로컬 파일 rename (필요 시)
2. 각 파일 헤더 갱신
3. Cross-reference 일괄 갱신
4. git commit
5. ★ 서버 등록 skill in-place 갱신  ← 의사결정 24 신규
6. ★ CLI 새 세션 진입 → 위 충돌 감지 체크 통과 확인  ← 의사결정 24 신규
```

---

### 0.3 vibe coding 금지 원칙 유지

졸업 후에도 다음 원칙은 유지:

- 새 결정이 필요한 경우 **항상 Lock 해제 절차** (별도 논의 세션) 거침
- spec 본문의 절대 변경 금지 항목 (§3.1) 임의 변경 금지
- 학습 도중 "그냥 reward 좀 더 키워서 돌려보자" 식의 즉흥 결정 금지
- spec 결함 발견 시 → FAILURE_LOG entry 추가 → Lock 재논의 → spec 수정 → 재학습

### 0.4 새 Lock 해제 절차 (졸업 후 운영)

```
1. 사용자가 명시적으로 "L-XXX를 해제하자" 라고 요청
2. 별도 논의 세션 진행 (Claude AI 세션 권장)
3. 결론 도달 후 CLAUDE.md 업데이트
4. SKILL.md 동시 업데이트 (필요 시)
5. PROGRESS.md 에 새 의사결정 entry 추가
6. (영향 시) FAILURE_LOG.md entry 추가
```

---

## 1. Trigger Keywords

다음 키워드가 사용자 메시지에 등장하면 본 SKILL이 활성화된다:

```
파이프 라우팅, pipe routing, 파이프 자동배치, pipe-routing-rl,
단일 에이전트 + 전이학습 (이 프로젝트 맥락),
Dense Reward, PBS, Potential-Based Shaping (이 프로젝트 맥락),
autoresearch (이 프로젝트 맥락),
wandb (이 프로젝트 맥락),
A* benchmark (이 프로젝트 맥락),
CLAUDE.md, SKILL.md,
새 spec, v2 구조
```

---

## 2. 본 프로젝트의 정체성 (절대 흔들리면 안 됨)

### 2.1 핵심 철학 4가지 (2026-05-14 업데이트)

1. **숙련 엔지니어 시연 = 정답 채택 금지**
   - 시연을 정답으로 삼으면 시연이 천장이 됨
   - 본 프로젝트의 목적은 엔지니어를 **뛰어넘는 것**
   - 평가는 물리적 측정 가능 metric으로 (Step 4~6 은 Layer 1 constraint + Layer 2 relative quality)

2. **A\* = 모범답안이 아니라 평가 도구**
   - A\*는 학습 신호로 쓰지 않음
   - 학습 후 평가 시점에만 비교 baseline으로 활용
   - Step 1~3 에서만 적용. Step 4~6은 물리 metric 사용

3. **autoresearch는 Claude API + RL 학습 + 평가의 자동 순환**
   - reward 가중치 / 항목 / hyperparameter 변형
   - 한 번에 한 인자만 변형 (ablation 원칙)
   - 새 reward 항목은 4-Gate 자동 검증 통과 필수
   - **wandb 통합 의무** (모든 run 추적)

4. **단일 에이전트 + 전이학습 (Hierarchical 폐기)**
   - Step 1~10 전체를 단일 네트워크로 학습
   - Step 간은 전이학습으로 연결 (구체 전략 L-D1)
   - Dense Reward (PBS) 로 sparse reward 문제 완화
   - Macro-action 도입은 Phase 1 후 결정 (L-D3)

### 2.2 구조적 정체성 (2026-05-14 재작성)

```
[Step 1~6] Single Agent 학습
  Backbone hidden dim 256
  150-dim observation (zero-padding)
  MaskablePPO + action_masks()
  Reward = baseline (§16.3) + Dense PBS (§16.7)

[Step 6 → 7 전환]
  Macro 도입 의식 없음 (Hierarchical 폐기)
  Step 6 정책 → Step 7 정책 초기화 (전이학습 전략 L-D1)
  Observation 150-dim 유지 (다른 파이프 정보를 obs[136:150] 활용)

[Step 7~10] Single Agent 계속
  Backbone hidden dim 256 (동일, 변경 절대 금지)
  매 학습마다 Step 1~6 회귀 검증 통과 필수
  양보 (yielding) 학습은 reward 설계 + obs 확장으로 처리 (L-D1 후속)
```

---

## 3. 코딩 규칙 (Lock 해제 후 적용)

### 3.1 절대 변경 금지 항목 (2026-05-14 업데이트)

| 항목 | 값 | 이유 |
|------|-----|------|
| Backbone hidden dim | 256 (Step 1~10 전체) | 전이학습 보장 |
| Step 1~10 Observation dim | 150 (zero-padding) | 전이학습 보장 |
| Action space 기본 정의 | 7-direction discrete | 마스킹 외 변경 불가, **macro-action 추가 검토는 L-D3** |
| RL 알고리즘 | sb3-contrib MaskablePPO | spec 고정 |
| 좌표계 | 오른손 좌표계, mm 단위 | 모듈 간 일관성 |
| Face 6방향 제약 | start/goal direction (전 Step) | 물리 원칙 |
| PBS 안전조건 | Φ는 state-only / Φ(terminal)=0 / γΦ(s')-Φ(s) / **γ_PBS == γ_PPO** | 정책 보존 이론적 보장 |
| **PBS 구현 방식** (2026-06-18 신규) | `PotentialBasedShaping` helper class 만 사용. 직접 r_shape 작성 금지 | 4대 조건 우회 방지 |
| wandb 통합 | 모든 autoresearch run 의무 | 가시성 / 재현성 |

> **폐기 항목 (Hierarchical 폐기 부수)**: Macro Backbone hidden dim 512, Macro 별도 obs space, Macro freeze/unfreeze 의식. 모두 항목 자체가 사라짐.

### 3.2 시각화 제약

- **Matplotlib (mpl_toolkits.mplot3d) + IPython HTML만 사용**
- Plotly 사용 금지
- PyVista 사용 금지
- 가상 디스플레이(xvfb 등) 사용 금지

### 3.3 환경 메모리 제약

- Occupancy grid: `bool` 타입
- SDF Field: `uint8` 양자화
- Gradient: 별도 저장 금지 (런타임 차분 계산)
- Partial EDT 업데이트 (전체 EDT는 episode reset 시 1회만)

### 3.4 학습/평가 분리 원칙

```
[학습 시점]
  - A* 사용 금지 (학습 신호로 쓰면 안 됨)
  - 숙련공 시연 사용 금지

[평가 시점]
  - Step 1~3: A* benchmark와 비교
  - Step 4~6:
      Layer 1 - Hard constraint 충족 여부 (binary)
      Layer 2 - 물리 metric 상대 비교 (kg, mm, autoresearch 변형 ranking)
  - Step 7~10: 다중 파이프 metric (구체는 L-D1 후속)
```

### 3.5 Dense Reward 구현 안전장치 (2026-05-14 신규, 2026-06-18 구체화)

```
PBS (Potential-Based Shaping) 형식만 허용:
  r_shape = γ_PPO · Φ(s') - Φ(s)

Φ 설계 시 4-Gate 검증과 별도로 PBS 4대 조건 hard check:
  1. Φ(s) 는 state만의 함수 (action 의존 시 reject)
  2. Φ(terminal) == 0 (위반 시 reject) — goal / collision / timeout / out_of_bounds 전부
  3. r_shape 수식이 γΦ(s') - Φ(s) 형식 (단순 cost 차이 시 reject)
  4. γ_PBS == γ_PPO (불일치 시 reject) — 2026-06-18 신규
```

**구현 의무 (2026-06-18, L-D2 해제)**:

```python
# CLAUDE.md §16.7.5 의 PotentialBasedShaping helper class 사용 의무
# 직접 r_shape 작성 금지 (PBS 조건 우회 위험)

from your_project.shaping import PotentialBasedShaping

pbs = PotentialBasedShaping(
    phi_fn=lambda s: alpha * phi_goal(s) + beta * phi_cong(s),
    gamma_ppo=model.gamma,  # PPO 의 gamma 와 동일 값 강제
    terminal_phi=0.0,
)
r_shape = pbs.shaping_reward(s, s_next, is_terminal)
```

**Phase 1 적용 Φ (CLAUDE.md §16.7.2)**:

```
Φ(s) = α · Φ_goal(s) + β · Φ_cong(s)
  Φ_goal(s) = -d_manhattan(s, goal) / d_initial    범위 [-1, 0]
  Φ_cong(s) = -1 / (mean_SDF_5x5x5(s) + 1.0)       범위 음수, |Φ_cong| ~ O(1)
  Φ_future  = Phase 1 미사용

초기값: α=1.0, β=0.1
Stage 1 sweep: α ∈ {0.5, 1.0, 2.0, 5.0} × β ∈ {0.0, 0.1, 0.3}
```


### 3.6 wandb 통합 의무 (2026-05-14 신규)

```python
# 모든 autoresearch 학습 run에 callback 필수
from wandb.integration.sb3 import WandbCallback

wandb.init(
    project="pipe-routing-rl",
    name=f"step{N}_variant_{variant_id}",
    config={...},
    tags=[f"step{N}", "autoresearch"]
)
model.learn(callback=WandbCallback(verbose=2))

# 핸드오프 파일에 wandb URL 기록 의무 (stepN_wandb_run_url.txt)
```

### 3.7 회귀 감시 의무 (2026-06-11 신규, 2026-06-18 구체화)

**전이학습/curriculum 학습은 가중치 보존을 보장하지 않는다.** Catastrophic forgetting은 학습 도중 어느 시점에도 발생 가능하며, 특히 새 step 학습 초반에 가장 빠르게 진행된다. 따라서 본 프로젝트는 **회귀 검증을 학습 종료 후 1회가 아니라 학습 중 주기적 모니터링으로** 운영한다.

**Step N ≥ 2 학습 시 의무 사항:**

```
1. 평가 콜백 주기적 실행 (2026-06-18 구체화, L-D1 해제):
   - 매 25 rollout (≒ 50K timestep) 마다
   - Step 1 ~ Step N-1 의 회귀 시나리오 150개 모두 평가
   - 결과를 wandb 에 학습 reward 곡선과 같은 panel 에 기록

2. 회귀 지표 산출 (3종 동시 기록):
   - 각 과거 step 의 success_rate / length_ratio / Layer 1 통과율
   - Backward Transfer (BWT):
       BWT_k = perf_k(end_of_stepN) - perf_k(end_of_stepK)
       음수 = 망각, 0 = 보존, 양수 = 후속 학습이 과거를 도움
   - Forgetting measure:
       Fk = max_{t ≤ now} perf_k(t) - perf_k(now)
       과거 step k 의 역대 최고 대비 현재 격차

3. 회귀 임계값 (2026-06-18 구체화, CLAUDE.md §11.0.7 참조):
   Stage 2 (정상 학습) 기준:
     - success_rate 손실 ≤ 5%p
     - length_ratio 증가 ≤ 10%
     - bend_count_ratio 증가 ≤ 15% (Step 2+)
     - Layer 1 hard constraint 100% 절대 (Step 3+)
     - BWT_k ≥ −0.05
     - Forgetting measure F_k ≤ 0.10
   Stage 1 (screening) 은 위 값들이 약 2배 느슨.

4. 트리거 escalation (CLAUDE.md §12.3.2 구체):
   Level 0 — 즉시 중단 (Layer 1 hard 5% 이상 위반)
   Level 1 — 경고 (1번째 soft 위반) → wandb alert tag
   Level 2 — 자동 조정 (2번째 연속 위반) → mixed curriculum 30%→45%, 학습률 50% 감소
   Level 3 — 학습 중단 (3번째 연속 위반) → 체크포인트 복귀, 사용자 알림
```

**가중치 자체를 보는 지표는 보조용으로만 사용한다:**

```
‖θ_N − θ_{N-1}‖, layer 별 cosine similarity 등은 wandb 보조 panel 로만
→ "어느 layer 가 흔들렸는가" 정도의 진단 단서
→ 가중치 변화량 ≠ 능력 변화량 (보존/망각 판정은 항상 성능 기반)
```

**회귀 시나리오 set 의 사전 조건 (2026-06-18 해제됨):**

- ~~L-A5 (평가 시나리오 구성) 해제 전에는 회귀 감시 의미 약함~~ → 2026-06-18 L-A5 해제로 충족
- 회귀 시나리오: Step 별 150개 고정 seed (Easy 45 / Medium 75 / Hard 30) — CLAUDE.md §11.0.1 참조
- Seed range: Step N 의 N\*100000+20000 ~ N\*100000+29999 범위 (CLAUDE.md §11.0.4)

### 3.8 시나리오 Generator 의무 (2026-06-18 신규, L-A6 해제)

```
[Generator 사용 의무]
모든 시나리오 (학습 / screening / regression / final eval) 는
CLAUDE.md §11.0.8 의 procedural generator v1.0.0 으로 생성.

[Generator API contract]
def generate_scenario(seed: int, step: int, difficulty: str) -> Scenario:
    """
    Args:
        seed:        고정 seed (§11.0.4 range 분리 체계 준수)
        step:        1~10 (Step 번호)
        difficulty:  "easy" | "medium" | "hard"
    
    Returns:
        Scenario(grid, start, goal, start_dir_idx, goal_dir_idx, ...)
    
    Raises:
        ScenarioGenerationFailure: rejection sampling 1000회 실패 시
    """

[Generator version 기록 의무]
- 시나리오 set YAML 파일에 generator_version 필드 (§11.0.6)
- wandb config 에 generator_version (§16.6.6)
- major version 변경 시 PROGRESS.md 의사결정 entry
```

### 3.9 Phase 1 가중치 초기값 (2026-06-18 신규, L-16.3-w 해제)

CLAUDE.md §16.3.1 참조. SKILL 에서는 운영 규칙만 명시:

```
[Step 1 가중치 — Phase 1 시작 초기값]
  w1 (length)         = 0.1
  w2 (collision)      = 2.0
  w3 (goal_bonus)     = 50.0
  w4 (direction_align) = 5.0    ← §12.4 고정 (sweep 대상 아님)
  w5 (wrong_dir)      = 15.0    ← §12.4 고정 (sweep 대상 아님)

[Step 1 autoresearch sweep — Round 2]
  w1 ∈ {0.05, 0.1, 0.2, 0.5} × w2 ∈ {1, 2, 5} × w3 ∈ {20, 50, 100}
  → 27 variants

[Step 2~6 가중치 (w6~w17)]
  Phase 1 후 각 Step 진입 직전 후행 결정.
  상대 스케일 원칙 (hard ~20 / soft ~0.5) 만 CLAUDE.md §16.3.3 명시.
```

### 3.10 단일 파일 / 모듈 분리 규칙 (2026-06-24 졸업판)

본 SKILL §5 (파일 구조) 권장 폴더 구조를 따른다. 추가 모듈 분리 규칙:

- 한 파일 한 책임 (single responsibility): env, shaping, generator, autoresearch, training 분리
- env class 는 base + step1 분리. Step 2~10 추가 시 base 상속
- shaping 의 PotentialBasedShaping 은 다른 곳에서 import 만 (정의 단일점)
- test 파일은 src 와 1:1 대응 (test_pbs_safety.py ↔ shaping/potential_based.py 등)
- 새 모듈 추가 시 PROGRESS.md 에 의사결정 entry (구조 변경 추적)

---

## 4. autoresearch 운영 규칙

### 4.1 변형 인자 분류 (2026-05-14 업데이트)

```python
# A. 자유 변형 가능
ALLOWED_FREE = [
    'reward_weights',  # 모든 baseline 가중치
    'reward_items',    # 추가/제거 (단, 4-Gate 통과)
    'pbs_alpha',       # 2026-06-18 구체화: Φ_goal 가중치 α
    'pbs_beta',        # 2026-06-18 구체화: Φ_cong 가중치 β
    'pbs_phi_goal_norm',     # 2026-06-18 신규: d_initial 정규화 방식 (manhattan/euclidean/sdf-aware)
    'pbs_phi_cong_window',   # 2026-06-18 신규: SDF window size (3³/5³/7³)
    'learning_rate', 'gamma', 'n_steps', 'batch_size', 'ent_coef',
    'activation', 'dropout', 'layer_norm_position', 'optimizer_type',
]

# B. 변경 절대 금지
FORBIDDEN = [
    'hidden_dim',         # 256 fixed (Step 1~10 전체)
    'observation_dim',    # 150 fixed
    'action_space_definition',  # 마스킹은 변경 가능, 정의 불변
    'major_layer_count',
    'pbs_safety_conditions',  # 2026-05-14 신규: PBS 4대 조건 (§3.5)
    'pbs_form',               # 2026-06-18 신규: γΦ(s') - Φ(s) 형식 불변
    'pbs_terminal_zero',      # 2026-06-18 신규: Φ(terminal) = 0 불변
    'pbs_state_only',         # 2026-06-18 신규: Φ state-only 시그니처 불변
    'pbs_gamma_sync',         # 2026-06-18 신규: γ_PBS == γ_PPO 불변
]

# C. 조건부 허용 (case by case 검토 필수)
CONDITIONAL = [
    'minor_hidden_dim_adjust',  # 다음 step 진입 시 원복 가능해야 함
    'layer_addition',           # 가중치 mapping 명확해야 함
    'macro_action_addition',    # 2026-05-14 신규: L-D3 해제 후만 가능
    'phi_future_addition',      # 2026-06-18 신규: Phase 2+ Φ_future 도입 (별도 4-Gate + PBS 안전조건 통과)
]
```

### 4.2 4-Gate 자동 검증 (새 reward 항목 추가 시)

```python
def validate_new_reward_item(proposal):
    if not is_grid_computable(proposal.formula):
        return REJECT("Gate 1: Not grid-computable")
    if conflicts_with_existing(proposal):
        return REJECT("Gate 2: Conflicts with existing items")
    if hurts_evaluation_metric(proposal):
        return REJECT("Gate 3: Misaligned with evaluation")
    if sign_inconsistent(proposal):
        return REJECT("Gate 4: Sign inconsistent")
    
    # 2026-05-14 신규: PBS 항목은 추가 안전조건 (2026-06-18: 4대 조건으로 확장)
    if proposal.is_pbs:
        if not is_state_only(proposal.phi):
            return REJECT("PBS Gate 1: Phi depends on action")
        if not phi_terminal_is_zero(proposal.phi):
            return REJECT("PBS Gate 2: Phi(terminal) != 0")
        if not is_pbs_form(proposal.formula):
            return REJECT("PBS Gate 3: Not γΦ(s')-Φ(s) form")
        if proposal.gamma_pbs != proposal.gamma_ppo:    # 2026-06-18 신규
            return REJECT("PBS Gate 4: γ_PBS != γ_PPO")
        if not uses_helper_class(proposal):              # 2026-06-18 신규
            return REJECT("PBS Gate 5: Must use PotentialBasedShaping helper class")
    
    return ACCEPT
```

### 4.3 동시 실행 원칙

- **한 번에 한 인자만 변형** (ablation 원칙)
- 같은 인자의 다른 값을 동시 학습은 허용 (예: w_length sweep)
- 두 인자를 동시에 바꾸는 것 금지

### 4.4 2-Stage Screening (2026-06-18 L-C2 해제, 구체값 명시)

```
정상 학습 시간 (Phase 1 Step 1):  2,000,000 timestep

Stage 1 (1차 스크리닝):  250,000 timestep (정상의 1/8)
  변형 N개 모두 학습
  하위 50% 제거 (successive halving 표준)

Stage 2 (정상 학습):       2,000,000 timestep
  살아남은 후보만 정상 학습
  최종 평가 → 1개 채택

동시 학습 가능 모델 수:  T4 GPU 1개 기준 2~3개
                          (Pro+ / A100 환경에서 4~6개 가능)

종료 조건 (OR):
  - Round 5 도달 (절대 상한)
  - 최근 2 Round best 변형 동일 (개선 정체)
  - wandb best variant 학습 곡선 plateau (마지막 200K success_rate 증가 < 2%p)
  - 사용자 강제 종료
```

구체 spec: CLAUDE.md §16.5~16.6 참조.

### 4.5 Sequential Sweep + Local Grid

```
1. Sequential sweep: 인자 A → B → C ... (각각 1차원 sweep)
2. Local grid: 좋은 영역 주변에서 인자 간 grid search
3. interaction 효과 포착
```

### 4.6 wandb 가시성 활용 (2026-05-14 신규)

```
Stage 1 screening 결과 분석:
  wandb parallel coordinates plot 으로 변형 간 패턴 즉시 확인
  
천장 진단:
  best variant 결과의 시간순 진화 그래프
  - 단조 증가 지속 → search 부족
  - 정체 plateau → 천장 가능성

인자 importance:
  Optuna importance analysis (autoresearch가 Optuna 기반인 경우)
  → "어떤 인자가 결과 분산을 가장 많이 설명하는지" 정량 추출
```

### 4.7 Optuna 통합 (2026-06-18 신규, L-C2 해제)

```python
import optuna

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(n_startup_trials=12),  # Stage 1 grid 보장
    pruner=optuna.pruners.MedianPruner(),                      # plateau 자동 차단
    storage="sqlite:///autoresearch.db",                        # 영구 보관
    study_name=f"step{N}_round{R}",
)

# Stage 1: 명시 grid enqueue (sweep 보장)
for variant in stage1_grid:
    study.enqueue_trial(variant)

# Stage 2 이후: TPE 가 알아서 sampling
study.optimize(objective, n_trials=N_total)

# Round 종료 시 importance 추출 (다음 Round sweep 대상 결정 근거)
importances = optuna.importance.get_param_importances(study)
```

구체 spec: CLAUDE.md §16.6.5 참조.

### 4.8 wandb naming 체계 (2026-06-18 신규, L-C2 해제)

```python
wandb.init(
    project="pipe-routing-rl",
    name=f"step{N}_round{R}_stage{S}_var{V:03d}",
    config={
        "step": N, "round": R, "stage": S, "variant_id": V,
        "alpha": ..., "beta": ...,           # PBS (L-D2)
        "w1": ..., "w2": ..., "w3": ...,     # baseline (L-16.3-w)
        "git_commit": ..., "generator_version": ...,
    },
    tags=[f"step{N}", f"round{R}", f"stage{S}", "autoresearch",
          "phase1" if N == 1 else "phase2plus"],
)
```

parallel coordinates plot 의미를 위해 sweep 대상 모두 config logging 의무.

### 4.9 Phase 1 autoresearch Round 운영 순서 (2026-06-18 신규)

```
Round 1: L-D2 α × β sweep (12 variants)
  Stage 1: 250K × 12 → 하위 6 제거 / Stage 2: 2M × 6 → best 1
  결과: best (α, β) 확정

Round 2: L-16.3-w 가중치 sweep (27 variants)
  α, β = Round 1 best 고정
  Stage 1: 250K × 27 → 하위 14 제거 / Stage 2: 2M × 13 → best 1
  결과: best (w1, w2, w3) 확정

Round 3 (optional): hyperparameter (learning_rate, batch_size, ent_coef, gamma)
Round 4 (optional): interaction grid (α × w3, β × w1 등)
Round 5 (조건부): 추가 reward 항목 (4-Gate 통과 시)
```

구체 spec: CLAUDE.md §16.6.7 참조.

---

## 5. 파일 구조 (2026-06-24 졸업판 권장안)

```
pipe-routing-rl-v2/
├── CLAUDE.md                        ← 본 프로젝트 spec 본체 (졸업판)
├── SKILL.md                         ← 본 SKILL (졸업판)
├── PROGRESS.md                      ← 의사결정 이력
├── FAILURE_LOG.md                   ← 실패 / 시행착오 이력
├── reference/
│   └── common_spec.md               ← 기존 spec (참조용 보존)
│
├── envs/                            ← 환경 구현
│   ├── __init__.py
│   ├── base_env.py                  ← Step 1~10 공통 base class
│   └── step1_env.py                 ← Phase 1 첫 구현
│
├── shaping/                         ← Dense Reward (PBS)
│   └── potential_based.py           ← PotentialBasedShaping helper class (§16.7.5)
│
├── generators/                      ← 시나리오 생성 (v1.0.0)
│   ├── __init__.py
│   ├── poisson_disk.py
│   ├── scenario_generator.py        ← §11.0.8.1~11.0.8.2
│   ├── difficulty_params.py         ← §11.0.8.3 DIFFICULTY_PARAMS dict
│   └── version.py                   ← generator_version 관리
│
├── scenarios/                       ← 시나리오 set (§11.0.4 seed range)
│   ├── step1/
│   │   ├── screening_seeds.yaml     ← 75개
│   │   ├── regression_seeds.yaml    ← 150개
│   │   ├── final_eval_seeds.yaml    ← 500개
│   │   └── manual_seeds/            ← FAILURE_LOG 누적 시 채워짐 (§11.0.5)
│   └── ... (step2~10, 각 Step 진입 시 생성)
│
├── autoresearch/                    ← autoresearch 운영
│   ├── optuna_study.py              ← TPE + MedianPruner (§16.6.5)
│   ├── wandb_callback.py            ← wandb 통합 (§16.6.6)
│   ├── stage_runner.py              ← 2-Stage screening (§16.5)
│   └── round_orchestrator.py        ← Round 운영 (§16.6.7)
│
├── training/                        ← 학습 entry point
│   ├── train_step1.py               ← Phase 1 학습
│   ├── regression_callback.py       ← 회귀 감시 (§12.3)
│   └── transfer_learner.py          ← 전이학습 mixed curriculum (§12.2)
│
├── tests/                           ← unit test
│   ├── test_pbs_safety.py           ← PBS 4대 조건 (§3.5)
│   ├── test_coord_conversion.py     ← L-C4 좌표 unit test
│   ├── test_generator.py            ← Generator 시나리오 시각 검증
│   ├── test_reward_scale.py         ← reward 단위 분석 (§16.3.1)
│   └── test_regression_callback.py  ← 회귀 감시 callback
│
├── handoffs/                        ← Step 간 핸드오프 (§13)
│   └── step1/                       ← Phase 1 종료 시 채워짐
│       ├── stepN_policy.zip
│       ├── stepN_regression_report.json
│       └── stepN_wandb_run_url.txt
│
├── pyproject.toml                   ← 의존성 관리 (poetry / uv 등)
├── .gitignore
└── .git/                            ← git 추적 의무
```

> **운영 의무**: 모든 파일 변경은 git commit 으로 추적. PROGRESS.md 의사결정 entry / FAILURE_LOG entry 추가 시 같은 commit 또는 별도 commit 으로 자기 자체 추적.

---

## 6. 사용자가 본 SKILL 활성화 시점에 Claude가 해야 할 것

### 6.1 첫 인사 후 즉시 확인 (2026-06-24 졸업 후 갱신)

```
[harness engineering 단계 (현재)]
1. 사용자 요청 분석:
   - 코드 변경 요청 → §3 코딩 규칙 자동 적용하며 진행
   - spec 변경 요청 → "Claude AI 세션에서 별도 논의 권장" 안내
   - "L-XXX 해제" 요청 → 별도 논의 세션 진행

2. 활성 Lock 4개 (L-A3', L-A4, L-C4, L-D3) 의 자기 자연 시점 도래 시:
   - L-A3': Phase 1 학습 완료 후 "결과 분포 보고 임계값 정하자" 제안
   - L-A4:  Step 4 진입 임박 시 "hard constraint 정의가 필요하다" 안내
   - L-C4:  env class 구현 시 "좌표 unit test 작성" 자동 제안
   - L-D3:  Phase 1 plateau 감지 시 "macro-action 도입 검토 필요" 안내

3. vibe coding 차단 유지:
   "그냥 reward 좀 키워서 돌려봐" 같은 즉흥 요청 → Lock 해제 절차 제안
   spec 결함 발견 시 → FAILURE_LOG entry 추가 + 재논의

4. 2026-05-14 이후: 사용자가 "Macro" 또는 "Hierarchical" 언급 시
   "해당 구조는 2026-05-14 의사결정 3-r1 에서 폐기됨" 명확히 안내
```

### 6.2 Lock 해제 세션 진행 방식

```
1. 해당 Lock 항목의 배경 / 의존성 설명
2. 결정해야 할 sub-questions 제시
3. 사용자 답변 받기
4. 막힐 만한 지점 / 추가 결정 필요 항목 짚기
5. 합의 시 lock 해제안 정리
6. 사용자 최종 confirm
7. CLAUDE.md / SKILL.md 업데이트
8. PROGRESS.md 에 새 Session 추가
9. §101 Lock 해제 로그에 기록
```

### 6.3 새 의사결정 시 인접 문헌 사전 검색 (2026-05-14 신규)

```
2026-05-14 Claude 자기비판 반영:
  - 의사결정 3 (Hierarchical 채택) 당시 HRLP 같은 도메인 인접 문헌 검색 안 함
  - 결과: 잘못된 처방을 권유, 사용자가 한 차례 더 검증해서 폐기
  
교훈: 새 구조 결정 / 알고리즘 선택 시
  - 도메인 인접 분야 (RL for placement / routing / planning) 최근 논문 사전 검색
  - 최소 2-3편 참조 후 의사결정 제안
  - 검색 결과를 PROGRESS의 "기각된 대안" 또는 "근거" 에 명시적으로 인용
```

### 6.4 절대 하지 말 것

- ❌ Lock 해제 안 된 상태에서 코드 작성 (vibe coding)
- ❌ 사용자 의도 추측해서 Lock 항목 임의 결정
- ❌ "일단 시작하고 나중에 결정" 식 진행
- ❌ 본 SKILL의 절대 변경 금지 항목 (§3.1) 변경 시도
- ❌ Hierarchical / Macro / Micro 구조 재도입 (2026-05-14 폐기 결정)
- ❌ PBS 안전조건 (§3.5) 위반하는 dense reward 제안
- ❌ **`PotentialBasedShaping` helper class 우회하여 r_shape 직접 작성** (2026-06-18 신규, L-D2 해제, §16.7.5 참조)
- ❌ **PPO 의 gamma 와 다른 값으로 PBS γ 사용** (2026-06-18 신규, PBS Gate 4 위반)
- ❌ **회귀 검증 임계값 / 측정 주기 / 트리거 대응이 미정 상태에서 Step N+1 학습 시작** (2026-06-11 신규, §3.7 의무 참조)
- ❌ **가중치 변화량 norm 만으로 능력 보존 판정** (2026-06-11 신규, 항상 성능 기반 검증)
- ❌ **`inspect.signature()` parameter 수 1개만 보고 state-only 확정으로 단정** (2026-06-25 신규, 의사결정 22 — closure 캡처 가능성 항상 고려)

### 6.5 Lock 명명 규칙 (CLAUDE.md §99 참조)

Lock ID 의 카테고리 식별자 (L-A / L-B / L-C / L-D / 섹션 직참조) 범례는 **CLAUDE.md §99 의 "Lock 명명 규칙" 표가 single source of truth** 다. 본 SKILL 에서는 별도 표를 두지 않고 해당 위치를 참조한다.

---

## 7. 본 SKILL 자체의 업데이트 규칙

본 SKILL.md 는 다음 시점에 업데이트된다:

1. CLAUDE.md 의 §99 활성 Lock 항목 해제 시 (남은 L-A3', L-A4, L-C4, L-D3)
2. CLAUDE.md 의 §3.1 절대 변경 금지 항목 변경 시
3. 새로운 절대 규칙 발견 시 (학습 중 spec 결함 발견 등)
4. 상위 의사결정 (PROGRESS 의사결정 N) 재검토 / 폐기 / 신규 시

### 7.1 졸업 후 운영 모드

```
[일상 운영]
- Claude Code CLI 에서 코드 변경 시 본 SKILL 자동 적용
- §3 코딩 규칙 (Generator 의무, PBS helper, 회귀 감시, wandb 등) 자동 준수
- §3.1 절대 변경 금지 항목 임의 변경 차단

[새 의사결정 발생 시]
- Claude AI 세션에서 별도 논의 후 spec 변경
- 4개 파일 (CLAUDE.md / SKILL.md / PROGRESS.md / FAILURE_LOG.md) 일관 갱신
- git commit 으로 변경 추적

[학습 실패 / spec 결함 발견 시]
- FAILURE_LOG entry 추가 (시간 역순)
- 실패 케이스 → 수작업 seed pool 영구 등록 (CLAUDE.md §11.0.5)
- 영향받는 Lock 재논의 가능
```

### 7.2 향후 졸업 마일스톤

- **Phase 1 완료**: Step 1 학습 결과 분석 후 L-A3' 해제
- **Step 4 진입**: L-A4 해제
- **Step 7 진입**: §12.3.4 다중 파이프 spec 보강 + L-D1 후속 결정
- **모든 Step 완료**: 본 v2 spec 의 최종 완성. 모든 활성 Lock 해제 상태.

---

**문서 버전:** v1.1
**졸업일:** 2026-06-24
**마지막 갱신:** 2026-06-25 (§0.5 신설 — 서버 등록 skill 동기 의무. §6.4 PBS signature 단정 금지 추가. 의사결정 23/24)
**이전 갱신:** 2026-06-24 v1.0 (\_ing 시스템 졸업), 2026-06-18 v0.6 (3개 Lock 추가 해제), 2026-06-18 v0.5 (L-A5/L-D1/L-D2), 2026-06-11, 2026-05-14
