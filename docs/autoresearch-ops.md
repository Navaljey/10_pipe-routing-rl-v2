# autoresearch 운영 참조 문서

> CLAUDE.md에서 분리된 참조 문서. autoresearch 루프(reward sweep, Optuna, wandb), baseline reward 가중치, 4-Gate 검증, Dense Reward(PBS) 구현/튜닝 작업 시 로드한다.
> "변경 절대 금지" 항목(원 §16.2.B, backbone hidden dim / obs 150-dim / action space 등)은 모든 세션에서 항상 지켜야 하는 guardrail이라 CLAUDE.md 본문에도 요약이 남아 있다. 아래 §16.2는 원문 전체(A/B/C)를 그대로 보존한다.

---

## 16. autoresearch 운영 방침

### 16.1 기본 원칙

```
autoresearch = Claude API + RL 학습 + 자동 평가의 순환 루프
목적: 각 Step에서 baseline reward를 출발점으로
      변형을 자동 생성·학습·평가하여 최적 reward 채택
```

### 16.2 변형 대상 인자

#### A. 변형 가능

| 분류 | 항목 | 비고 |
|------|------|------|
| Reward 가중치 | 모든 reward 항목의 가중치 | Claude API 자유 판단 |
| Reward 항목 | 새 항목 추가 / 기존 항목 제거 | **§16.4 자동 게이트 통과 필수** |
| Hyperparameter | learning rate, gamma, n_steps, batch_size, ent_coef 등 | 자유 |
| 부가 구조 | Activation, Dropout, LayerNorm 위치, Optimizer 종류 | 자유 |

#### B. 변경 절대 금지 (전이학습 보장)

| 항목 | 이유 |
|------|------|
| Backbone hidden dim | Step 1~10 전체 256 (전 Step 동일) |
| Observation dim 구조 | 150-dim 고정 슬롯 할당 |
| Action space 정의 | 마스킹은 변경 가능, 기본 정의 불변 (※ macro-action 도입 시 L-D3 재해석) |
| Layer 수의 큰 변경 | 가중치 mapping 호환성 |

#### C. 조건부 허용

| 항목 | 조건 |
|------|------|
| Hidden dim 미세 조정 (예: 256 → 224 / 288) | 다음 step 진입 시 256으로 되돌릴 수 있어야 함 |
| Layer 추가 (residual 등) | 가중치 mapping 명확해야 함 |

### 16.3 baseline reward function (Step별)

```
Step 1: r = -w1·length - w2·collision + w3·goal_bonus
        + w4·direction_align - w5·wrong_dir

Step 2: Step 1 + (-w6·bend_radius_violation - w7·excess_bend_count)

Step 3: Step 2 + (-w8·slope_violation - w9·z_uplift_in_gravity)

Step 4: Step 3 + (-w10·branch_total_length_excess
                   - w11·branch_angle_violation
                   - w12·branch_on_bend)

Step 5: Step 4 + (-w13·valve_height_violation
                   - w14·valve_clearance_violation
                   - w15·valve_missing)

Step 6: Step 5 + (-w16·support_interval_violation
                   - w17·total_system_weight_excess)
```

> **baseline의 항목과 부호는 lock**. autoresearch는 가중치 값 변형 + 새 항목 추가 가능.

#### 16.3.1 Phase 1 가중치 초기값 (2026-06-18, L-16.3-w 해제)

```
[Step 1 — Phase 1 시작 초기값]
  w1 (length)         = 0.1     ← step 당 가벼운 길이 penalty
  w2 (collision)      = 2.0     ← 마스킹 외 충돌 억제, length 의 2배 자릿수
  w3 (goal_bonus)     = 50.0    ← 도달 시 +50, episode penalty 대비 약 3배
  w4 (direction_align) = 5.0    ← §12.4 고정값 (이미 spec 박힘)
  w5 (wrong_dir)      = 15.0    ← §12.4 고정값 (이미 spec 박힘)
```

**단위 분석 근거** (sparse reward trap 방지):

```
Episode 평균 (Medium 난이도, 학습 초기):
  length 50 step / collision 5회 / goal 도달 시:
    r = -0.1*50 - 2.0*5 + 50.0*1 + 5*1 - 15*0
      = -5 - 10 + 50 + 5
      = +40  (net positive 확보)

  goal 미도달 (timeout):
    r = -0.1*100 - 2.0*10 + 0
      = -30
  
  도달/미도달 차이 = 70, 충분한 학습 신호
```

FAILURE_LOG §6.5 placeholder 예시 (goal_bonus 100 → 500 의 sparse trap) 가 가리키는 함정 회피. w3 가 episode 누적 penalty 보다 충분히 크다.

#### 16.3.2 Step 1 autoresearch sweep 범위 (2026-06-18, L-16.3-w 해제)

```
Stage 1 sweep (Round 2 — Round 1 의 α/β best 고정 후):
  w1 ∈ {0.05, 0.1, 0.2, 0.5}      # 4 values
  w2 ∈ {1.0, 2.0, 5.0}             # 3 values
  w3 ∈ {20, 50, 100}               # 3 values
  → 4 × 3 × 3 = 36 variants

Stage 2: 살아남은 후보 ± 50% local grid
```

w4, w5 는 §12.4 고정값이므로 sweep 대상 아님.

#### 16.3.3 Step 2~6 가중치 — 상대 스케일 원칙 (Phase 1 후 후행 결정)

w6 ~ w17 의 구체값은 **각 Step 진입 직전 결정**. 다만 다음 원칙을 따른다:

```
1. Hard constraint 위반 (slope_violation, bend_radius_violation, valve_height_violation 등) 의 w:
   해당 Step reward 합산이 항상 net negative 가 되도록 큰 값 (~20)
   
2. Soft constraint (excess_bend_count, branch_total_length_excess 등) 의 w:
   length penalty (w1) 와 자릿수 비슷 (~0.5)
   
3. 새 Step 의 w 도입 시 이전 Step 의 r 합산을 크게 변경하지 않도록
   회귀 검증 통과 위함 (§12.3 회귀 임계값과 정합)
```

### 16.4 새 reward 항목 자동 검증 게이트 (4-Gate)

autoresearch가 새 reward 항목을 제안할 때 다음 4개 게이트를 모두 통과해야 학습 진행:

```python
def validate_new_reward_item(proposal):
    # Gate 1: 측정 가능성 (격자 좌표로 계산 가능한 수식인지)
    if not is_grid_computable(proposal.formula):
        return REJECT("Not grid-computable")

    # Gate 2: 기존 항목과 충돌 (반대 방향 작동 등)
    if conflicts_with_existing(proposal):
        return REJECT("Conflicts with existing items")

    # Gate 3: 평가 metric 정렬 (해당 step의 evaluation metric을 해치지 않음)
    if hurts_evaluation_metric(proposal):
        return REJECT("Misaligned with evaluation")

    # Gate 4: 부호 일관성 (예: "길이 짧을수록 좋다"인데 "길이 길수록 보너스" → reject)
    if sign_inconsistent(proposal):
        return REJECT("Sign inconsistent")

    return ACCEPT
```

> 게이트 통과 != 효과 보장. 학습 후 평가에서 결과 나쁘면 자동 폐기.

### 16.5 변형 실행 방식

#### A. 동시 실행 원칙
**한 번에 한 인자만 변형** (ablation 원칙).
인과관계 추적을 위해 동시 다인자 변형 금지.
단, **같은 인자의 다른 값**을 동시 학습하는 것은 허용 (예: w_length=[0.5, 1.0, 1.5, 2.0] 동시).

#### B. 학습 시간 압축 (2-stage screening) — 2026-06-18 L-C2 해제

```
정상 학습 시간 (Phase 1 Step 1):  2,000,000 timestep

Stage 1 (1차 스크리닝):  250,000 timestep (정상의 1/8)
  - 변형 N개 모두 학습
  - 평가 metric 측정 → 하위 50% 제거

Stage 2 (정상 학습):       2,000,000 timestep
  - 살아남은 후보만 정상 학습
  - 최종 평가 → 1개 채택

동시 학습 가능 모델 수:  T4 GPU 1개 기준 2~3개 (안전 2, 속도 우선 3)
                          Colab Pro+ / A100 환경에서는 4~6개 가능
```

**근거**: 1/8 비율은 (a) 학습 곡선 초기 trajectory 안정화 충분 + (b) 12 variants 합리적 시간 (T4 기준 약 2시간) 균형점. 50% 제거는 successive halving 표준 비율. 70% 제거는 짧은 학습 노이즈로 좋은 variant 잘려나갈 위험.

#### C. 변형 제안 메커니즘 (Optuna TPE + Claude API)

```
1. Stage 1 (12~36 variants):
   Optuna 의 enqueue_trial() 로 grid 명시 등록 (sweep 보장)
   예: α × β = 12 variants 또는 w1 × w2 × w3 = 36 variants

2. Stage 2 (살아남은 후보):
   TPE (Tree-structured Parzen Estimator) sampler 로 local search
   예: best 주변 ± 50% 영역에서 효율적 탐색

3. Round 종료 시:
   Optuna importance analysis 로 "어떤 인자가 결과 분산을 가장 설명하는지" 정량 추출
   다음 Round 의 sweep 대상 결정 근거
```

#### D. 인자 간 interaction 처리
```
1차 sequential sweep (Round 별):
  Round 1: α, β (L-D2 PBS 가중치) — w 는 초기값 고정
  Round 2: w1, w2, w3 — α, β 는 Round 1 best 고정
  Round 3+: 인자 추가 또는 interaction grid

2차 grid search (좋은 영역 주변):
  인자 간 interaction 효과 포착
  예: α ∈ [best ± 50%] × w3 ∈ [best ± 50%]
```

### 16.6 autoresearch 종료 조건 (2026-06-18 L-C2 해제)

#### A. Round 단위 정의

```
1 Round = Stage 1 (N variants) + Stage 2 (N/2 variants) + 평가
        ≒ 1.5 ~ 3 시간 (T4 GPU, 동시 3개 가정)
```

#### B. 종료 조건 (OR)

```
1. Round 5 도달 (절대 상한)
2. 최근 2 Round 의 best 변형이 동일 (개선 정체)
3. wandb best variant 학습 곡선 plateau 명확
   기준: 마지막 200K timestep 의 success_rate 증가 < 2.0%p (부동소수점 epsilon 1e-6 허용)
4. 사용자 강제 종료
```

**근거**: Round 5 상한은 Stage 1 reward tuning 의 표준 깊이. 그 이상은 천장 (ceiling) 가능성. plateau 임계값 2%p 는 통계 노이즈 (1%p) 보다 약간 큰 값.

### 16.6.5 Optuna 통합 spec (2026-06-18 L-C2 해제)

#### A. Optuna 채택 근거

- `optuna.importance.get_param_importances()` 가 "어떤 인자가 결과 분산을 가장 설명하는가" 정량 추출
- Phase 1 천장 vs search 부족 진단의 핵심 도구
- TPE sampler 가 단순 grid 보다 효율적

#### B. 운영 spec

```python
import optuna

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(n_startup_trials=12),  # Stage 1 grid 보장
    pruner=optuna.pruners.MedianPruner(),                      # plateau 자동 차단
    storage="sqlite:///autoresearch.db",                        # 영구 보관
    study_name=f"step{N}_round{R}",
)

# Stage 1: 명시 grid enqueue
for variant in stage1_grid:
    study.enqueue_trial(variant)

# Stage 2 이후: TPE 가 알아서 sampling
study.optimize(objective, n_trials=N_total)

# Round 종료 시 importance 추출
importances = optuna.importance.get_param_importances(study)
```

### 16.6.6 wandb naming 체계 (2026-06-18 L-C2 해제)

```python
wandb.init(
    project="pipe-routing-rl",                       # 전 Step / 전 Phase 공통
    name=f"step{N}_round{R}_stage{S}_var{V:03d}",    # 예: step1_round1_stage1_var007
    config={
        "step": N,
        "round": R,
        "stage": S,
        "variant_id": V,
        "alpha": ...,          # PBS Φ_goal 가중치
        "beta": ...,           # PBS Φ_cong 가중치
        "w1": ..., "w2": ..., "w3": ...,  # baseline 가중치 (L-16.3-w)
        # 기타 sweep 대상 hyperparameter
        "git_commit": ...,     # 재현성
        "generator_version": ...,  # 시나리오 호환성
    },
    tags=[
        f"step{N}",
        f"round{R}",
        f"stage{S}",
        "autoresearch",
        "phase1" if N == 1 else "phase2plus",
    ],
)
```

parallel coordinates plot 의 의미를 위해 sweep 대상은 모두 config 에 logging. tag 는 wandb UI 필터링용.

### 16.6.7 Phase 1 autoresearch 운영 순서 (2026-06-18 L-D2 + L-16.3-w + L-C2 정합)

```
Round 1: L-D2 α × β sweep (12 variants)
  Stage 1: 250K timestep × 12 → 하위 6 제거
  Stage 2: 2M timestep × 6 → best 1 선택
  결과: best (α, β) 확정

Round 2: L-16.3-w 가중치 sweep (36 variants)
  α, β 는 Round 1 best 고정
  Stage 1: 250K timestep × 36 → 하위 18 제거 (50%)
  Stage 2: 2M timestep × 18 → best 1 선택
  결과: best (w1, w2, w3) 확정

Round 3 (optional): hyperparameter sweep
  candidate: learning_rate, batch_size, ent_coef, gamma
  Stage 1 / Stage 2 동일 패턴

Round 4 (optional): interaction grid
  예: α × w3 grid 또는 β × w1 grid

Round 5 (조건부): 추가 reward 항목 검토
  4-Gate 통과 시에만

종료: Round 5 도달 or plateau 감지 or 사용자 강제 종료
```

### 16.7 Dense Reward (Potential-Based Shaping) — 의사결정 10, L-D2 해제

> **2026-05-14 신규**: HRLP (Tan & Mu 2024) 영감. 자세한 근거는 PROGRESS.md Session 2026-05-14 의사결정 10 참조.
> **2026-06-18 L-D2 해제**: Phase 1 적용 spec 구체화. Φ 구성, 초기값, sweep 범위, 코드 강제 안전장치, wandb 진단 panel 정의.

#### 16.7.1 도입 형식

baseline reward (§16.3) 에 다음 dense term을 추가:

```
r_total(s, a, s') = r_baseline(s, a, s') + r_shape(s, s')

r_shape(s, s') = γ_PPO · Φ(s') - Φ(s)     ← Potential-Based Shaping (Ng et al. 1999)
```

`γ_PPO` 는 PPO 의 discount factor 와 **반드시 동일 값**. autoresearch sweep 시 γ 변경하면 PBS 의 γ 도 함께 변경 (`pbs_gamma_sync` constraint).

#### 16.7.2 Phase 1 Φ 구성 (L-D2 해제 결과)

```
Phase 1 (Step 1 dense reward 도입):

  Φ(s) = α · Φ_goal(s) + β · Φ_cong(s)
  
  Φ_goal(s) = -d_manhattan(s, goal) / d_initial
              범위: [-1, 0]
              d_manhattan: 7-direction discrete action 과 정합
              d_initial: episode 시작 시점의 d_manhattan(s_0, goal)
                         → 정규화로 |Φ_goal| ≤ 1 보장
  
  Φ_cong(s) = -1 / (mean_SDF_5x5x5(s) + 1.0)
              mean_SDF_5x5x5: 현재 위치 주변 5x5x5 cube 의 평균 SDF
              범위: 음수, |Φ_cong| ~ O(1)
              해석: 막힌 영역 진입 회피
  
  Φ_future(s) = Phase 1 미사용. Phase 2+ 검토.
                (정의 모호성 — mini A* / value approximation / heuristic
                 중 어느 것도 비용·신뢰도 trade-off 미해결)
```

#### 16.7.3 Terminal 처리 (PBS 조건 2 명시)

```
Φ(s) where s is terminal:

  goal_reached terminal:    Φ = 0  (자연스러움, d_goal = 0)
  collision terminal:        Φ = 0  (명시적 set, PBS 조건 2)
  timeout terminal:          Φ = 0  (명시적 set, PBS 조건 2)
  out_of_bounds terminal:    Φ = 0  (명시적 set, PBS 조건 2)
```

비-goal terminal 에서 명시적 0 set 을 누락하면 정책 보존 이론적 보장이 깨짐. **구현 시 unit test 의무**.

#### 16.7.4 PBS 안전장치 (이론적 보장 조건)

PBS는 다음 조건 하에서 **optimal policy 보존이 이론적으로 증명**됨 (Ng, Harada & Russell 1999):

```
조건 1. Φ(s)는 state만의 함수 (action 의존 금지)
조건 2. Φ(terminal_state) = 0
조건 3. r_shape = γ·Φ(s') - Φ(s) 형식 (단순 cost 차이 아님)
조건 4. (본 프로젝트 추가): γ_PBS == γ_PPO
```

위 조건을 지키면 reward hacking 위험 없이 학습 속도만 빨라짐. 4번째 조건은 본 프로젝트의 PBS-PPO 호환성을 위한 안전장치.

#### 16.7.5 코드 레벨 강제 — `PotentialBasedShaping` helper class

PBS 조건을 type level + runtime check 로 강제:

```python
from typing import Callable
import inspect
import numpy as np

class PotentialBasedShaping:
    """PBS 4대 조건 hard-enforced shaping reward wrapper."""
    
    def __init__(
        self,
        phi_fn: Callable[[np.ndarray], float],  # state-only 시그니처
        gamma_ppo: float,                       # MUST equal PPO's gamma
        terminal_phi: float = 0.0,              # PBS 조건 2
    ):
        # 조건 1 검증: phi_fn이 state만 받는지 시그니처 체크
        sig = inspect.signature(phi_fn)
        if len(sig.parameters) != 1:
            raise ValueError(
                "PBS 조건 1 위반: Φ must be state-only (single argument)"
            )
        
        self.phi_fn = phi_fn
        self.gamma = gamma_ppo
        self.terminal_phi = terminal_phi
    
    def shaping_reward(self, s, s_next, is_terminal: bool) -> float:
        """조건 2, 3, 4를 만족하는 r_shape 계산."""
        # 조건 2: terminal Φ = 0
        phi_next = self.terminal_phi if is_terminal else self.phi_fn(s_next)
        phi_curr = self.phi_fn(s)
        # 조건 3, 4: γ_ppo·Φ(s') - Φ(s)
        return self.gamma * phi_next - phi_curr
```

본 class 를 거치지 않은 `r_shape` 직접 작성은 SKILL §3.5 규칙 위반.

#### 16.7.6 초기값과 autoresearch sweep 범위

```
초기값 (Phase 1 1차 학습):
  α = 1.0   (Φ_goal 가중치, 핵심)
  β = 0.1   (Φ_cong 가중치, 보수적 시작)

autoresearch Stage 1 sweep:
  α ∈ {0.5, 1.0, 2.0, 5.0}
  β ∈ {0.0, 0.1, 0.3}
  → 4 × 3 = 12 variants

autoresearch Stage 2 (살아남은 후보):
  α, β ± 50% local grid
  (예: Stage 1 best 가 α=2.0, β=0.1 이면
       Stage 2: α ∈ {1.0, 1.5, 2.0, 2.5, 3.0}, β ∈ {0.05, 0.075, 0.1, 0.125, 0.15})
```

**12 variants 의 wall-clock 시간 검토 필요**: L-C2 해제 시 동시 학습 가능 모델 수 (Colab GPU 자원) 가 결정되면 wall-clock 산출 가능. 12 variants 가 과도하면 Stage 1 을 α 단독 sweep (4 variants) → β 별도 sweep (3 variants) 의 sequential 로 분리 가능.

#### 16.7.7 autoresearch 변형 인자 규칙 (SKILL §4.1 연계)

```python
# Phase 1 autoresearch에서 자유 변형 가능 (PBS 4-Gate 통과 조건):
ALLOWED_FREE_PBS = [
    'alpha',          # Φ_goal 가중치
    'beta',           # Φ_cong 가중치
    'phi_goal_norm',  # d_initial 정규화 방식 (manhattan / euclidean / sdf-aware)
    'phi_cong_window_size',  # SDF window (3x3x3 / 5x5x5 / 7x7x7)
]

# 변경 절대 금지:
FORBIDDEN_PBS = [
    'pbs_form',           # γΦ(s') - Φ(s) 형식 불변
    'phi_state_only',     # state-only 시그니처 불변
    'terminal_phi_zero',  # Φ(terminal) = 0 불변
    'gamma_sync',         # γ_PBS == γ_PPO 불변
]

# 조건부 허용:
CONDITIONAL_PBS = [
    'phi_future_addition',  # Phase 2+ Φ_future 도입 (별도 4-Gate + PBS 안전조건 통과)
]
```

#### 16.7.8 wandb 진단 panel (의사결정 9 연계)

dense reward 효과 진단을 위해 wandb 에 다음 metric 을 분리 logging:

```
r_baseline_mean         : baseline reward 의 step 평균
r_shape_mean            : shaping reward 의 step 평균
r_shape_to_baseline_ratio : |r_shape_mean| / |r_baseline_mean|
                          → 정상 범위: 0.1 ~ 1.0
                          → > 2.0: shape 가 baseline 압도 (정책 왜곡 위험)
                          → < 0.05: shape 가 학습 신호로 미작동

phi_goal_at_step0       : episode 시작 시점 Φ_goal 값
phi_goal_at_terminal    : episode 종료 시점 Φ_goal 값 (goal_reached 한정)
phi_goal_trajectory     : Φ_goal 의 episode 진행 곡선 (평균)

phi_cong_distribution   : Φ_cong 의 histogram (학습 진행에 따른 shift 추적)

action_value_phi_corr   : Q-value gradient 와 Φ gradient 의 코사인 유사도
                          → ≈ 0: dense reward 가 학습 신호로 작동 안 함 (조정 필요)
                          → > 0.5: dense reward 가 정책 방향 일치
```

**진단 우선순위**:
1. `r_shape_to_baseline_ratio` — 첫 진단. 자릿수가 안 맞으면 다른 metric 의미 없음
2. `action_value_phi_corr` — dense reward 가 학습에 실제 기여하는지
3. `phi_goal_trajectory` — Φ 가 episode 진행에 따라 정상 변화하는지

세 metric 이 모두 정상인데 학습 plateau 라면 **천장 (ceiling)** 일 가능성. 그 외에는 search 부족 또는 Φ 설계 결함.

### 16.8 wandb 통합 — 의사결정 9

> **2026-05-14 신규**: autoresearch 가시성 확보. 자세한 근거는 PROGRESS.md Session 2026-05-14 의사결정 9 참조.

#### 도입 의무

모든 autoresearch 학습 run은 wandb 로그를 남긴다. sb3-contrib MaskablePPO + wandb native callback 통합.

```python
from wandb.integration.sb3 import WandbCallback

run = wandb.init(
    project="pipe-routing-rl",
    name=f"step{N}_variant_{variant_id}",
    config={...전체 config...},
    tags=[f"step{N}", "autoresearch"]
)

model.learn(
    total_timesteps=TIMESTEPS,
    callback=WandbCallback(verbose=2)
)
```

#### 활용 방식

1. **개별 학습 추적**: learning curve, episode reward, value loss 등 자동 기록
2. **다중 실험 비교**: Parallel Coordinates Plot으로 변형 간 패턴 즉시 인식
3. **천장 진단**: best variant 결과의 시간순 진화 확인 → 천장 vs 추가 search 여지 판단
4. **Colab 세션 끊김 대비**: 클라우드 저장 → 세션 끊겨도 결과 보존

#### Optuna importance analysis 보조 (선택)

autoresearch가 Optuna 기반 운영 시 (L-C2 결정 시점) 다음을 추가 활용:

```python
import optuna
importances = optuna.importance.get_param_importances(study)
# {'learning_rate': 0.42, 'w_length': 0.28, ...}
```

→ "어떤 인자가 결과에 가장 영향" 정량적 추출. wandb parallel coordinates 와 결합 시 "어떤 인자의 어떤 영역이 좋은지" 시각적 + 정량적 진단 가능.

---

