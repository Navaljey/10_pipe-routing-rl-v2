# 전이학습 및 핸드오프 참조 문서

> CLAUDE.md에서 분리된 참조 문서. Step 간 전이학습 메커니즘, 회귀 감시 체계, Action Masking 누적표, 핸드오프 패키지 구성을 구현할 때 로드한다.
> 출발/목표 Face 6방향 제약(원 §12.4)은 전 Step 공통 불변 규칙이라 CLAUDE.md 본문에 남아 있다.

---

## 12. 모델 상속 및 전이학습 전략

> **⚠️ 2026-05-14 변경**: §12.2 (Step 6→7 전환), §12.3 (Step 7~10 Macro 신규 정의) 폐기됨. Hierarchical 폐기로 인한 부수 변경. 자세한 근거는 PROGRESS.md Session 2026-05-14 참조.

### 12.1 Step 1~10 단일 에이전트 (전 Step 동일)

```
Backbone Hidden Dim: 256 (전 Step 동일, 변경 절대 금지)
Architecture: MLP Fusion
Projection: 불필요
Observation: 150-dim Zero-padding (전 Step 동일)
Action: MaskablePPO + action_masks()
        (잠정 macro-action 도입 시 L-D3 참조)

전이학습 규칙:
  - 하위 Step 모델 로드 실패 시 학습 시작 금지
  - state_dict key naming 통일 → strict=True 로드 가능
  - normalizer.json 정의값 동일 참조
```

### 12.2 Step 간 전이학습 메커니즘 (2026-06-18 L-D1 해제, A+B 혼합 채택)

#### 12.2.1 채택된 전략 — 후보 A (순차 fine-tuning) + 후보 B (mixed curriculum) 혼합

후보 C (EWC) 와 후보 D (LoRA) 는 다음 근거로 비채택:

- **C (EWC)**: RL의 sample inefficiency 와 EWC의 Fisher 정보 추정이 잘 안 맞음. PPO의 on-policy 특성과도 충돌.
- **D (LoRA)**: 본 프로젝트 backbone hidden_dim=256 이 LoRA 효과 보기엔 작음. LoRA 는 대형 모델에서 빛남.

#### 12.2.2 학습 데이터 mix 비율 (Warm-up + 정상)

```
Step N 학습 시 (N ≥ 2):
  1. Step N-1 정책 가중치 θ_{N-1} 로 초기화 (fine-tuning, 후보 A)
  2. 학습률은 신규 학습의 50% 로 시작 (fine-tuning 표준 관행)
  3. Warm-up phase (학습 초기 100K timestep):
       50%  Step N 신규 시나리오 (procedural generator)
       40%  Step 1 ~ Step N-1 시나리오 (mixed curriculum, 후보 B)
       10%  FAILURE_LOG 누적 실패 케이스 (regression hardening)
  4. Warm-up 후 정상 phase:
       60%  Step N 신규 시나리오
       30%  Step 1 ~ Step N-1 시나리오
       10%  FAILURE_LOG 누적 실패 케이스
```

비율은 continual RL 문헌의 표준 관행을 따른 잠정값. Phase 1 후 BWT 데이터 보고 조정 가능 (autoresearch sweep 대상에 포함).

#### 12.2.3 비율 변경 시 안전 조건

- Step N-1 시나리오 비율을 30% 미만으로 낮추는 변경은 회귀 감시 임계값 (§11.0.7) 위반 위험 → autoresearch sweep 대상에서 제외 (FORBIDDEN)
- FAILURE_LOG 비율은 케이스 수에 따라 자동 조정 (등록된 케이스가 적으면 10% 채우기 위해 동일 케이스 반복 sampling)

### 12.3 Step 간 회귀 감시 체계 (2026-06-18 L-D1 해제, 의사결정 13 구체화)

> **2026-05-14 변경**: 이전의 "Macro 신규 도입 + Micro freeze + 3-stage 전환 프로토콜" 폐기.
> **2026-06-18 보강**: 회귀 감시 임계값 / 측정 주기 / 트리거 체계 구체화.

#### 12.3.1 측정 주기 (wandb logging과 정합)

```
MaskablePPO n_steps=2048 가정.
1 rollout ≒ 2048 timestep.

회귀 감시 콜백:
  - 매 25 rollout (≒ 50K timestep) 마다 실행
  - Step 1 ~ Step N-1 의 회귀 시나리오 150개 모두 평가
  - 결과를 wandb 에 학습 reward 곡선과 같은 panel 에 logging
  - Step N 총 학습량 2M timestep 가정 시 = 40번 측정 (충분한 해상도)

Stage 1 screening (autoresearch 1차):
  - 짧은 학습 (500K timestep 가정) 동안 10번 측정 (매 50K)
  - Stage 1 임계값 (§11.0.7 Stage 1 컬럼) 적용
```

#### 12.3.2 트리거 escalation (3단계 + 즉시 중단 case)

```
[Level 0 — 즉시 중단]
  조건: Layer 1 hard constraint 5% 이상 위반 (예: slope_violation, ASME compliance)
  대응:
    - 학습 즉시 중단
    - 이전 체크포인트 복귀
    - 사용자 알림 (Telegram bot)
    - spec 재검토 세션 권장

[Level 1 — 경고 (1번째 위반)]
  조건: Stage 2 임계값 1개 이상 soft 위반
  대응:
    - wandb 에 alert tag 기록
    - 학습 계속, 추가 모니터링

[Level 2 — 자동 조정 (2번째 연속 위반)]
  조건: 다음 회귀 측정에서도 동일 또는 다른 임계값 위반
  대응:
    - Mixed curriculum 비율 상향: 30% → 45%
    - 학습률 50% 감소
    - 학습 계속

[Level 3 — 학습 중단 (3번째 연속 위반)]
  조건: Level 2 조정 후에도 다음 측정에서 위반
  대응:
    - 학습 즉시 중단
    - 이전 체크포인트 복귀
    - 사용자 알림
    - Lock 해제 세션 또는 spec 재검토 필요
```

#### 12.3.3 회귀 지표 (3종 동시 기록, SKILL §3.7과 정합)

```
지표 1 — 각 과거 step 의 평가 metric 직접 측정 (Step 별 success_rate, length_ratio, ...)
지표 2 — Backward Transfer (BWT):  BWT_k = perf_k(end_of_stepN) - perf_k(end_of_stepK)
지표 3 — Forgetting measure:        F_k = max_{t ≤ now} perf_k(t) - perf_k(now)
```

임계값은 §11.0.7 표 참조.

#### 12.3.4 Step 6 → 7 전환 (다중 파이프 진입)

```
Step 6 정책 → Step 7 정책 초기화 (§12.2 전이학습 전략 적용)
Observation space: 150-dim 유지 (다른 파이프 정보를 obs[136:150] 활용)
Network: 동일 단일 에이전트 (별도 Macro 없음)
회귀 검증: §12.3.1~12.3.3 그대로 적용 (Step 1~6 회귀 시나리오 = 150개 × 6 Step = 900개 전체 평가)
```



---

> (원 §12.4 출발/목표 연결 방향 제약(Face 6방향)은 CLAUDE.md 본문 참조.)

### 12.5 [전 Step 공통] Action Masking + MaskablePPO

| Step | 마스킹 규칙 (누적) |
|------|-------------------|
| Step 1 | 충돌 + 경계 초과 + 자기경로 역주행 + 출발방향 강제 |
| Step 2 | + 벤딩 반경 위반 차단 |
| Step 3 | + 중력관 Z 상승 소프트 마스킹 |
| Step 4 | + 분기점 전용 행동 제약 |
| Step 5 | + 밸브 접근성 확보 불가 방향 차단 |
| Step 6 | + 서포트 비용 과다 경로 소프트 마스킹 |
| Step 7~10 | + 다른 파이프 영역 침입 마스킹 (구체 정의는 L-D1 후속) |

---

## 13. 핸드오프 패키지 (재정의)

> **⚠️ 2026-05-14 변경**: §13.2 (Step 6→7 전환 핸드오프 확장 4파일), §13.3 (Macro 핸드오프) 폐기. Hierarchical 폐기로 인한 부수 변경.

### 13.1 Step 핸드오프 (Step N → N+1, N ∈ [1,9])

> **★ Step 별 폴더 분리 의무 (의사결정 25, 2026-06-28):**
> 모든 핸드오프 파일은 `base_dir/step{N}/` 하위에 저장된다.
> `save_handoff(step_n=N, save_dir=base_dir)` → `base_dir/step{N}/` 자동 생성.
> Step 1 → 2 전이 시: `load_handoff(step_n=1, save_dir=base_dir)` → `base_dir/step1/` 탐색.
> **Step 간 자산 독립성 보장: Step 2 저장이 Step 1 자산을 덮어쓰지 않는다.**

```
base_dir/
  step1/                  ← Step 1 학습 완료 후 생성
    best_model.zip
    best_model_meta.json
    step1_regression_report.json
    step1_wandb_run_url.txt
    step1_env_config.json
    step1_reward_config.json
    step1_kpi_report.json
    step1_normalizer.json       (VecNormalize 저장 시)
    step1_action_space_map.json
    step1_training_log.csv
    step1_validation_set.json
  step2/                  ← Step 2 학습 완료 후 생성
    ...
```

전 Step 공통 핸드오프 (단일 에이전트 구조이므로 Step 6→7도 동일):

| 파일명 | 역할 |
|-------|------|
| `best_model.zip` | 단일 에이전트 가중치 (MaskablePPO.save) |
| `best_model_meta.json` | 아키텍처 정의 (HandoffConfig) |
| `stepN_normalizer.json` | 정규화 파라미터 (VecNormalize) |
| `stepN_kpi_report.json` | 평가 metric 결과 (Step 1~3: A* 비교, Step 4~6: Layer 1/2, Step 7~10: 다중 파이프) |
| `stepN_training_log.csv` | 학습 이력 |
| `stepN_action_space_map.json` | 행동 공간 |
| `stepN_reward_config.json` | 채택된 reward 설정 (baseline + dense PBS) |
| `stepN_env_config.json` | 환경 설정 |
| `stepN_validation_set.json` | 평가 시나리오 + benchmark 결과 |
| `stepN_regression_report.json` | Step 1~N-1 회귀 검증 결과 (Step 2 이후) |
| `stepN_wandb_run_url.txt` | 본 학습의 wandb run URL (재현/분석용) |

### 13.2 내부 텐서 규격
- **Input Dimension**: 150-dim 고정 (Step 1~10 전체)
- **Normalization**: `normalizer.json` 정의값 동일 참조
- **Weight Key Mapping**: `state_dict` key naming 통일 → `strict=True` 로드

---

