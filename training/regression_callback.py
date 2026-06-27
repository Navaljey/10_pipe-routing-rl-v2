"""회귀 감시 callback — CLAUDE.md §12.3.

설계:
  - Phase 1 (Step 1 단독): regression_envs={} → 즉시 no-op. 오버헤드 없음.
  - Step 2+: 과거 Step 회귀 시나리오 150개 평가 → BWT / Forgetting / per-step metric.
  - 4-level escalation trigger (CLAUDE.md §12.3.2).
  - 매 25 rollout (n_steps=2048 기준 ≈ 50K timestep) 마다 실행 (CLAUDE.md §12.3.1).
  - wandb logging (CLAUDE.md §16.6.6 naming 체계 연동).

테스트 가능성:
  - StepMetrics, EscalationThresholds, EscalationController 는 sb3 비의존 순수 클래스.
  - evaluate_step_policy() 는 callable policy_fn 을 받아 sb3 비의존적으로 테스트 가능.
  - RegressionCallback(BaseCallback) 은 sb3 wrapper (실 학습 시 사용).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. StepMetrics: 특정 time-point 에서 하나의 과거 Step 평가 결과
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class StepMetrics:
    """과거 Step K 에 대한 time t 의 평가 결과. CLAUDE.md §12.3.3."""

    step_id: int
    success_rate: float          # [0, 1]
    mean_episode_length: float   # 에피소드 평균 길이 (step 수)
    layer1_pass_rate: float = 1.0  # Step 3+ 에서만 의미 있음; 그 이전은 1.0
    n_episodes: int = 0
    timestamp_steps: int = 0     # 측정 시점 (학습 총 step)

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "success_rate": float(self.success_rate),
            "mean_episode_length": float(self.mean_episode_length),
            "layer1_pass_rate": float(self.layer1_pass_rate),
            "n_episodes": int(self.n_episodes),
            "timestamp_steps": int(self.timestamp_steps),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. EscalationThresholds: Stage 1 / Stage 2 임계값 (CLAUDE.md §11.0.7)
# ─────────────────────────────────────────────────────────────────────────────

_STAGE1_DEFAULTS = dict(
    success_rate_drop_max=0.10,    # -10%p
    length_ratio_increase_max=0.20,  # +20%
    layer1_pass_rate_min=0.95,     # 95% 이상
    bwt_min=-0.10,
    forgetting_max=0.20,
    layer1_hard_violation_max=0.05,  # Level 0: 5%+ 즉시 중단
)
_STAGE2_DEFAULTS = dict(
    success_rate_drop_max=0.05,    # -5%p (Stage 2 더 엄격)
    length_ratio_increase_max=0.10,  # +10%
    layer1_pass_rate_min=1.00,     # 100% 절대
    bwt_min=-0.05,
    forgetting_max=0.10,
    layer1_hard_violation_max=0.05,  # Level 0: 동일
)


@dataclass
class EscalationThresholds:
    """CLAUDE.md §11.0.7 Stage 1/2 임계값. stage='stage1' | 'stage2'."""

    stage: str = "stage2"
    success_rate_drop_max: float = _STAGE2_DEFAULTS["success_rate_drop_max"]
    length_ratio_increase_max: float = _STAGE2_DEFAULTS["length_ratio_increase_max"]
    layer1_pass_rate_min: float = _STAGE2_DEFAULTS["layer1_pass_rate_min"]
    bwt_min: float = _STAGE2_DEFAULTS["bwt_min"]
    forgetting_max: float = _STAGE2_DEFAULTS["forgetting_max"]
    layer1_hard_violation_max: float = _STAGE2_DEFAULTS["layer1_hard_violation_max"]

    @classmethod
    def stage1(cls) -> EscalationThresholds:
        return cls(stage="stage1", **_STAGE1_DEFAULTS)  # type: ignore[arg-type]

    @classmethod
    def stage2(cls) -> EscalationThresholds:
        return cls(stage="stage2", **_STAGE2_DEFAULTS)  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# 3. EscalationLevel enum
# ─────────────────────────────────────────────────────────────────────────────


class EscalationLevel(IntEnum):
    OK = -1          # 위반 없음
    LEVEL0 = 0       # 즉시 중단 (Layer 1 hard ≥5% 위반)
    LEVEL1 = 1       # 경고 (1번째 위반)
    LEVEL2 = 2       # 자동 조정 (2번째 연속 위반)
    LEVEL3 = 3       # 학습 중단 (3번째 연속 위반)


# ─────────────────────────────────────────────────────────────────────────────
# 4. EscalationController: 순수 상태 머신 (sb3 비의존)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EscalationController:
    """CLAUDE.md §12.3.2 4-level escalation 상태 머신.

    baselines[step_id]  : Step K 완료 시점의 success_rate (BWT 기준선)
    best_seen[step_id]  : 지금까지 본 최대 success_rate (Forgetting 계산용)
    consecutive_violations : 연속 위반 횟수 (Level 1/2/3 판정)
    """

    thresholds: EscalationThresholds
    baselines: dict[int, float] = field(default_factory=dict)   # step_id -> baseline SR
    best_seen: dict[int, float] = field(default_factory=dict)   # step_id -> max SR seen

    consecutive_violations: int = 0
    _stop_training: bool = field(default=False, init=False)

    @property
    def stop_training(self) -> bool:
        return self._stop_training

    def update(
        self,
        current_metrics: list[StepMetrics],
    ) -> tuple[EscalationLevel, list[str]]:
        """현재 평가 결과로 escalation level 결정 및 상태 업데이트.

        Returns
        -------
        (level, reasons) : level < 0 은 OK, 0~3 은 각 level.
        """
        reasons: list[str] = []

        # best_seen 갱신
        for m in current_metrics:
            prev_best = self.best_seen.get(m.step_id, 0.0)
            self.best_seen[m.step_id] = max(prev_best, m.success_rate)

        # Level 0 검사 (Layer 1 hard violation — 즉시 중단)
        for m in current_metrics:
            layer1_violation_rate = 1.0 - m.layer1_pass_rate
            if layer1_violation_rate >= self.thresholds.layer1_hard_violation_max:
                reason = (
                    f"Step {m.step_id}: Layer 1 hard violation rate "
                    f"{layer1_violation_rate:.1%} >= "
                    f"{self.thresholds.layer1_hard_violation_max:.1%} (Level 0)"
                )
                reasons.append(reason)
                logger.error("[Level 0] %s", reason)
                self._stop_training = True
                return EscalationLevel.LEVEL0, reasons

        # 소프트 위반 검사
        soft_violated = False
        for m in current_metrics:
            sid = m.step_id
            baseline_sr = self.baselines.get(sid, m.success_rate)  # baseline 없으면 현재값

            # success_rate 절대 손실
            sr_drop = baseline_sr - m.success_rate
            if sr_drop > self.thresholds.success_rate_drop_max:
                reasons.append(
                    f"Step {sid}: success_rate drop {sr_drop:.3f} "
                    f"> {self.thresholds.success_rate_drop_max:.3f}"
                )
                soft_violated = True

            # BWT
            bwt = m.success_rate - baseline_sr
            if bwt < self.thresholds.bwt_min:
                reasons.append(
                    f"Step {sid}: BWT {bwt:.3f} < {self.thresholds.bwt_min:.3f}"
                )
                soft_violated = True

            # Forgetting
            best = self.best_seen.get(sid, m.success_rate)
            forgetting = best - m.success_rate
            if forgetting > self.thresholds.forgetting_max:
                reasons.append(
                    f"Step {sid}: Forgetting {forgetting:.3f} "
                    f"> {self.thresholds.forgetting_max:.3f}"
                )
                soft_violated = True

            # Layer 1 pass rate (소프트 체크; hard는 위에서 처리)
            if m.layer1_pass_rate < self.thresholds.layer1_pass_rate_min:
                reasons.append(
                    f"Step {sid}: layer1_pass_rate {m.layer1_pass_rate:.3f} "
                    f"< {self.thresholds.layer1_pass_rate_min:.3f}"
                )
                soft_violated = True

        if not soft_violated:
            self.consecutive_violations = 0
            return EscalationLevel.OK, []

        self.consecutive_violations += 1
        level = EscalationLevel(min(self.consecutive_violations, 3))

        if level >= EscalationLevel.LEVEL3:
            self._stop_training = True
            logger.error("[Level 3] 3회 연속 위반 → 학습 중단. reasons=%s", reasons)
        elif level == EscalationLevel.LEVEL2:
            logger.warning("[Level 2] 2회 연속 위반 → auto-adjust. reasons=%s", reasons)
        else:
            logger.warning("[Level 1] 1회 위반 → 경고. reasons=%s", reasons)

        return level, reasons

    def compute_derived_metrics(
        self, current_metrics: list[StepMetrics]
    ) -> dict[int, dict[str, float]]:
        """BWT / Forgetting 계산 (wandb logging 용). CLAUDE.md §12.3.3."""
        result: dict[int, dict[str, float]] = {}
        for m in current_metrics:
            sid = m.step_id
            baseline_sr = self.baselines.get(sid, m.success_rate)
            best_sr = self.best_seen.get(sid, m.success_rate)
            result[sid] = {
                "success_rate": m.success_rate,
                "bwt": m.success_rate - baseline_sr,
                "forgetting": best_sr - m.success_rate,
                "layer1_pass_rate": m.layer1_pass_rate,
            }
        return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. evaluate_step_policy: 순수 평가 함수 (sb3 비의존 인터페이스)
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_step_policy(
    policy_fn: Callable[[np.ndarray, np.ndarray], int],
    env,
    step_id: int,
    n_episodes: int = 150,
    timestamp_steps: int = 0,
) -> StepMetrics:
    """n_episodes 동안 policy_fn 으로 env 를 rollout → StepMetrics 반환.

    Parameters
    ----------
    policy_fn :
        (obs, action_mask) -> action_idx  형식의 callable.
        MaskablePPO 의 predict() 를 감싸거나 random policy 로 대체 가능.
    env :
        gymnasium Env (Step K regression 환경). action_masks() 메서드 필요.
    step_id :
        평가 대상 Step 번호 (BWT / Forgetting tracking 용).
    n_episodes :
        평가 에피소드 수. CLAUDE.md §11.0.1 regression = 150개.
    timestamp_steps :
        현재 학습 step (wandb logging 용).
    """
    successes: list[int] = []
    lengths: list[int] = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        terminated_info: dict = {}

        while not done:
            action_mask = env.action_masks()
            action = policy_fn(obs, action_mask)
            obs, _, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            steps += 1
            if done:
                terminated_info = info

        successes.append(1 if terminated_info.get("termination") == "goal_reached" else 0)
        lengths.append(steps)

    return StepMetrics(
        step_id=step_id,
        success_rate=float(np.mean(successes)),
        mean_episode_length=float(np.mean(lengths)),
        layer1_pass_rate=1.0,  # Step 1: Layer 1 N/A
        n_episodes=n_episodes,
        timestamp_steps=timestamp_steps,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. RegressionCallback: sb3 BaseCallback wrapper
# ─────────────────────────────────────────────────────────────────────────────


class RegressionCallback(BaseCallback):
    """sb3 회귀 감시 callback. CLAUDE.md §12.3.

    Phase 1 (Step 1 단독): regression_step_envs={} → 즉시 no-op.
    Step 2+: 매 eval_every_n_rollouts rollout 마다 과거 Step 평가 실행.

    Parameters
    ----------
    regression_step_envs :
        {step_id: gymnasium_env} dict.
        Phase 1 (Step 1): {} (빈 dict) 를 전달 → callback 비활성.
    controller :
        EscalationController 인스턴스. baselines 는 호출자가 사전 설정.
    eval_every_n_rollouts :
        평가 주기 (rollout 수). CLAUDE.md §12.3.1: 25 rollout ≈ 50K step.
    n_eval_episodes :
        평가 에피소드 수. CLAUDE.md §11.0.1: 150개.
    wandb_run :
        wandb.run 객체 (None 이면 wandb logging 생략).
    verbose :
        sb3 BaseCallback verbose.
    """

    def __init__(
        self,
        regression_step_envs: dict[int, object],
        controller: EscalationController,
        eval_every_n_rollouts: int = 25,
        n_eval_episodes: int = 150,
        wandb_run=None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self._regression_envs = regression_step_envs  # {step_id: env}
        self.controller = controller
        self.eval_every_n_rollouts = eval_every_n_rollouts
        self.n_eval_episodes = n_eval_episodes
        self._wandb_run = wandb_run

        self._rollout_count: int = 0
        self._eval_history: list[dict] = []  # logged eval results

    # ── sb3 interface ─────────────────────────────────────────────────────────

    def _on_step(self) -> bool:
        """매 step 호출. Level 0 / Level 3 stop flag 만 체크."""
        return not self.controller.stop_training

    def _on_rollout_end(self) -> None:
        """rollout 완료 후 호출. 25 rollout 마다 평가 실행."""
        if not self._regression_envs:
            return  # Phase 1 (Step 1): no-op

        self._rollout_count += 1
        if self._rollout_count % self.eval_every_n_rollouts != 0:
            return

        self._run_evaluation()

    # ── 평가 실행 ─────────────────────────────────────────────────────────────

    def _run_evaluation(self) -> None:
        """regression 평가 실행 → escalation 체크 → wandb logging."""
        current_metrics: list[StepMetrics] = []

        for step_id, env in self._regression_envs.items():
            metrics = evaluate_step_policy(
                policy_fn=self._policy_fn,
                env=env,
                step_id=step_id,
                n_episodes=self.n_eval_episodes,
                timestamp_steps=self.num_timesteps,
            )
            current_metrics.append(metrics)

        level, reasons = self.controller.update(current_metrics)
        derived = self.controller.compute_derived_metrics(current_metrics)

        self._log_to_wandb(level, current_metrics, derived)
        self._eval_history.append({
            "rollout": self._rollout_count,
            "timesteps": self.num_timesteps,
            "level": int(level),
            "reasons": reasons,
            "metrics": {m.step_id: m.to_dict() for m in current_metrics},
            "derived": derived,
        })

        if self.verbose >= 1:
            logger.info(
                "[RegressionCallback] rollout=%d steps=%d level=%d reasons=%s",
                self._rollout_count,
                self.num_timesteps,
                int(level),
                reasons[:3],
            )

    def _policy_fn(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        """sb3 모델로부터 action 추출. MaskablePPO.predict() 래핑."""
        action, _ = self.model.predict(  # type: ignore[union-attr]
            obs,
            action_masks=action_mask,
            deterministic=True,
        )
        return int(action)

    def _log_to_wandb(
        self,
        level: EscalationLevel,
        metrics: list[StepMetrics],
        derived: dict[int, dict[str, float]],
    ) -> None:
        if self._wandb_run is None:
            return
        log_dict: dict[str, float] = {"regression/escalation_level": float(level)}
        for sid, d in derived.items():
            prefix = f"regression/step{sid}"
            for k, v in d.items():
                log_dict[f"{prefix}/{k}"] = float(v)
        self._wandb_run.log(log_dict, step=self.num_timesteps)
