"""회귀 감시 callback 검증 — CLAUDE.md §12.3.

검증 범위:
  - Phase 1 (Step 1): regression_envs={} → 완전 no-op
  - EscalationController 상태 머신 (Level 0~3)
  - BWT / Forgetting 계산 정합 (CLAUDE.md §12.3.3)
  - Stage 1 / Stage 2 임계값 분리 (CLAUDE.md §11.0.7)
  - evaluate_step_policy() 인터페이스 (sb3 비의존 callable policy)
  - RegressionCallback 구성 (sb3 BaseCallback 상속 확인)
"""

from __future__ import annotations

import numpy as np
import pytest

from training.regression_callback import (
    EscalationController,
    EscalationLevel,
    EscalationThresholds,
    RegressionCallback,
    StepMetrics,
    evaluate_step_policy,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helper fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def thresholds_s2() -> EscalationThresholds:
    return EscalationThresholds.stage2()


@pytest.fixture
def thresholds_s1() -> EscalationThresholds:
    return EscalationThresholds.stage1()


def _make_metrics(
    step_id: int = 1,
    success_rate: float = 0.90,
    layer1_pass_rate: float = 1.0,
) -> StepMetrics:
    return StepMetrics(
        step_id=step_id,
        success_rate=success_rate,
        mean_episode_length=50.0,
        layer1_pass_rate=layer1_pass_rate,
        n_episodes=150,
        timestamp_steps=500_000,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Phase 1 no-op: regression_envs={} → callback 비활성
# ─────────────────────────────────────────────────────────────────────────────


def test_phase1_callback_inactive_on_rollout_end() -> None:
    """Phase 1: regression_envs={} → _on_rollout_end() no-op, stop=False."""
    controller = EscalationController(thresholds=EscalationThresholds.stage2())
    cb = RegressionCallback(
        regression_step_envs={},
        controller=controller,
        eval_every_n_rollouts=25,
    )
    cb.num_timesteps = 50_000
    # 25 rollout 실행 시뮬레이션
    for _ in range(25):
        cb._on_rollout_end()

    assert not controller.stop_training, "Phase 1 에서 stop_training 이 True 가 됨"
    assert cb._rollout_count == 0, "Phase 1 에서 rollout_count 가 증가함"


def test_phase1_on_step_returns_true() -> None:
    """Phase 1: _on_step() 은 항상 True (학습 계속)."""
    controller = EscalationController(thresholds=EscalationThresholds.stage2())
    cb = RegressionCallback(regression_step_envs={}, controller=controller)
    assert cb._on_step() is True


# ─────────────────────────────────────────────────────────────────────────────
# 2. EscalationController 초기 상태
# ─────────────────────────────────────────────────────────────────────────────


def test_escalation_initial_state(thresholds_s2: EscalationThresholds) -> None:
    """초기 상태: consecutive_violations=0, stop_training=False."""
    ctrl = EscalationController(thresholds=thresholds_s2)
    assert ctrl.consecutive_violations == 0
    assert not ctrl.stop_training


def test_no_violation_resets_consecutive(thresholds_s2: EscalationThresholds) -> None:
    """위반 없는 update 는 consecutive_violations 를 0 으로 리셋."""
    ctrl = EscalationController(
        thresholds=thresholds_s2,
        baselines={1: 0.90},
    )
    ctrl.consecutive_violations = 2  # 이전에 위반이 있었다 가정

    good_metrics = [_make_metrics(step_id=1, success_rate=0.90)]
    level, reasons = ctrl.update(good_metrics)

    assert level == EscalationLevel.OK
    assert ctrl.consecutive_violations == 0
    assert not reasons


# ─────────────────────────────────────────────────────────────────────────────
# 3. Level 1/2/3 escalation (soft violation 연속)
# ─────────────────────────────────────────────────────────────────────────────


def test_level1_on_first_soft_violation(thresholds_s2: EscalationThresholds) -> None:
    """1번째 soft 위반 → Level 1, consecutive=1, stop=False."""
    ctrl = EscalationController(
        thresholds=thresholds_s2,
        baselines={1: 0.90},
    )
    # success_rate 0.90 → 0.80: drop=0.10 > 0.05 threshold
    bad = [_make_metrics(step_id=1, success_rate=0.80)]
    level, reasons = ctrl.update(bad)

    assert level == EscalationLevel.LEVEL1
    assert ctrl.consecutive_violations == 1
    assert not ctrl.stop_training
    assert reasons


def test_level2_on_second_consecutive_violation(thresholds_s2: EscalationThresholds) -> None:
    """2번째 연속 soft 위반 → Level 2, consecutive=2, stop=False."""
    ctrl = EscalationController(
        thresholds=thresholds_s2,
        baselines={1: 0.90},
    )
    bad = [_make_metrics(step_id=1, success_rate=0.80)]
    ctrl.update(bad)  # Level 1
    level, _ = ctrl.update(bad)  # Level 2

    assert level == EscalationLevel.LEVEL2
    assert ctrl.consecutive_violations == 2
    assert not ctrl.stop_training


def test_level3_on_third_consecutive_violation(thresholds_s2: EscalationThresholds) -> None:
    """3번째 연속 soft 위반 → Level 3, stop_training=True."""
    ctrl = EscalationController(
        thresholds=thresholds_s2,
        baselines={1: 0.90},
    )
    bad = [_make_metrics(step_id=1, success_rate=0.80)]
    ctrl.update(bad)
    ctrl.update(bad)
    level, _ = ctrl.update(bad)  # Level 3

    assert level == EscalationLevel.LEVEL3
    assert ctrl.stop_training


# ─────────────────────────────────────────────────────────────────────────────
# 4. Level 0: Layer 1 hard violation (즉시 중단)
# ─────────────────────────────────────────────────────────────────────────────


def test_level0_layer1_hard_violation(thresholds_s2: EscalationThresholds) -> None:
    """Layer 1 pass_rate < 95% → Level 0, 즉시 stop_training=True."""
    ctrl = EscalationController(thresholds=thresholds_s2, baselines={3: 0.90})
    hard_fail = [_make_metrics(step_id=3, success_rate=0.85, layer1_pass_rate=0.94)]
    level, reasons = ctrl.update(hard_fail)

    assert level == EscalationLevel.LEVEL0
    assert ctrl.stop_training
    assert reasons


def test_level0_takes_priority_over_soft(thresholds_s2: EscalationThresholds) -> None:
    """Layer 1 hard + soft 동시 위반 → Level 0 우선 (soft Level check skip)."""
    ctrl = EscalationController(thresholds=thresholds_s2, baselines={3: 0.90})
    combined = [_make_metrics(step_id=3, success_rate=0.70, layer1_pass_rate=0.80)]
    level, _ = ctrl.update(combined)
    assert level == EscalationLevel.LEVEL0


# ─────────────────────────────────────────────────────────────────────────────
# 5. BWT / Forgetting 계산 정합 (CLAUDE.md §12.3.3)
# ─────────────────────────────────────────────────────────────────────────────


def test_bwt_calculation() -> None:
    """BWT_k = current_SR - baseline_SR."""
    ctrl = EscalationController(
        thresholds=EscalationThresholds.stage2(),
        baselines={1: 0.85},
    )
    m = _make_metrics(step_id=1, success_rate=0.80)
    ctrl.update([m])
    derived = ctrl.compute_derived_metrics([m])

    expected_bwt = 0.80 - 0.85
    assert derived[1]["bwt"] == pytest.approx(expected_bwt, abs=1e-9)


def test_forgetting_calculation() -> None:
    """F_k = max_seen_SR - current_SR."""
    ctrl = EscalationController(
        thresholds=EscalationThresholds.stage2(),
        baselines={1: 0.80},
        best_seen={1: 0.95},  # 이전에 0.95 를 본 적 있음
    )
    m = _make_metrics(step_id=1, success_rate=0.80)
    derived = ctrl.compute_derived_metrics([m])

    expected_forgetting = 0.95 - 0.80
    assert derived[1]["forgetting"] == pytest.approx(expected_forgetting, abs=1e-9)


def test_forgetting_zero_when_no_regression() -> None:
    """지금이 최고 성능이면 Forgetting = 0."""
    ctrl = EscalationController(
        thresholds=EscalationThresholds.stage2(),
        baselines={1: 0.80},
    )
    m = _make_metrics(step_id=1, success_rate=0.90)
    ctrl.update([m])  # best_seen[1] = 0.90
    derived = ctrl.compute_derived_metrics([m])
    assert derived[1]["forgetting"] == pytest.approx(0.0, abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Stage 1 vs Stage 2 임계값 분리 (CLAUDE.md §11.0.7)
# ─────────────────────────────────────────────────────────────────────────────


def test_stage1_thresholds_looser_than_stage2() -> None:
    """Stage 1 임계값이 Stage 2 보다 느슨하다 (drop_max 더 큰 값 허용)."""
    s1 = EscalationThresholds.stage1()
    s2 = EscalationThresholds.stage2()
    assert s1.success_rate_drop_max > s2.success_rate_drop_max
    assert s1.forgetting_max > s2.forgetting_max
    assert s1.layer1_pass_rate_min < s2.layer1_pass_rate_min


def test_stage2_threshold_values() -> None:
    """Stage 2 임계값이 spec 값과 일치한다. CLAUDE.md §11.0.7."""
    s2 = EscalationThresholds.stage2()
    assert s2.success_rate_drop_max == pytest.approx(0.05)
    assert s2.layer1_pass_rate_min == pytest.approx(1.00)
    assert s2.bwt_min == pytest.approx(-0.05)
    assert s2.forgetting_max == pytest.approx(0.10)


def test_stage1_threshold_values() -> None:
    """Stage 1 임계값이 spec 값과 일치한다. CLAUDE.md §11.0.7."""
    s1 = EscalationThresholds.stage1()
    assert s1.success_rate_drop_max == pytest.approx(0.10)
    assert s1.layer1_pass_rate_min == pytest.approx(0.95)
    assert s1.bwt_min == pytest.approx(-0.10)
    assert s1.forgetting_max == pytest.approx(0.20)


# ─────────────────────────────────────────────────────────────────────────────
# 7. evaluate_step_policy(): callable policy 인터페이스
# ─────────────────────────────────────────────────────────────────────────────


def test_evaluate_step_policy_with_goal_policy() -> None:
    """goal_reached policy → success_rate=1.0."""
    from envs.step1_env import Step1Env

    env = Step1Env(mode="regression", difficulty="easy", seed=0, alpha=0.0, beta=0.0)

    def goal_seeking_policy(obs: np.ndarray, action_mask: np.ndarray) -> int:
        """action_mask 에서 첫 valid action 선택 (random walk 대용)."""
        valid = np.where(action_mask)[0]
        return int(valid[0])

    metrics = evaluate_step_policy(
        policy_fn=goal_seeking_policy,
        env=env,
        step_id=1,
        n_episodes=10,
        timestamp_steps=0,
    )

    assert isinstance(metrics, StepMetrics)
    assert metrics.step_id == 1
    assert metrics.n_episodes == 10
    assert 0.0 <= metrics.success_rate <= 1.0
    assert np.isfinite(metrics.mean_episode_length)


def test_evaluate_step_policy_returns_step_metrics() -> None:
    """evaluate_step_policy() 가 StepMetrics 를 반환한다."""
    from envs.step1_env import Step1Env

    env = Step1Env(mode="regression", difficulty="easy", seed=0, alpha=0.0, beta=0.0)

    def random_policy(obs: np.ndarray, action_mask: np.ndarray) -> int:
        rng = np.random.default_rng(42)
        valid = np.where(action_mask)[0]
        return int(rng.choice(valid))

    metrics = evaluate_step_policy(
        policy_fn=random_policy,
        env=env,
        step_id=1,
        n_episodes=5,
    )

    assert isinstance(metrics, StepMetrics)
    assert metrics.step_id == 1


# ─────────────────────────────────────────────────────────────────────────────
# 8. RegressionCallback 구성 검증 (sb3 BaseCallback 상속)
# ─────────────────────────────────────────────────────────────────────────────


def test_regression_callback_is_sb3_callback() -> None:
    """RegressionCallback 은 sb3 BaseCallback 을 상속한다."""
    from stable_baselines3.common.callbacks import BaseCallback

    controller = EscalationController(thresholds=EscalationThresholds.stage2())
    cb = RegressionCallback(regression_step_envs={}, controller=controller)
    assert isinstance(cb, BaseCallback)


def test_regression_callback_attributes() -> None:
    """RegressionCallback 필수 속성 확인."""
    controller = EscalationController(thresholds=EscalationThresholds.stage2())
    cb = RegressionCallback(
        regression_step_envs={},
        controller=controller,
        eval_every_n_rollouts=25,
        n_eval_episodes=150,
    )
    assert cb.eval_every_n_rollouts == 25
    assert cb.n_eval_episodes == 150
    assert cb._rollout_count == 0


# ─────────────────────────────────────────────────────────────────────────────
# 9. StepMetrics.to_dict() JSON-serializable
# ─────────────────────────────────────────────────────────────────────────────


def test_step_metrics_to_dict_json_serializable() -> None:
    """StepMetrics.to_dict() 결과가 JSON 직렬화 가능하다."""
    import json

    m = _make_metrics()
    d = m.to_dict()
    serialized = json.dumps(d)  # 예외 없으면 통과
    assert "success_rate" in serialized
    assert "step_id" in serialized
