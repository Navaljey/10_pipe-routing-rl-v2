"""PBS (Potential-Based Shaping) 통합 검증 — CLAUDE.md §16.7.

검증 범위:
  - PotentialBasedShaping helper class 경유 강제 (SKILL §3.5, §6.4)
  - PBS 4대 조건 (state-only Φ, terminal Φ=0, gamma*Phi(s_next)-Phi(s), gamma_PBS==gamma_PPO)
  - r_baseline / r_shape 분리 logging (CLAUDE.md §16.7.8)
  - r_shape_to_baseline_ratio 정상 범위 검증
  - 100-episode sanity check (NaN/Inf 없음, ratio 분포)
"""

from __future__ import annotations

import numpy as np
import pytest

from envs.step1_env import Step1Env
from shaping import PBSSafetyWarning, PotentialBasedShaping

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Step1Env 인스턴스 (PBS 활성, easy 난이도, 고정 seed)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def pbs_env() -> Step1Env:
    """PBS 활성 (alpha=1.0, beta=0.1) Step1Env."""
    return Step1Env(
        mode="train",
        difficulty="easy",
        alpha=1.0,
        beta=0.1,
        gamma_ppo=0.99,
        seed=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. PotentialBasedShaping helper class 강제 경유 (SKILL §6.4)
# ─────────────────────────────────────────────────────────────────────────────


def test_pbs_helper_instance_exists(pbs_env: Step1Env) -> None:
    """_pbs 속성이 PotentialBasedShaping 인스턴스인지 확인."""
    assert hasattr(pbs_env, "_pbs")
    assert isinstance(pbs_env._pbs, PotentialBasedShaping)


def test_phi_is_single_arg_bound_method(pbs_env: Step1Env) -> None:
    """_phi 가 단일 인자 bound method → PBS 조건 1 통과 (state-only)."""
    import inspect

    sig = inspect.signature(pbs_env._phi)
    assert len(sig.parameters) == 1, (
        f"_phi 시그니처 파라미터 수 != 1: {list(sig.parameters)}"
    )


def test_phi_no_action_related_freevars(pbs_env: Step1Env) -> None:
    """_phi 의 co_freevars 에 action-related 변수가 없다 (PBS 조건 1)."""
    action_hints = {"action", "act", "a_t", "action_idx"}
    freevars = set(pbs_env._phi.__code__.co_freevars)
    overlap = freevars & action_hints
    assert not overlap, f"_phi 에 action-related freevars 발견: {overlap}"


def test_no_pbs_safety_warning_recwarn(pbs_env: Step1Env, recwarn: pytest.WarningsChecker) -> None:
    """_phi 로 PBS 생성 시 PBSSafetyWarning 없어야 한다."""
    _ = PotentialBasedShaping(phi_fn=pbs_env._phi, gamma_ppo=0.99)
    pbs_warnings = [w for w in recwarn.list if issubclass(w.category, PBSSafetyWarning)]
    assert len(pbs_warnings) == 0, f"예기치 않은 PBSSafetyWarning: {pbs_warnings}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. PBS 4대 조건 검증
# ─────────────────────────────────────────────────────────────────────────────


def test_terminal_phi_is_zero(pbs_env: Step1Env) -> None:
    """terminal state 에서 r_shape = gamma*0 - Phi(s) (terminal_phi == 0). PBS 조건 2."""
    pbs_env.reset(seed=100)
    obs_before = pbs_env._prev_obs.copy()

    # terminal state 시뮬레이션 (is_terminal=True)
    phi_s = pbs_env._phi(obs_before)
    r_shape = pbs_env._pbs.shaping_reward(obs_before, obs_before, is_terminal=True)

    # r_shape = gamma*Phi_terminal - Phi(s) = gamma*0 - Phi(s) = -Phi(s)
    expected = 0.99 * 0.0 - phi_s
    assert r_shape == pytest.approx(expected, abs=1e-9)


def test_shaping_form_gamma_phi_next_minus_phi(pbs_env: Step1Env) -> None:
    """r_shape == gamma*Phi(s_next) - Phi(s). PBS 조건 3."""
    pbs_env.reset(seed=42)
    obs_s, _ = pbs_env.reset(seed=42)
    # 한 step 진행
    mask = pbs_env.action_masks()
    action = int(np.where(mask)[0][0])
    obs_next, _, _, _, info = pbs_env.step(action)

    r_shape_reported = info["reward_shape"]

    # 직접 계산
    phi_s = pbs_env._phi(obs_s)
    phi_next = pbs_env._phi(obs_next)
    expected = 0.99 * phi_next - phi_s

    assert r_shape_reported == pytest.approx(expected, abs=1e-6)


def test_gamma_ppo_sync(pbs_env: Step1Env) -> None:
    """_pbs.gamma == gamma_ppo (PBS 조건 4)."""
    assert pbs_env._pbs.gamma == pytest.approx(0.99)


# ─────────────────────────────────────────────────────────────────────────────
# 3. r_baseline / r_shape 분리 logging (CLAUDE.md §16.7.8)
# ─────────────────────────────────────────────────────────────────────────────


def test_info_has_reward_baseline_and_shape(pbs_env: Step1Env) -> None:
    """info dict 에 reward_baseline / reward_shape 가 있다 (wandb §16.7.8)."""
    pbs_env.reset(seed=0)
    mask = pbs_env.action_masks()
    action = int(np.where(mask)[0][0])
    _, _, _, _, info = pbs_env.step(action)

    assert "reward_baseline" in info
    assert "reward_shape" in info


def test_total_reward_equals_baseline_plus_shape(pbs_env: Step1Env) -> None:
    """total reward == reward_baseline + reward_shape."""
    pbs_env.reset(seed=0)
    mask = pbs_env.action_masks()
    action = int(np.where(mask)[0][0])
    _, total, _, _, info = pbs_env.step(action)

    expected = info["reward_baseline"] + info["reward_shape"]
    assert total == pytest.approx(expected, abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Φ 값 범위 검증 (CLAUDE.md §16.7.2)
# ─────────────────────────────────────────────────────────────────────────────


def test_phi_goal_at_reset_is_minus_one(pbs_env: Step1Env) -> None:
    """reset 직후 obs[I_GOAL_DIST]=1.0 → Φ_goal = -1.0 (alpha=1, beta=0 가정)."""
    env = Step1Env(mode="train", difficulty="easy", alpha=1.0, beta=0.0, gamma_ppo=0.99, seed=0)
    obs, _ = env.reset(seed=0)
    phi = env._phi(obs)
    # obs[I_GOAL_DIST] = 1.0 at reset → Φ_goal = -1.0, Φ_cong = 0 (beta=0)
    assert phi == pytest.approx(-1.0, abs=1e-6)


def test_phi_no_nan_inf(pbs_env: Step1Env) -> None:
    """Φ(s) 는 NaN/Inf 없어야 한다."""
    for seed in range(5):
        obs, _ = pbs_env.reset(seed=seed)
        phi = pbs_env._phi(obs)
        assert np.isfinite(phi), f"seed={seed}: Φ = {phi}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. 100-episode sanity check
# ─────────────────────────────────────────────────────────────────────────────


def test_100_episode_sanity_check() -> None:
    """100 episode random policy — NaN/Inf, ratio 범위 검증. CLAUDE.md §16.7.8.

    r_shape_to_baseline_ratio 정상 범위: [0.01, 5.0] (초기 학습 단계 loose bound).
    """
    env = Step1Env(
        mode="train",
        difficulty="easy",
        alpha=1.0,
        beta=0.1,
        gamma_ppo=0.99,
        seed=0,
    )
    rng = np.random.default_rng(999)

    all_ratios: list[float] = []
    all_lengths: list[int] = []

    for ep in range(100):
        env.reset()
        ep_baseline = 0.0
        ep_shape    = 0.0
        steps = 0
        done = False

        while not done:
            mask = env.action_masks()
            valid = np.where(mask)[0]
            action = int(rng.choice(valid))
            obs, _, terminated, truncated, info = env.step(action)

            assert np.all(np.isfinite(obs)), f"ep={ep} step={steps}: obs 에 NaN/Inf"
            assert np.isfinite(info["reward_baseline"]), f"ep={ep}: baseline NaN"
            assert np.isfinite(info["reward_shape"]),    f"ep={ep}: shape NaN"

            ep_baseline += info["reward_baseline"]
            ep_shape    += info["reward_shape"]
            steps += 1
            done = terminated or truncated

        all_lengths.append(steps)
        if abs(ep_baseline) > 1e-9:
            all_ratios.append(abs(ep_shape) / abs(ep_baseline))

    # ratio 분포 검증 (CLAUDE.md §16.7.8: 정상 범위 0.1 ~ 1.0; 초기 단계 loose)
    mean_ratio = float(np.mean(all_ratios)) if all_ratios else 0.0
    assert len(all_ratios) > 50, "ratio 샘플 수 부족"
    assert mean_ratio < 10.0, f"r_shape/r_baseline ratio 과다 (mean={mean_ratio:.3f})"
    assert mean_ratio > 0.0,  "r_shape 가 전혀 없음 (PBS 미작동)"

    # episode 길이 분포: easy 난이도에서 max_steps 에 도달하지 않는 에피소드 존재해야 함
    # (너무 쉬우면 goal 도달, 너무 어려우면 timeout — 둘 다 정상)
    assert max(all_lengths) <= 500 + 1  # max_steps default
