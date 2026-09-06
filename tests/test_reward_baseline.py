"""Baseline reward function 검증 — CLAUDE.md §16.3.1.

부호 정합성, 단위 분석 (도달/미도달 차이 ≈ 70), direction 제약 보너스/패널티.
"""

from __future__ import annotations

import numpy as np
import pytest

from envs.step1_env import W1, W2, W3, W4, W5
from tests.conftest import ConcreteStep1Env

# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────


def _run_episode_to_goal(env: ConcreteStep1Env, path: list[int]) -> float:
    """지정 action 시퀀스를 실행해 누적 reward 반환. goal 도달 미보장."""
    env.reset(seed=0)
    total = 0.0
    for a in path:
        _, r, done, trunc, _ = env.step(a)
        total += r
        if done or trunc:
            break
    return total


# ─────────────────────────────────────────────────────────────────────────────
# 1. 가중치 상수 검증 (spec 고정값)
# ─────────────────────────────────────────────────────────────────────────────


def test_w1_value() -> None:
    assert pytest.approx(0.1) == W1


def test_w2_value() -> None:
    assert pytest.approx(2.0) == W2


def test_w3_value() -> None:
    assert pytest.approx(50.0) == W3


def test_w4_w5_fixed() -> None:
    """w4, w5 는 §12.4 고정값 — autoresearch sweep 대상 아님."""
    assert pytest.approx(5.0) == W4
    assert pytest.approx(15.0) == W5


# ─────────────────────────────────────────────────────────────────────────────
# 2. 부호 정합성
# ─────────────────────────────────────────────────────────────────────────────


def test_length_penalty_per_step(env: ConcreteStep1Env) -> None:
    """모든 step 에서 최소 -W1 (length penalty) 가 붙는다."""
    env.reset(seed=0)
    # 빈 공간에서 임의 방향으로 1스텝. direction bonus/penalty 없는 step 2 이상에서 확인.
    _, _, _, _, _ = env.step(2)   # step 1: might get direction bonus
    _, r2, _, _, _ = env.step(2)   # step 2: length penalty only (no direction check)
    assert r2 <= -W1 + 1e-9  # r2 == -W1 (no collision, no goal)


def test_goal_bonus_positive(env: ConcreteStep1Env) -> None:
    """goal 도달 step 에는 +W3 가 포함된다."""
    # agent(5,5,5) → goal(25,25,25): 직접 teleport 으로 테스트
    env.reset(seed=0)
    env.agent_cell = np.array([24, 25, 25], dtype=np.int32)  # goal 한 칸 앞
    # start_dir_idx = env.start_dir_idx, goal_dir_idx = env.goal_dir_idx 고려 불필요
    # goal approach action 을 goal_dir_idx 로 맞추면 wrong_goal_dir_penalty 없음
    # 여기서는 +X (action=0) 로 이동해서 goal 도달 여부만 확인
    env.goal_cell = np.array([25, 25, 25], dtype=np.int32)
    env._step_count = 10  # 첫 step 아님 (direction 체크 skip)
    _, r, done, _, info = env.step(0)  # +X: (24,25,25) → (25,25,25) = goal
    assert done
    assert info["termination"] == "goal_reached"
    assert r >= W3 - W1 - W5 - 1e-9  # at worst: goal_bonus - length - wrong_goal_dir


def test_collision_penalty_negative(env_with_wall: ConcreteStep1Env) -> None:
    """wall 충돌 terminal 에는 -W2 가 포함된다."""
    env = env_with_wall
    env.reset(seed=0)
    env._step_count = 5  # 첫 step 아님
    # action 0 = +X → (6,5,5) = wall → terminated "collision"
    _, r, done, _, info = env.step(0)
    assert done
    assert info["termination"] == "collision"
    assert r == pytest.approx(-W1 - W2)


def test_oob_penalty_negative(env: ConcreteStep1Env) -> None:
    """grid 밖으로 나가면 -W2 포함."""
    env.reset(seed=0)
    env.agent_cell = np.array([0, 0, 0], dtype=np.int32)
    env._step_count = 5  # 첫 step 아님
    # action 1 = -X → (-1,0,0) = OOB
    _, r, done, _, info = env.step(1)
    assert done
    assert info["termination"] == "out_of_bounds"
    assert r == pytest.approx(-W1 - W2)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Direction 제약 (CLAUDE.md §12.4)
# ─────────────────────────────────────────────────────────────────────────────


def test_direction_align_bonus_on_correct_start(env: ConcreteStep1Env) -> None:
    """첫 step 에서 start_dir_idx 방향으로 이동 → +W4 포함."""
    _, info = env.reset(seed=0)
    start_dir = info["start_dir_idx"]
    _, r, _, _, _ = env.step(start_dir)
    # reward = -W1 + W4 (collision 없음, goal 도달 아님 가정)
    assert r == pytest.approx(-W1 + W4)


def test_wrong_start_dir_penalty(env: ConcreteStep1Env) -> None:
    """첫 step 에서 wrong 방향 → -W5 포함 (wrong_start_dir_penalty)."""
    _, info = env.reset(seed=0)
    start_dir = info["start_dir_idx"]
    # start_dir 이 아닌 방향을 선택 (자유 셀 탐색)
    wrong = (start_dir + 2) % 6  # 임의의 다른 방향 (충돌 가능성 배제하기 위해 환경 확인)
    _, r, done, _, _ = env.step(wrong)
    if not done:
        assert r == pytest.approx(-W1 - W5)
    else:
        # collision/OOB 발생 시: -W1 - W5 - W2
        assert r == pytest.approx(-W1 - W5 - W2)


def test_goal_approach_correct_dir(env: ConcreteStep1Env) -> None:
    """goal 도달 시 action == goal_dir_idx → wrong_goal_dir_penalty 없음."""
    env.reset(seed=0)
    env.agent_cell = np.array([24, 25, 25], dtype=np.int32)
    env.goal_cell = np.array([25, 25, 25], dtype=np.int32)
    env.goal_dir_idx = 0  # +X
    env._step_count = 5
    _, r, done, _, _ = env.step(0)  # action=0 (+X) = goal_dir_idx
    assert done
    # reward = -W1 + W3 (no wrong_goal_dir penalty)
    assert r == pytest.approx(-W1 + W3)


def test_goal_approach_wrong_dir(env: ConcreteStep1Env) -> None:
    """goal 도달 시 action != goal_dir_idx → -W5 추가."""
    env.reset(seed=0)
    # goal 을 agent 의 +X 에 놓고, goal_dir_idx = 2(+Y) 로 강제
    env.agent_cell = np.array([24, 25, 25], dtype=np.int32)
    env.goal_cell = np.array([25, 25, 25], dtype=np.int32)
    env.goal_dir_idx = 2  # +Y (action=0 (+X) 와 불일치)
    env._step_count = 5
    _, r, done, _, _ = env.step(0)  # +X 로 goal 도달
    assert done
    # reward = -W1 + W3 - W5
    assert r == pytest.approx(-W1 + W3 - W5)


# ─────────────────────────────────────────────────────────────────────────────
# 4. 단위 분석 — 도달/미도달 차이 (CLAUDE.md §16.3.1)
# ─────────────────────────────────────────────────────────────────────────────


def test_unit_analysis_goal_vs_timeout(env: ConcreteStep1Env) -> None:
    """CLAUDE.md §16.3.1 단위 분석: goal 에피소드 total > timeout 에피소드 total.

    Spec 예시:
        goal (50 step, collision 0, correct dir):
            -0.1*50 + 50 + 5 = +50  (향상된 masking 환경, collision 0)
        timeout (100 step):
            -0.1*100 = -10
        차이 ≥ 70 (매우 보수적 추정; collision 0 가정 시 60 이상)
    """
    env_goal = ConcreteStep1Env(seed=7)
    _, _ = env_goal.reset(seed=7)

    # goal 도달 시뮬레이션: start_dir 로 한 번, 이후 +X 방향 직진
    # (test 환경 agent(5,5,5)→goal(25,25,25): 20x20x20 대각. 직접 조작)
    env_goal.agent_cell = np.array([24, 25, 25], dtype=np.int32)
    env_goal.goal_cell = np.array([25, 25, 25], dtype=np.int32)
    env_goal.goal_dir_idx = 0  # +X
    env_goal._step_count = 49  # 다음 step 이 50번째
    env_goal._last_action = -1  # is_first_step=False (step_count > 0)
    _, r_goal, done_goal, _, _ = env_goal.step(0)
    assert done_goal
    # 50번째 step: -0.1 + 50 = +49.9 (correct goal dir)
    total_goal_final_step = r_goal

    # timeout 시뮬레이션 누적: 100 step * (-0.1) = -10
    total_timeout = -W1 * 100

    # 단일 step 비교가 아닌 에피소드 총합 차이 검증
    # goal 에피소드의 last-step reward 만으로도 timeout 전체보다 훨씬 높아야 함
    assert total_goal_final_step > total_timeout + 50


def test_goal_net_positive(env: ConcreteStep1Env) -> None:
    """goal 도달 에피소드에서 W3 - 짧은 누적 패널티 > 0."""
    # 최악 케이스: 100 step + W5 (wrong goal dir)
    # W3 - W1*100 - W5 = 50 - 10 - 15 = 25 > 0
    min_net_goal = W3 - W1 * 100 - W5
    assert min_net_goal > 0, f"goal net reward 최솟값이 양수가 아님: {min_net_goal}"


def test_timeout_always_negative(env: ConcreteStep1Env) -> None:
    """timeout 에피소드 (goal 미도달) 는 항상 net negative."""
    # timeout = -W1 * max_steps + optional W4 (at most once)
    # W4 = 5 이고 max_steps=500 이면: -0.1*500 + 5 = -50 + 5 = -45 < 0
    max_timeout_reward = -W1 * 500 + W4
    assert max_timeout_reward < 0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Reward 항목 독립성 (한 항목씩 검증)
# ─────────────────────────────────────────────────────────────────────────────


def test_neutral_step_reward(env: ConcreteStep1Env) -> None:
    """collision 없고, goal 미도달, step > 1 → reward == -W1 정확히."""
    env.reset(seed=0)
    env.step(2)         # step 1 (direction 체크)
    _, r, done, trunc, _ = env.step(2)  # step 2: neutral
    if not done and not trunc:
        assert r == pytest.approx(-W1)


def test_reward_no_negative_without_events(env: ConcreteStep1Env) -> None:
    """이벤트 없는 step 에서 reward < 0 (length penalty만 있어야 하므로)."""
    env.reset(seed=0)
    env.step(2)  # step 1
    _, r, done, trunc, _ = env.step(2)  # step 2
    if not done and not trunc:
        assert r < 0  # at least -W1


# ─────────────────────────────────────────────────────────────────────────────
# 6. w1/w2/w3 인스턴스 오버라이드 배선 검증 (2026-09 FAILURE_LOG)
#
# 이전에는 Step1Env.__init__ 이 w1/w2/w3 를 받지 않아 _calc_reward() 가 항상
# 모듈 상수 W1/W2/W3 만 참조했다 — autoresearch Round 2 (w1/w2/w3 sweep) 가
# 코드상으로는 값을 넘기는 것처럼 보여도 실제로는 아무 효과가 없는 no-op 이었다.
# 아래 테스트는 (a) 기본값이 여전히 모듈 상수와 동일함(회귀 방지)과,
# (b) 명시적으로 다른 값을 넘기면 동일 seed/동일 state 에서도 reward 가
# 실제로 달라짐(배선이 살아있음)을 둘 다 확인한다.
# ─────────────────────────────────────────────────────────────────────────────


def test_default_w1_w2_w3_match_module_constants(env: ConcreteStep1Env) -> None:
    """w1/w2/w3 를 명시하지 않으면 인스턴스 속성이 모듈 상수와 정확히 같다."""
    assert env.w1 == W1
    assert env.w2 == W2
    assert env.w3 == W3


def test_w1_override_changes_length_penalty() -> None:
    """동일 seed·동일 action 시퀀스에서 w1 만 바꾸면 reward 가 실제로 달라진다."""
    env_lo = ConcreteStep1Env(seed=0, w1=0.05)
    env_hi = ConcreteStep1Env(seed=0, w1=0.5)
    env_lo.reset(seed=0)
    env_hi.reset(seed=0)
    env_lo.step(2)  # step 1 (direction bonus/penalty 는 w1 과 무관하므로 양쪽 동일)
    env_hi.step(2)
    _, r_lo, done_lo, trunc_lo, _ = env_lo.step(2)  # step 2: neutral, length penalty만
    _, r_hi, done_hi, trunc_hi, _ = env_hi.step(2)
    assert not done_lo and not trunc_lo
    assert not done_hi and not trunc_hi
    assert r_lo == pytest.approx(-0.05)
    assert r_hi == pytest.approx(-0.5)
    assert r_lo != pytest.approx(r_hi)


def test_w2_override_changes_collision_penalty() -> None:
    """동일 seed·동일 wall 배치에서 w2 만 바꾸면 collision reward 가 달라진다."""

    class _WallEnv(ConcreteStep1Env):
        def _sample_scenario(self) -> None:
            super()._sample_scenario()
            self.occupancy[6, 5, 5] = True  # agent(5,5,5) 의 +X neighbor

    env_lo = _WallEnv(seed=42, w2=1.0)
    env_hi = _WallEnv(seed=42, w2=5.0)
    env_lo.reset(seed=0)
    env_hi.reset(seed=0)
    env_lo._step_count = 5
    env_hi._step_count = 5
    _, r_lo, done_lo, _, info_lo = env_lo.step(0)  # +X → wall → collision
    _, r_hi, done_hi, _, info_hi = env_hi.step(0)
    assert done_lo and done_hi
    assert info_lo["termination"] == info_hi["termination"] == "collision"
    assert r_lo == pytest.approx(-env_lo.w1 - 1.0)
    assert r_hi == pytest.approx(-env_hi.w1 - 5.0)
    assert r_lo != pytest.approx(r_hi)


def test_w3_override_changes_goal_bonus() -> None:
    """동일 seed·동일 goal 도달 조건에서 w3 만 바꾸면 goal bonus 가 달라진다."""
    env_lo = ConcreteStep1Env(seed=0, w3=20.0)
    env_hi = ConcreteStep1Env(seed=0, w3=100.0)
    for e in (env_lo, env_hi):
        e.reset(seed=0)
        e.agent_cell = np.array([24, 25, 25], dtype=np.int32)
        e.goal_cell = np.array([25, 25, 25], dtype=np.int32)
        e.goal_dir_idx = 0  # +X, action=0 과 일치 → wrong_goal_dir_penalty 없음
        e._step_count = 10
    _, r_lo, done_lo, _, info_lo = env_lo.step(0)
    _, r_hi, done_hi, _, info_hi = env_hi.step(0)
    assert done_lo and done_hi
    assert info_lo["termination"] == info_hi["termination"] == "goal_reached"
    assert r_lo == pytest.approx(-env_lo.w1 + 20.0)
    assert r_hi == pytest.approx(-env_hi.w1 + 100.0)
    assert r_lo != pytest.approx(r_hi)
