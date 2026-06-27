"""Action mask 검증 — CLAUDE.md §12.5 Step 1 규칙.

규칙 검증:
  1. 첫 스텝: start_dir_idx 만 허용
  2. 충돌 / OOB 방향 차단
  3. 자기경로 역주행 차단
  4. Action 6 항상 마스킹
  5. action_masks() == _get_action_mask() (sb3-contrib 호환)
  6. deadlock 방지: 유효 action >= 1
"""

from __future__ import annotations

import numpy as np

from envs.base_env import FACE_DIRS, N_ACTIONS, N_FACE_DIRS
from tests.conftest import ConcreteStep1Env

# ─────────────────────────────────────────────────────────────────────────────
# 1. 첫 스텝 규칙 (CLAUDE.md §12.4 Hard Constraint)
# ─────────────────────────────────────────────────────────────────────────────


def test_first_step_only_start_dir(env: ConcreteStep1Env) -> None:
    """reset 직후 mask 는 start_dir_idx 만 True."""
    _, info = env.reset(seed=0)
    mask = env.action_masks()
    start_dir = info["start_dir_idx"]

    assert mask[start_dir], "start_dir_idx 방향이 마스킹돼 있다"
    for i in range(N_ACTIONS):
        if i != start_dir:
            assert not mask[i], f"action {i} 가 첫 step 에서 허용됨 (start_dir={start_dir})"


def test_first_step_mask_exact_one_true(env: ConcreteStep1Env) -> None:
    """첫 step mask 에서 True 는 정확히 1개."""
    env.reset(seed=0)
    mask = env.action_masks()
    assert mask.sum() == 1


# ─────────────────────────────────────────────────────────────────────────────
# 2. Action 6 항상 마스킹 (L-D3 macro placeholder)
# ─────────────────────────────────────────────────────────────────────────────


def test_action6_always_masked_at_reset(env: ConcreteStep1Env) -> None:
    """reset 직후 action 6 은 False."""
    env.reset(seed=0)
    mask = env.action_masks()
    assert not mask[6]


def test_action6_always_masked_after_steps(env: ConcreteStep1Env) -> None:
    """여러 step 이후에도 action 6 은 항상 False."""
    env.reset(seed=0)
    env.step(2)  # +Y step (첫 step)
    env.step(2)  # 2번째 step
    mask = env.action_masks()
    assert not mask[6]


# ─────────────────────────────────────────────────────────────────────────────
# 3. 충돌 방향 차단
# ─────────────────────────────────────────────────────────────────────────────


def test_collision_direction_masked(env_with_wall: ConcreteStep1Env) -> None:
    """+X 방향에 wall 이 있을 때 action 0 (+X) 는 마스킹된다."""
    env = env_with_wall
    _, info = env.reset(seed=0)

    # 첫 step 에서 wall 이 아닌 방향으로 이동
    start_dir = info["start_dir_idx"]
    env.step(start_dir)

    # 2번째 step 에서: agent 가 여전히 wall 에 인접한 경우 확인
    # agent_cell 을 수동으로 wall 인근으로 설정
    env.agent_cell = np.array([5, 5, 5], dtype=np.int32)
    env.path = [env.agent_cell.copy()]
    env._last_action = start_dir
    env._step_count = 1

    mask = env.action_masks()
    # FACE_DIRS[0] = +X → (6,5,5) = wall
    assert not mask[0], "+X 방향이 wall 인데 마스킹 안 됨"


def test_free_direction_not_masked(env: ConcreteStep1Env) -> None:
    """장애물 없는 환경에서 step > 0 이면 face dir 6개 중 일부는 True (역주행 제외)."""
    _, info = env.reset(seed=0)
    env.step(info["start_dir_idx"])  # 첫 step
    mask = env.action_masks()
    # 장애물 없으므로 최소 4 방향은 열려 있어야 함 (역방향 1 + action6 제외)
    assert mask[:N_FACE_DIRS].sum() >= 4


# ─────────────────────────────────────────────────────────────────────────────
# 4. OOB 방향 차단
# ─────────────────────────────────────────────────────────────────────────────


def test_oob_direction_masked(env: ConcreteStep1Env) -> None:
    """grid 경계 셀에서 경계 밖 방향은 마스킹된다."""
    env.reset(seed=0)
    env.agent_cell = np.array([0, 15, 15], dtype=np.int32)
    env.path = [env.agent_cell.copy(), env.agent_cell.copy()]  # dummy path for step > 0
    env._step_count = 2

    mask = env.action_masks()
    # FACE_DIRS[1] = -X → (-1,15,15) = OOB
    assert not mask[1], "-X 방향이 OOB 인데 마스킹 안 됨"


# ─────────────────────────────────────────────────────────────────────────────
# 5. 역주행 차단
# ─────────────────────────────────────────────────────────────────────────────


def test_reverse_direction_masked(env: ConcreteStep1Env) -> None:
    """이전 셀 방향으로 이동하는 action 은 마스킹된다."""
    _, info = env.reset(seed=0)
    start_dir = info["start_dir_idx"]

    env.step(start_dir)  # path: [start, start+dir]

    mask = env.action_masks()

    # 역방향 = -FACE_DIRS[start_dir]
    reverse_dir_vec = -FACE_DIRS[start_dir]
    reverse_action = next(
        i for i in range(N_FACE_DIRS) if np.array_equal(FACE_DIRS[i], reverse_dir_vec)
    )
    assert not mask[reverse_action], "역주행 방향이 마스킹 안 됨"


# ─────────────────────────────────────────────────────────────────────────────
# 6. sb3-contrib 호환 인터페이스
# ─────────────────────────────────────────────────────────────────────────────


def test_action_masks_method_exists(env: ConcreteStep1Env) -> None:
    """action_masks() public method 존재 (MaskablePPO 호환)."""
    assert hasattr(env, "action_masks")
    assert callable(env.action_masks)


def test_action_masks_equals_internal(env: ConcreteStep1Env) -> None:
    """action_masks() == _get_action_mask() (동일 결과)."""
    env.reset(seed=0)
    np.testing.assert_array_equal(env.action_masks(), env._get_action_mask())


def test_action_mask_dtype(env: ConcreteStep1Env) -> None:
    """mask dtype == bool."""
    env.reset(seed=0)
    mask = env.action_masks()
    assert mask.dtype == np.bool_


def test_action_mask_shape(env: ConcreteStep1Env) -> None:
    """mask shape == (N_ACTIONS,) == (7,)."""
    env.reset(seed=0)
    mask = env.action_masks()
    assert mask.shape == (N_ACTIONS,)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Deadlock 방지
# ─────────────────────────────────────────────────────────────────────────────


def test_mask_always_has_valid_action(env: ConcreteStep1Env) -> None:
    """임의 step 에서 action_masks() 에는 항상 최소 1개 이상의 True 가 있다."""
    rng = np.random.default_rng(0)
    env.reset(seed=0)
    # 100번 step 또는 done 까지
    done = False
    for _ in range(100):
        if done:
            env.reset(seed=int(rng.integers(0, 100)))
            done = False
        mask = env.action_masks()
        assert mask.any(), "action mask 가 모두 False (deadlock)"
        valid = np.where(mask)[0]
        action = int(rng.choice(valid))
        _, _, term, trunc, _ = env.step(action)
        done = term or trunc


# ─────────────────────────────────────────────────────────────────────────────
# 8. Step 1 환경에서 실제 action mask 통합 동작
# ─────────────────────────────────────────────────────────────────────────────


def _all_valid_actions(env: ConcreteStep1Env) -> list[int]:
    return list(np.where(env.action_masks())[0])


def test_step_with_masked_action_terminates(env_with_wall: ConcreteStep1Env) -> None:
    """wall 방향으로 강제 이동 시 collision terminal (mask 는 game-rule, 에이전트 강제 아님)."""
    env = env_with_wall
    _, info = env.reset(seed=0)
    env.step(info["start_dir_idx"])  # 첫 step 완료
    env.agent_cell = np.array([5, 5, 5], dtype=np.int32)
    env._step_count = 1
    # wall 방향 (action 0 = +X → (6,5,5) = obstacle) 강제 호출
    _, _, terminated, _, info_s = env.step(0)
    assert terminated
    assert info_s["termination"] == "collision"
