"""150-dim observation space 검증 — CLAUDE.md §4.3.

각 슬롯의 값 정합성, dtype, zero-padding 영역을 검증한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from envs.base_env import (
    FACE_DIRS,
    I_GOAL_DIR_IDX,
    I_GOAL_DIST,
    I_IS_FIRST_STEP,
    I_PATH_PROGRESS,
    I_PIPE_GRAVITY,
    I_PIPE_INSULATION,
    I_PIPE_RADIUS,
    I_PIPE_SIZE,
    I_STEP_PROGRESS,
    JIS_EFF_RADIUS_NORM,
    N_JIS_SIZES,
    OBS_DIM,
    OBS_RESERVE_START,
    OBS_STEP1_ACTIVE_END,
    S_AGENT_POS,
    S_COLLISION,
    S_GOAL_DIR,
    S_GOAL_POS,
    S_LAST_ACTION,
    S_PIPE_TYPE,
    S_RAYCAST,
    S_SDF_FACE,
    S_START_DIR,
    S_TO_GOAL_VEC,
)
from tests.conftest import ConcreteStep1Env

# ─────────────────────────────────────────────────────────────────────────────
# 1. 기본 spec 준수
# ─────────────────────────────────────────────────────────────────────────────


def test_obs_shape(env: ConcreteStep1Env) -> None:
    """obs shape == (150,). SKILL §3.1 절대 변경 금지."""
    obs, _ = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)


def test_obs_dtype(env: ConcreteStep1Env) -> None:
    """obs dtype == float64. (SKILL §3.3: 내부 저장은 uint8/bool, obs 출력은 float64)."""
    obs, _ = env.reset(seed=0)
    assert obs.dtype == np.float64


def test_obs_no_nan_inf(env: ConcreteStep1Env) -> None:
    """obs 에 NaN/Inf 없어야 한다."""
    obs, _ = env.reset(seed=0)
    assert np.all(np.isfinite(obs))


# ─────────────────────────────────────────────────────────────────────────────
# 2. 위치 슬롯 정합성
# ─────────────────────────────────────────────────────────────────────────────


def test_agent_pos_slot(env: ConcreteStep1Env) -> None:
    """obs[S_AGENT_POS] = agent_cell / (grid_shape - 1)."""
    obs, _ = env.reset(seed=0)
    expected = env.agent_cell / (np.array(env.grid_shape) - 1)
    np.testing.assert_allclose(obs[S_AGENT_POS], expected)


def test_goal_pos_slot(env: ConcreteStep1Env) -> None:
    """obs[S_GOAL_POS] = goal_cell / (grid_shape - 1)."""
    obs, _ = env.reset(seed=0)
    expected = env.goal_cell / (np.array(env.grid_shape) - 1)
    np.testing.assert_allclose(obs[S_GOAL_POS], expected)


def test_to_goal_vec_unit(env: ConcreteStep1Env) -> None:
    """obs[S_TO_GOAL_VEC] 는 단위 벡터여야 한다."""
    obs, _ = env.reset(seed=0)
    norm = float(np.linalg.norm(obs[S_TO_GOAL_VEC]))
    assert abs(norm - 1.0) < 1e-9


def test_goal_dist_at_reset_is_one(env: ConcreteStep1Env) -> None:
    """reset 직후 obs[I_GOAL_DIST] == 1.0 (d_initial 정규화)."""
    obs, _ = env.reset(seed=0)
    assert obs[I_GOAL_DIST] == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. SDF 슬롯 정합성
# ─────────────────────────────────────────────────────────────────────────────


def test_sdf_face_range(env: ConcreteStep1Env) -> None:
    """obs[S_SDF_FACE] 는 [0, 1] 범위. (장애물 없는 환경 → 값 > 0)."""
    obs, _ = env.reset(seed=0)
    sdf = obs[S_SDF_FACE]
    assert np.all(sdf >= 0.0)
    assert np.all(sdf <= 1.0)
    # 장애물 없는 빈 grid → face neighbors 모두 SDF > 0
    assert np.all(sdf > 0.0)


def test_sdf_face_neighbor_with_wall(env_with_wall: ConcreteStep1Env) -> None:
    """agent +X neighbor 가 wall 이면 obs[S_SDF_FACE][0] ≈ 0."""
    obs, _ = env_with_wall.reset(seed=0)
    # FACE_DIRS[0] = +X → index 0
    assert obs[S_SDF_FACE.start] == pytest.approx(0.0)


def test_collision_flag_set_for_wall(env_with_wall: ConcreteStep1Env) -> None:
    """agent +X 가 wall 이면 obs[S_COLLISION][0] == 1.0."""
    obs, _ = env_with_wall.reset(seed=0)
    assert obs[S_COLLISION.start] == 1.0


def test_collision_flag_zero_for_free(env: ConcreteStep1Env) -> None:
    """장애물 없는 환경에서 collision flag 모두 0."""
    obs, _ = env.reset(seed=0)
    assert np.all(obs[S_COLLISION] == 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Raycast 슬롯
# ─────────────────────────────────────────────────────────────────────────────


def test_raycast_positive(env: ConcreteStep1Env) -> None:
    """장애물 없는 환경에서 raycast > 0 (벽이 있어야 멈추므로 grid 끝까지 이동)."""
    obs, _ = env.reset(seed=0)
    assert np.all(obs[S_RAYCAST] > 0.0)


def test_raycast_range(env: ConcreteStep1Env) -> None:
    """obs[S_RAYCAST] 값은 [0, 1] 범위."""
    obs, _ = env.reset(seed=0)
    assert np.all(obs[S_RAYCAST] >= 0.0)
    assert np.all(obs[S_RAYCAST] <= 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Episode progress 슬롯
# ─────────────────────────────────────────────────────────────────────────────


def test_step_progress_zero_at_reset(env: ConcreteStep1Env) -> None:
    """reset 직후 obs[I_STEP_PROGRESS] == 0.0."""
    obs, _ = env.reset(seed=0)
    assert obs[I_STEP_PROGRESS] == pytest.approx(0.0)


def test_is_first_step_at_reset(env: ConcreteStep1Env) -> None:
    """reset 직후 obs[I_IS_FIRST_STEP] == 1.0."""
    obs, _ = env.reset(seed=0)
    assert obs[I_IS_FIRST_STEP] == pytest.approx(1.0)


def test_is_first_step_zero_after_action(env: ConcreteStep1Env) -> None:
    """한 step 진행 후 obs[I_IS_FIRST_STEP] == 0.0."""
    env.reset(seed=0)
    # 자유 방향 중 하나로 이동 (action 2 = +Y, 장애물 없음)
    obs, _, _, _, _ = env.step(2)
    assert obs[I_IS_FIRST_STEP] == pytest.approx(0.0)


def test_last_action_onehot_after_step(env: ConcreteStep1Env) -> None:
    """step(2) 후 obs[S_LAST_ACTION][2] == 1.0, 나머지 0."""
    env.reset(seed=0)
    obs, _, _, _, _ = env.step(2)
    expected = np.zeros(7, dtype=np.float64)
    expected[2] = 1.0
    np.testing.assert_array_equal(obs[S_LAST_ACTION], expected)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Pipe attributes 슬롯
# ─────────────────────────────────────────────────────────────────────────────


def test_pipe_size_slot(env: ConcreteStep1Env) -> None:
    """obs[I_PIPE_SIZE] = pipe_nominal_size_idx / (N_JIS_SIZES - 1)."""
    obs, _ = env.reset(seed=0)
    expected = env.pipe_nominal_size_idx / (N_JIS_SIZES - 1)
    assert obs[I_PIPE_SIZE] == pytest.approx(expected)


def test_pipe_insulation_slot(env: ConcreteStep1Env) -> None:
    """보온재 있음 → obs[I_PIPE_INSULATION] == 1.0."""
    obs, _ = env.reset(seed=0)
    assert obs[I_PIPE_INSULATION] == pytest.approx(1.0)


def test_pipe_radius_slot(env: ConcreteStep1Env) -> None:
    """obs[I_PIPE_RADIUS] = effective_radius / JIS_EFF_RADIUS_NORM."""
    obs, _ = env.reset(seed=0)
    expected = env.pipe_effective_radius_mm / JIS_EFF_RADIUS_NORM
    assert obs[I_PIPE_RADIUS] == pytest.approx(expected)


def test_pipe_type_onehot_t1(env: ConcreteStep1Env) -> None:
    """T1 (idx=0) → obs[S_PIPE_TYPE][0] == 1.0, 나머지 0."""
    obs, _ = env.reset(seed=0)
    expected = np.zeros(8, dtype=np.float64)
    expected[0] = 1.0
    np.testing.assert_array_equal(obs[S_PIPE_TYPE], expected)


def test_pipe_gravity_zero_for_t1(env: ConcreteStep1Env) -> None:
    """T1 (압력관) → obs[I_PIPE_GRAVITY] == 0.0."""
    obs, _ = env.reset(seed=0)
    assert obs[I_PIPE_GRAVITY] == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 7. CLAUDE.md §12.4 방향 슬롯
# ─────────────────────────────────────────────────────────────────────────────


def test_start_dir_is_face_unit_vector(env: ConcreteStep1Env) -> None:
    """obs[S_START_DIR] 는 FACE_DIRS 중 하나의 단위 벡터여야 한다."""
    obs, _ = env.reset(seed=0)
    vec = obs[S_START_DIR]
    assert any(np.allclose(vec, FACE_DIRS[i]) for i in range(6))


def test_goal_dir_is_face_unit_vector(env: ConcreteStep1Env) -> None:
    """obs[S_GOAL_DIR] 는 FACE_DIRS 중 하나의 단위 벡터여야 한다."""
    obs, _ = env.reset(seed=0)
    vec = obs[S_GOAL_DIR]
    assert any(np.allclose(vec, FACE_DIRS[i]) for i in range(6))


def test_goal_dir_idx_normalized(env: ConcreteStep1Env) -> None:
    """obs[I_GOAL_DIR_IDX] = goal_dir_idx / 5.0."""
    obs, _ = env.reset(seed=0)
    expected = env.goal_dir_idx / 5.0
    assert obs[I_GOAL_DIR_IDX] == pytest.approx(expected)


def test_start_goal_dirs_not_same_not_opposite(env: ConcreteStep1Env) -> None:
    """start_dir 과 goal_dir 이 동일하거나 정반대가 되면 안 된다. CLAUDE.md §12.4."""
    for seed in range(20):
        _obs, info = env.reset(seed=seed)
        si = info["start_dir_idx"]
        gi = info["goal_dir_idx"]
        assert si != gi
        assert not np.array_equal(FACE_DIRS[si], -FACE_DIRS[gi])


# ─────────────────────────────────────────────────────────────────────────────
# 8. Zero-padding 영역 (Phase 1 보장)
# ─────────────────────────────────────────────────────────────────────────────


def test_step2_to_step6_slots_are_zero(env: ConcreteStep1Env) -> None:
    """obs[98:136] (Step 2~6 슬롯) 은 Step 1 에서 모두 0. CLAUDE.md §4.3."""
    obs, _ = env.reset(seed=0)
    np.testing.assert_array_equal(
        obs[OBS_STEP1_ACTIVE_END:OBS_RESERVE_START], 0.0
    )


def test_reserve_slots_are_zero(env: ConcreteStep1Env) -> None:
    """obs[136:150] (다중 파이프 예비) Phase 1 에서 모두 0. CLAUDE.md §4.3."""
    obs, _ = env.reset(seed=0)
    np.testing.assert_array_equal(obs[OBS_RESERVE_START:], 0.0)


def test_step1_reserved_block_is_zero(env: ConcreteStep1Env) -> None:
    """obs[79:91] (Step 1 내부 reserved) 모두 0."""
    obs, _ = env.reset(seed=0)
    np.testing.assert_array_equal(obs[79:91], 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 9. 에피소드 진행 후 obs 업데이트 확인
# ─────────────────────────────────────────────────────────────────────────────


def test_obs_updates_after_step(env: ConcreteStep1Env) -> None:
    """step 후 agent 위치가 바뀌면 obs[S_AGENT_POS] 도 바뀐다."""
    obs0, _ = env.reset(seed=0)
    # +Y 방향으로 이동 (action 2)
    obs1, _, _, _, _ = env.step(2)
    assert not np.allclose(obs0[S_AGENT_POS], obs1[S_AGENT_POS])


def test_path_progress_increases(env: ConcreteStep1Env) -> None:
    """step 진행할수록 obs[I_PATH_PROGRESS] 가 증가한다."""
    env.reset(seed=0)
    obs0, _, _, _, _ = env.step(2)  # +Y
    obs1, _, _, _, _ = env.step(2)  # +Y 추가
    assert obs1[I_PATH_PROGRESS] > obs0[I_PATH_PROGRESS]
