"""Step 1 environment — CLAUDE.md §2.1, §11.2, Step 1 (장애물 회피).

Phase 1 학습 대상.
- _get_obs()   : obs[0:98] 구현 + obs[98:150] zero-padding.
- _calc_reward(): baseline reward (CLAUDE.md §16.3.1).
- _sample_scenario(): NotImplementedError — 단계 3, generator 연결.
- _get_action_mask(): NotImplementedError — 단계 3, §12.5 마스킹 규칙.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from envs.base_env import (
    EDGE_DIRS,
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
    N_FACE_DIRS,
    N_JIS_SIZES,
    OBS_DIM,
    S_AGENT_POS,
    S_COLLISION,
    S_GOAL_DIR,
    S_GOAL_POS,
    S_LAST_ACTION,
    S_PATH_TANGENT,
    S_PIPE_TYPE,
    S_RAYCAST,
    S_SDF_GRADIENT,
    S_START_DIR,
    S_TO_GOAL_VEC,
    SDF_MAX_CELLS,
    VERTEX_DIRS,
    BaseEnv,
)

# ─────────────────────────────────────────────────────────────────────────────
# Baseline reward 가중치 (CLAUDE.md §16.3.1, Phase 1 초기값)
# w4, w5 는 §12.4 고정값 — autoresearch sweep 대상 아님
# ─────────────────────────────────────────────────────────────────────────────
W1: float = 0.1   # step 당 length penalty
W2: float = 2.0   # collision/OOB terminal penalty
W3: float = 50.0  # goal 도달 bonus
W4: float = 5.0   # start direction align bonus (§12.4 고정)
W5: float = 15.0  # wrong direction penalty (§12.4 고정)


class Step1Env(BaseEnv):
    """Step 1 (장애물 회피) 환경. CLAUDE.md §11.2, §16.3.1.

    obs[0:98]  : 활성 슬롯 (base_env.py obs slot map 참조).
    obs[98:150]: zero-padding (Step 2~6 슬롯 + 예비).

    현재 상태:
        _sample_scenario() : NotImplementedError — 단계 3.
        _get_action_mask() : NotImplementedError — 단계 3.
    """

    # ── Observation (CLAUDE.md §4.3, obs[0:98]) ───────────────────────────────

    def _get_obs(self) -> npt.NDArray[np.float64]:
        """150-dim observation 구성. CLAUDE.md §4.3."""
        obs = np.zeros(OBS_DIM, dtype=np.float64)
        shape_f = np.array(self.grid_shape, dtype=np.float64)

        # [0:3] agent 정규화 위치
        obs[S_AGENT_POS] = self.agent_cell / (shape_f - 1)

        # [3:6] goal 정규화 위치
        obs[S_GOAL_POS] = self.goal_cell / (shape_f - 1)

        # [6:9] agent→goal 단위 벡터
        delta = (self.goal_cell - self.agent_cell).astype(np.float64)
        eucl = float(np.linalg.norm(delta))
        obs[S_TO_GOAL_VEC] = delta / eucl if eucl > 0 else delta

        # [9] Manhattan dist / d_initial
        obs[I_GOAL_DIST] = float(np.abs(delta).sum()) / self.d_initial

        # [10:16] SDF face-adjacent (6 방향)
        max_dist = float(max(self.grid_shape))
        for i, d in enumerate(FACE_DIRS):
            obs[10 + i] = self._get_sdf(self.agent_cell + d) / SDF_MAX_CELLS

        # [16:28] SDF edge-adjacent (12 방향)
        for i, d in enumerate(EDGE_DIRS):
            obs[16 + i] = self._get_sdf(self.agent_cell + d) / SDF_MAX_CELLS

        # [28:36] SDF vertex-adjacent (8 방향)
        for i, d in enumerate(VERTEX_DIRS):
            obs[28 + i] = self._get_sdf(self.agent_cell + d) / SDF_MAX_CELLS

        # [36:42] Raycast in 6 FACE_DIRS
        for i, d in enumerate(FACE_DIRS):
            obs[S_RAYCAST.start + i] = self._raycast(d) / max_dist

        # [42:48] Binary collision flags in 6 FACE_DIRS
        for i, d in enumerate(FACE_DIRS):
            obs[S_COLLISION.start + i] = 1.0 if self.is_occupied(self.agent_cell + d) else 0.0

        # [48] [49] episode progress
        obs[I_STEP_PROGRESS] = self._step_count / self.max_steps
        obs[I_PATH_PROGRESS] = len(self.path) / self.max_steps

        # [50:62] pipe attributes
        obs[I_PIPE_SIZE] = self.pipe_nominal_size_idx / (N_JIS_SIZES - 1)
        obs[I_PIPE_INSULATION] = 1.0 if self.pipe_has_insulation else 0.0
        obs[I_PIPE_RADIUS] = self.pipe_effective_radius_mm / JIS_EFF_RADIUS_NORM
        if 0 <= self.pipe_type_idx < 8:
            obs[S_PIPE_TYPE.start + self.pipe_type_idx] = 1.0
        obs[I_PIPE_GRAVITY] = 1.0 if self.pipe_gravity else 0.0

        # [62:69] last action one-hot (7-dim). [69] is_first_step
        if self._last_action >= 0:
            obs[S_LAST_ACTION.start + self._last_action] = 1.0
            obs[I_IS_FIRST_STEP] = 0.0
        else:
            obs[I_IS_FIRST_STEP] = 1.0  # reset 직후

        # [70:73] path tangent: 최근 5스텝 평균 방향 단위 벡터
        n_path = len(self.path)
        if n_path >= 2:
            recent = min(5, n_path - 1)
            tangent = np.zeros(3, dtype=np.float64)
            for j in range(n_path - recent, n_path):
                tangent += (self.path[j] - self.path[j - 1]).astype(np.float64)
            norm = float(np.linalg.norm(tangent))
            obs[S_PATH_TANGENT] = tangent / norm if norm > 0 else tangent

        # [73:79] SDF gradient in 6 FACE_DIRS (∂SDF/∂dir ≈ SDF[neighbor] - SDF[agent])
        sdf_agent = self._get_sdf(self.agent_cell)
        for i, d in enumerate(FACE_DIRS):
            obs[S_SDF_GRADIENT.start + i] = (
                self._get_sdf(self.agent_cell + d) - sdf_agent
            ) / SDF_MAX_CELLS

        # [79:91] Step 1 reserved — zero

        # [91:94] start_dir unit vector (CLAUDE.md §12.4)
        obs[S_START_DIR] = FACE_DIRS[self.start_dir_idx].astype(np.float64)

        # [94:97] goal_dir unit vector (CLAUDE.md §12.4)
        obs[S_GOAL_DIR] = FACE_DIRS[self.goal_dir_idx].astype(np.float64)

        # [97] goal_dir_idx normalized
        obs[I_GOAL_DIR_IDX] = self.goal_dir_idx / (N_FACE_DIRS - 1)

        # obs[98:150] Step 2~6 슬롯 + 예비 — zero-padding (already zero)
        return obs

    # ── Reward (CLAUDE.md §16.3.1) ────────────────────────────────────────────

    def _calc_reward(self, action: int, terminated: bool, truncated: bool) -> float:
        """Baseline reward. CLAUDE.md §16.3.1 (Phase 1 초기값 w1~w5).

        r = -w1·(per-step length)
            - w2·(terminal collision or OOB)
            + w3·(goal reached)
            + w4·(start direction align, step 1 only)
            - w5·(wrong start direction, step 1 only)
            - w5·(wrong goal approach direction, on goal_reached)

        Note: PBS shaping (§16.7) 은 단계 3 에서 wrapper 로 추가.
        """
        reward = 0.0

        # ─ length penalty: 매 스텝 -w1 (CLAUDE.md §16.3.1 "step 당 가벼운 길이 penalty")
        reward -= W1

        # ─ collision / OOB terminal penalty
        if terminated and self._termination_reason in ("collision", "out_of_bounds"):
            reward -= W2

        # ─ goal bonus
        if self._termination_reason == "goal_reached":
            reward += W3

        # ─ start direction 제약 (CLAUDE.md §12.4, 첫 스텝에서만)
        if self._step_count == 1:
            if action == self.start_dir_idx:
                reward += W4
            else:
                reward -= W5

        # ─ goal approach direction 제약 (CLAUDE.md §12.4, goal 도달 시)
        # Edge/Vertex 진입은 6-direction discrete 에서 발생하지 않음 (face action only)
        if self._termination_reason == "goal_reached" and action != self.goal_dir_idx:
            reward -= W5

        return reward

    # ── Abstract stubs (단계 3) ───────────────────────────────────────────────

    def _sample_scenario(self) -> None:
        """Step 1 시나리오 초기화. CLAUDE.md §11.0.8 generator 연결 (단계 3)."""
        raise NotImplementedError(
            "_sample_scenario() 는 단계 3 (generator 연결) 에서 구현 예정."
        )

    def _get_action_mask(self) -> npt.NDArray[np.bool_]:
        """7-direction action mask. CLAUDE.md §12.5. 단계 3 에서 구현."""
        raise NotImplementedError(
            "_get_action_mask() 는 단계 3 (action mask 구현) 에서 구현 예정."
        )
