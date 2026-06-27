"""Step 1 environment — CLAUDE.md §2.1, §11.2, §12.4, §12.5, §16.3.1, §16.7.

Phase 1 학습 대상 (장애물 회피).
단계 3 완료: action mask + PBS wrapper + Generator 통합.

하이퍼파라미터 (autoresearch sweep 대상):
    alpha, beta       : PBS Φ 가중치 (CLAUDE.md §16.7.6, L-D2 해제)
    gamma_ppo         : PPO discount factor (gamma_PBS == gamma_PPO 강제, 조건 4)
    mode              : "train" | "screening" | "regression" | "final_eval"
    difficulty        : "easy" | "medium" | "hard"
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
    JIS_EFF_RADIUS_MM,
    JIS_EFF_RADIUS_NORM,
    N_ACTIONS,
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
    S_SDF_EDGE,
    S_SDF_FACE,
    S_SDF_GRADIENT,
    S_SDF_VERTEX,
    S_START_DIR,
    S_TO_GOAL_VEC,
    SDF_MAX_CELLS,
    VERTEX_DIRS,
    BaseEnv,
)
from envs.scenario_generator import (
    STEP1_FINAL_EVAL_SEEDS,
    STEP1_REGRESSION_SEEDS,
    STEP1_SCREENING_SEEDS,
    STEP1_TRAIN_SEED_END,
    STEP1_TRAIN_SEED_START,
    ScenarioGenerationFailure,
    ScenarioGeneratorV1,
)
from shaping import PotentialBasedShaping

# ─────────────────────────────────────────────────────────────────────────────
# Baseline reward 가중치 (CLAUDE.md §16.3.1, Phase 1 초기값)
# w4, w5 는 §12.4 고정값 — autoresearch sweep 대상 아님
# ─────────────────────────────────────────────────────────────────────────────
W1: float = 0.1    # step 당 length penalty
W2: float = 2.0    # collision/OOB terminal penalty
W3: float = 50.0   # goal 도달 bonus
W4: float = 5.0    # start direction align bonus (§12.4 고정)
W5: float = 15.0   # wrong direction penalty (§12.4 고정)

# 유효 모드
_VALID_MODES = frozenset({"train", "screening", "regression", "final_eval"})


class Step1Env(BaseEnv):
    """Step 1 (장애물 회피) 환경. CLAUDE.md §11.2, §16.3.1, §16.7.

    obs[0:98]  : 활성 슬롯 (base_env.py obs slot map 참조).
    obs[98:150]: zero-padding (Step 2~6 슬롯 + 예비).

    Parameters
    ----------
    mode :
        시나리오 seed pool 모드. "train" 은 150000~ 순차 사용.
    difficulty :
        "easy" | "medium" | "hard". CLAUDE.md §11.0.3.
    alpha :
        PBS Φ_goal 가중치. CLAUDE.md §16.7.6 초기값 1.0.
    beta :
        PBS Φ_cong 가중치. CLAUDE.md §16.7.6 초기값 0.1.
    gamma_ppo :
        PPO discount factor. gamma_PBS == gamma_PPO 조건 4 강제.
        autoresearch sweep 시 gamma 변경하면 이 값도 함께 변경.
    **kwargs :
        BaseEnv 로 전달 (space_size_mm, max_steps, seed).
    """

    def __init__(
        self,
        mode: str = "train",
        difficulty: str = "medium",
        alpha: float = 1.0,
        beta: float = 0.1,
        gamma_ppo: float = 0.99,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)

        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
        if difficulty not in ("easy", "medium", "hard"):
            raise ValueError(f"difficulty must be easy/medium/hard, got {difficulty!r}")

        self.mode = mode
        self.difficulty = difficulty
        self.alpha = alpha
        self.beta = beta

        # Scenario generator v1.0.0
        self._generator = ScenarioGeneratorV1()

        # Seed pool 인덱스
        self._train_seed_counter: int = STEP1_TRAIN_SEED_START
        self._pool_idx: dict[str, int] = {
            "screening": 0,
            "regression": 0,
            "final_eval": 0,
        }

        # PBS wrapper (CLAUDE.md §16.7.5 — helper class 반드시 경유)
        self._pbs = PotentialBasedShaping(
            phi_fn=self._phi,   # bound method, signature: (obs) → float (조건 1 통과)
            gamma_ppo=gamma_ppo,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PBS potential function (CLAUDE.md §16.7.2, Phase 1 spec)
    # ─────────────────────────────────────────────────────────────────────────

    def _phi(self, obs: npt.NDArray[np.float64]) -> float:
        """Φ(s) = alpha*Φ_goal(s) + β·Φ_cong(s). CLAUDE.md §16.7.2.

        Parameters
        ----------
        obs :
            150-dim observation vector (state representation).
            action-independent — PBS 조건 1 충족.

        Notes
        -----
        Φ_goal = -obs[I_GOAL_DIST]
                 = -d_manhattan / d_initial, 범위 [-1, 0].

        Φ_cong = -1 / (mean_SDF_neighbors * SDF_MAX_CELLS + 1.0)
                 26-neighbor 평균으로 5x5x5 window 근사.
        """
        phi_goal = -float(obs[I_GOAL_DIST])  # 가까울수록 0 에 수렴

        # 26-neighbor SDF 의 평균 (셀 단위로 환산)
        sdf_26 = np.concatenate([obs[S_SDF_FACE], obs[S_SDF_EDGE], obs[S_SDF_VERTEX]])
        mean_sdf_cells = float(sdf_26.mean()) * SDF_MAX_CELLS
        phi_cong = -1.0 / (mean_sdf_cells + 1.0)  # 막힌 공간일수록 음수 커짐

        return self.alpha * phi_goal + self.beta * phi_cong

    # ─────────────────────────────────────────────────────────────────────────
    # PBS shaping reward override (CLAUDE.md §16.7)
    # ─────────────────────────────────────────────────────────────────────────

    def _calc_shape(
        self,
        s: npt.NDArray[np.float64],
        s_next: npt.NDArray[np.float64],
        is_terminal: bool,
    ) -> float:
        """r_shape = gamma*Φ(s') - Φ(s). CLAUDE.md §16.7, PotentialBasedShaping 필수 경유."""
        return self._pbs.shaping_reward(s, s_next, is_terminal)

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario seed management (CLAUDE.md §11.0.4)
    # ─────────────────────────────────────────────────────────────────────────

    def _next_seed(self) -> int:
        """모드에 따른 다음 episode seed 반환. CLAUDE.md §11.0.4."""
        if self.mode == "train":
            s = self._train_seed_counter
            self._train_seed_counter += 1
            if self._train_seed_counter > STEP1_TRAIN_SEED_END:
                self._train_seed_counter = STEP1_TRAIN_SEED_START
            return s
        pool_map = {
            "screening":  STEP1_SCREENING_SEEDS,
            "regression": STEP1_REGRESSION_SEEDS,
            "final_eval": STEP1_FINAL_EVAL_SEEDS,
        }
        pool = pool_map[self.mode]
        idx = self._pool_idx[self.mode] % len(pool)
        self._pool_idx[self.mode] += 1
        return pool[idx]

    # ─────────────────────────────────────────────────────────────────────────
    # _sample_scenario (CLAUDE.md §11.0.8 generator 연결)
    # ─────────────────────────────────────────────────────────────────────────

    def _sample_scenario(self) -> None:
        """Generator v1.0.0 으로 시나리오 생성. CLAUDE.md §11.0.8."""
        # reset(seed=N) 이 있으면 그 seed 사용, 아니면 pool 에서 다음 seed
        forced = self._forced_scenario_seed
        seed = forced if forced is not None else self._next_seed()

        occupancy = start = goal = None
        for _ in range(5):
            try:
                occupancy, start, goal = self._generator.generate(seed, self.difficulty)
                break
            except ScenarioGenerationFailure:
                seed = self._next_seed()  # fallback: 다음 seed 로 재시도 (최대 5회)
        if occupancy is None:
            raise ScenarioGenerationFailure(seed=seed, difficulty=self.difficulty)

        self.occupancy[:] = occupancy
        self.agent_cell = start.copy()
        self.goal_cell = goal.copy()

        # 파이프 속성: T1, random JIS size (Step 1)
        size_idx = int(self._episode_rng.integers(0, N_JIS_SIZES))
        self.pipe_nominal_size_idx = size_idx
        self.pipe_has_insulation = bool(self._episode_rng.integers(0, 2))
        self.pipe_effective_radius_mm = float(JIS_EFF_RADIUS_MM[size_idx])
        self.pipe_type_idx = 0   # T1 (Step 1, CLAUDE.md §3)
        self.pipe_gravity = False

    # ─────────────────────────────────────────────────────────────────────────
    # Action mask (CLAUDE.md §12.5 Step 1 규칙)
    # ─────────────────────────────────────────────────────────────────────────

    def _get_action_mask(self) -> npt.NDArray[np.bool_]:
        """Step 1 action mask. CLAUDE.md §12.5.

        규칙 (누적):
          1. 첫 스텝: start_dir_idx 만 허용 (배관 공학 원칙, 조건 Hard)
          2. 충돌 방향 차단 (occupied 또는 OOB)
          3. 자기경로 역주행 차단
          4. Action 6 (macro placeholder, L-D3): 항상 마스킹

        Deadlock 방지:
          위 규칙 적용 후 유효 action 0 이면 역주행 제약 해제 (최소 생존 보장).
        """
        mask = np.zeros(N_ACTIONS, dtype=np.bool_)

        # Rule 1: 첫 스텝 → start_dir_idx 강제 (CLAUDE.md §12.4 Hard Constraint)
        if self._step_count == 0:
            mask[self.start_dir_idx] = True
            return mask

        # Rule 2: 충돌 / OOB 방향 차단
        for i, d in enumerate(FACE_DIRS):
            if not self.is_occupied(self.agent_cell + d):
                mask[i] = True

        # Rule 3: 역주행 차단 (이전 cell 방향)
        if len(self.path) >= 2:
            reverse_dir = self.path[-2] - self.agent_cell  # 현재→이전 = 역방향
            for i, d in enumerate(FACE_DIRS):
                if np.array_equal(d, reverse_dir):
                    mask[i] = False

        # Rule 4: Action 6 항상 마스킹 (L-D3 macro placeholder)
        mask[6] = False

        # Deadlock 방지: 유효 action 없으면 역주행 제약 해제
        if not np.any(mask[:N_FACE_DIRS]):
            for i, d in enumerate(FACE_DIRS):
                if not self.is_occupied(self.agent_cell + d):
                    mask[i] = True

        # 최후 수단: 모든 face dir 허용 (극단적 deadlock)
        if not np.any(mask):
            mask[:N_FACE_DIRS] = True

        return mask

    # ─────────────────────────────────────────────────────────────────────────
    # Observation (CLAUDE.md §4.3, obs[0:98])
    # ─────────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> npt.NDArray[np.float64]:
        """150-dim observation 구성. CLAUDE.md §4.3."""
        obs = np.zeros(OBS_DIM, dtype=np.float64)
        shape_f = np.array(self.grid_shape, dtype=np.float64)

        obs[S_AGENT_POS] = self.agent_cell / (shape_f - 1)
        obs[S_GOAL_POS]  = self.goal_cell  / (shape_f - 1)

        delta = (self.goal_cell - self.agent_cell).astype(np.float64)
        eucl  = float(np.linalg.norm(delta))
        obs[S_TO_GOAL_VEC] = delta / eucl if eucl > 0 else delta
        obs[I_GOAL_DIST]   = float(np.abs(delta).sum()) / self.d_initial

        max_dist = float(max(self.grid_shape))
        for i, d in enumerate(FACE_DIRS):
            obs[S_SDF_FACE.start + i]  = self._get_sdf(self.agent_cell + d) / SDF_MAX_CELLS
        for i, d in enumerate(EDGE_DIRS):
            obs[S_SDF_EDGE.start + i]  = self._get_sdf(self.agent_cell + d) / SDF_MAX_CELLS
        for i, d in enumerate(VERTEX_DIRS):
            obs[S_SDF_VERTEX.start + i] = self._get_sdf(self.agent_cell + d) / SDF_MAX_CELLS
        for i, d in enumerate(FACE_DIRS):
            obs[S_RAYCAST.start + i]   = self._raycast(d) / max_dist
            obs[S_COLLISION.start + i] = 1.0 if self.is_occupied(self.agent_cell + d) else 0.0

        obs[I_STEP_PROGRESS] = self._step_count / self.max_steps
        obs[I_PATH_PROGRESS] = len(self.path) / self.max_steps

        obs[I_PIPE_SIZE]       = self.pipe_nominal_size_idx / (N_JIS_SIZES - 1)
        obs[I_PIPE_INSULATION] = 1.0 if self.pipe_has_insulation else 0.0
        obs[I_PIPE_RADIUS]     = self.pipe_effective_radius_mm / JIS_EFF_RADIUS_NORM
        if 0 <= self.pipe_type_idx < 8:
            obs[S_PIPE_TYPE.start + self.pipe_type_idx] = 1.0
        obs[I_PIPE_GRAVITY] = 1.0 if self.pipe_gravity else 0.0

        if self._last_action >= 0:
            obs[S_LAST_ACTION.start + self._last_action] = 1.0
            obs[I_IS_FIRST_STEP] = 0.0
        else:
            obs[I_IS_FIRST_STEP] = 1.0

        n_path = len(self.path)
        if n_path >= 2:
            recent = min(5, n_path - 1)
            tangent = np.zeros(3, dtype=np.float64)
            for j in range(n_path - recent, n_path):
                tangent += (self.path[j] - self.path[j - 1]).astype(np.float64)
            norm = float(np.linalg.norm(tangent))
            obs[S_PATH_TANGENT] = tangent / norm if norm > 0 else tangent

        sdf_agent = self._get_sdf(self.agent_cell)
        for i, d in enumerate(FACE_DIRS):
            obs[S_SDF_GRADIENT.start + i] = (
                self._get_sdf(self.agent_cell + d) - sdf_agent
            ) / SDF_MAX_CELLS

        obs[S_START_DIR] = FACE_DIRS[self.start_dir_idx].astype(np.float64)
        obs[S_GOAL_DIR]  = FACE_DIRS[self.goal_dir_idx].astype(np.float64)
        obs[I_GOAL_DIR_IDX] = self.goal_dir_idx / (N_FACE_DIRS - 1)

        # obs[98:150] zero-padding (already zero)
        return obs

    # ─────────────────────────────────────────────────────────────────────────
    # Baseline reward (CLAUDE.md §16.3.1)
    # ─────────────────────────────────────────────────────────────────────────

    def _calc_reward(self, action: int, terminated: bool, truncated: bool) -> float:
        """Baseline reward. CLAUDE.md §16.3.1 (Phase 1 초기값 w1~w5).

        PBS shaping 은 BaseEnv.step() 에서 _calc_shape() 로 별도 추가.
        """
        reward = -W1  # length penalty (매 스텝)

        if terminated and self._termination_reason in ("collision", "out_of_bounds"):
            reward -= W2

        if self._termination_reason == "goal_reached":
            reward += W3

        if self._step_count == 1:
            reward += W4 if action == self.start_dir_idx else -W5

        if self._termination_reason == "goal_reached" and action != self.goal_dir_idx:
            reward -= W5

        return reward
