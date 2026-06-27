"""Base environment — CLAUDE.md §2.1, §4.3, §9, §12.4, §12.5.

Step 1~10 공통 환경 base class.
reward / observation / action_mask 는 하위 클래스에서 구현.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
from gymnasium import Env, spaces

# ─────────────────────────────────────────────────────────────────────────────
# 절대 변경 금지 상수 (SKILL.md §3.1)
# ─────────────────────────────────────────────────────────────────────────────
GRID_SHAPE: tuple[int, int, int] = (30, 30, 30)  # (nx, ny, nz) cells
N_ACTIONS: int = 7  # 7-direction discrete
OBS_DIM: int = 150  # zero-padding, Step 1~10 고정

# Face 방향 인덱스 (CLAUDE.md §12.4) — 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z
FACE_DIRS: npt.NDArray[np.int32] = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
    dtype=np.int32,
)
N_FACE_DIRS: int = 6

# 기본 공간 크기 (CLAUDE.md §10.1 예시값)
DEFAULT_SPACE_MM: npt.NDArray[np.float64] = np.array(
    [20_000.0, 15_000.0, 5_000.0], dtype=np.float64
)

# ─────────────────────────────────────────────────────────────────────────────
# SDF 관련 상수 및 인접 방향 배열 (CLAUDE.md §4.3)
# ─────────────────────────────────────────────────────────────────────────────
SDF_MAX_CELLS: float = 52.0  # sqrt(30^2 * 3) ≈ 51.96, uint8 정규화 상한

# 26-neighbor 중 edge(L1=2) / vertex(L1=3) 방향
EDGE_DIRS: npt.NDArray[np.int32] = np.array(
    [
        [i, j, k]
        for i in (-1, 0, 1)
        for j in (-1, 0, 1)
        for k in (-1, 0, 1)
        if abs(i) + abs(j) + abs(k) == 2
    ],
    dtype=np.int32,
)  # shape (12, 3)

VERTEX_DIRS: npt.NDArray[np.int32] = np.array(
    [[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)],
    dtype=np.int32,
)  # shape (8, 3)

# ─────────────────────────────────────────────────────────────────────────────
# JIS 파이프 규격 테이블 (CLAUDE.md §4.1)
# ─────────────────────────────────────────────────────────────────────────────
JIS_NOMINAL_SIZES: tuple[str, ...] = (
    "15A", "20A", "25A", "32A", "40A", "50A", "65A", "80A", "100A",
    "125A", "150A", "200A", "250A", "300A", "350A", "400A", "450A", "500A",
)
# 보온재(50mm) 포함 유효 반경 mm (CLAUDE.md §4.1 표)
JIS_EFF_RADIUS_MM: npt.NDArray[np.float64] = np.array(
    [
        60.85, 63.60, 67.00, 71.35, 74.30, 80.25, 88.15, 94.55,
        107.15, 119.90, 132.60, 158.15, 183.70, 209.25,
        227.80, 253.20, 278.60, 304.00,
    ],
    dtype=np.float64,
)
JIS_EFF_RADIUS_NORM: float = 320.0  # 정규화 상수 (max 304.0 초과)
N_JIS_SIZES: int = len(JIS_NOMINAL_SIZES)  # 18

# ─────────────────────────────────────────────────────────────────────────────
# obs[0:98] 슬롯 정의 (CLAUDE.md §4.3, Step 1 active range = obs[0:98])
#
# Slot map (Step 1 기준):
#   [0:3]   agent 정규화 위치 (x/nx, y/ny, z/nz)
#   [3:6]   goal 정규화 위치
#   [6:9]   agent→goal 단위 벡터
#   [9]     Manhattan dist / d_initial
#   [10:16] SDF face-adjacent 6셀 (정규화)
#   [16:28] SDF edge-adjacent 12셀 (정규화)
#   [28:36] SDF vertex-adjacent 8셀 (정규화)
#   [36:42] Raycast 6 FACE_DIRS (정규화)
#   [42:48] 충돌 플래그 6 FACE_DIRS (binary)
#   [48]    step_count / max_steps
#   [49]    path_length / max_steps
#   [50]    pipe nominal_size_idx / 17.0
#   [51]    pipe has_insulation
#   [52]    pipe effective_radius / JIS_EFF_RADIUS_NORM
#   [53:61] pipe type one-hot T1..T8
#   [61]    pipe gravity
#   [62:69] last action one-hot (7-dim)
#   [69]    is_first_step (reset 직후 = 1.0)
#   [70:73] path tangent (최근 5스텝 평균 방향)
#   [73:79] SDF gradient in 6 FACE_DIRS
#   [79:91] Step 1 reserved (zero)
#   [91:94] start_dir unit vector (CLAUDE.md §12.4)
#   [94:97] goal_dir unit vector (CLAUDE.md §12.4)
#   [97]    goal_dir_idx / 5.0
# ─────────────────────────────────────────────────────────────────────────────
S_AGENT_POS = slice(0, 3)
S_GOAL_POS = slice(3, 6)
S_TO_GOAL_VEC = slice(6, 9)
I_GOAL_DIST = 9
S_SDF_FACE = slice(10, 16)
S_SDF_EDGE = slice(16, 28)
S_SDF_VERTEX = slice(28, 36)
S_RAYCAST = slice(36, 42)
S_COLLISION = slice(42, 48)
I_STEP_PROGRESS = 48
I_PATH_PROGRESS = 49
I_PIPE_SIZE = 50
I_PIPE_INSULATION = 51
I_PIPE_RADIUS = 52
S_PIPE_TYPE = slice(53, 61)
I_PIPE_GRAVITY = 61
S_LAST_ACTION = slice(62, 69)
I_IS_FIRST_STEP = 69
S_PATH_TANGENT = slice(70, 73)
S_SDF_GRADIENT = slice(73, 79)
# obs[79:91] Step 1 reserved
S_START_DIR = slice(91, 94)  # CLAUDE.md §12.4
S_GOAL_DIR = slice(94, 97)   # CLAUDE.md §12.4
I_GOAL_DIR_IDX = 97
# obs[98:136] Step 2~6 slots (zero in Step 1)
# obs[136:150] multi-pipe reserve (zero in Phase 1)
OBS_STEP1_ACTIVE_END: int = 98
OBS_RESERVE_START: int = 136


# ─────────────────────────────────────────────────────────────────────────────
# 좌표 변환 함수 (CLAUDE.md §9, L-C4)
# ─────────────────────────────────────────────────────────────────────────────


def cell_to_mm(
    cell: npt.ArrayLike,
    cell_size_mm: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Cell 인덱스 → mm 좌표 (셀 중심점). CLAUDE.md §9."""
    return (np.asarray(cell, dtype=np.float64) + 0.5) * cell_size_mm


def mm_to_cell(
    pos_mm: npt.ArrayLike,
    cell_size_mm: npt.NDArray[np.float64],
    grid_shape: tuple[int, int, int] = GRID_SHAPE,
) -> npt.NDArray[np.int32]:
    """mm 좌표 → Cell 인덱스. 범위 밖이면 ValueError. CLAUDE.md §9, L-C4."""
    pos = np.asarray(pos_mm, dtype=np.float64)
    cell = np.floor(pos / cell_size_mm).astype(np.int32)
    shape = np.array(grid_shape, dtype=np.int32)
    if np.any(cell < 0) or np.any(cell >= shape):
        raise ValueError(
            f"pos_mm {pos} → cell {cell} 가 grid_shape {grid_shape} 범위 밖. "
            "음수 좌표나 공간 크기 초과는 거부됨."
        )
    return cell


# ─────────────────────────────────────────────────────────────────────────────
# Base environment
# ─────────────────────────────────────────────────────────────────────────────


class BaseEnv(Env, ABC):
    """Step 1~10 공통 base environment. CLAUDE.md §2.1, §12.4, §12.5.

    Subclass responsibility:
        _sample_scenario() : start/goal/occupancy grid 초기화 (단계 3).
        _get_obs()         : 150-dim observation 반환 (단계 2).
        _calc_reward()     : reward 계산 (단계 2).
        _get_action_mask() : 7-direction mask 반환 (단계 3).
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(
        self,
        space_size_mm: npt.ArrayLike | None = None,
        max_steps: int = 500,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        self.space_size_mm: npt.NDArray[np.float64] = (
            DEFAULT_SPACE_MM.copy()
            if space_size_mm is None
            else np.asarray(space_size_mm, dtype=np.float64)
        )
        if self.space_size_mm.shape != (3,):
            raise ValueError(
                f"space_size_mm must be shape (3,), got {self.space_size_mm.shape}"
            )

        self.grid_shape: tuple[int, int, int] = GRID_SHAPE  # SKILL §3.1 절대 변경 금지
        self.cell_size_mm: npt.NDArray[np.float64] = self.space_size_mm / np.array(
            self.grid_shape, dtype=np.float64
        )
        self.max_steps: int = max_steps

        # observation_space / action_space (SKILL §3.1 절대 변경 금지)
        self.observation_space: spaces.Box = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float64
        )
        self.action_space: spaces.Discrete = spaces.Discrete(N_ACTIONS)

        # 에피소드 상태 (reset() 이후 유효)
        self.occupancy: npt.NDArray[np.bool_] = np.zeros(self.grid_shape, dtype=np.bool_)
        self.sdf_uint8: npt.NDArray[np.uint8] = np.zeros(self.grid_shape, dtype=np.uint8)
        self.agent_cell: npt.NDArray[np.int32] = np.zeros(3, dtype=np.int32)
        self.goal_cell: npt.NDArray[np.int32] = np.zeros(3, dtype=np.int32)
        self.start_dir_idx: int = 0
        self.goal_dir_idx: int = 2
        self.path: list[npt.NDArray[np.int32]] = []
        self._step_count: int = 0
        self.d_initial: float = 1.0  # initial Manhattan dist agent→goal
        self._last_action: int = -1  # -1 = reset 직후 (no action yet)
        self._termination_reason: str = ""  # _calc_reward() 에서 참조

        # 파이프 속성 (에피소드별 고정, _sample_scenario() 에서 설정)
        self.pipe_nominal_size_idx: int = 8    # default 100A
        self.pipe_has_insulation: bool = True
        self.pipe_effective_radius_mm: float = JIS_EFF_RADIUS_MM[8]
        self.pipe_type_idx: int = 0            # 0=T1 .. 7=T8
        self.pipe_gravity: bool = False

        # CLAUDE.md §11.0.6: PCG64 (default_rng) 만 사용
        self._episode_rng: np.random.Generator = np.random.default_rng(seed)

        # PBS shaping 용 이전 obs (reset/step 에서 갱신, CLAUDE.md §16.7)
        self._prev_obs: npt.NDArray[np.float64] = np.zeros(OBS_DIM, dtype=np.float64)
        # _sample_scenario() 에 외부 seed 전달 (reset(seed=N) 시 설정)
        self._forced_scenario_seed: int | None = None

        # 회귀 감시 callback 자리 (단계 4, SKILL §3.7, CLAUDE.md §12.3)
        self._regression_callback: Any | None = None

    # ── 좌표 변환 ─────────────────────────────────────────────────────────────

    def cell_to_mm(self, cell: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Instance 메서드 편의 래퍼. CLAUDE.md §9."""
        return cell_to_mm(cell, self.cell_size_mm)

    def mm_to_cell(self, pos_mm: npt.ArrayLike) -> npt.NDArray[np.int32]:
        """Instance 메서드 편의 래퍼. CLAUDE.md §9, L-C4."""
        return mm_to_cell(pos_mm, self.cell_size_mm, self.grid_shape)

    # ── SDF 계산 및 조회 (CLAUDE.md §4.3) ──────────────────────────────────────

    def _compute_full_sdf(self) -> None:
        """occupancy 에 대한 전체 EDT 계산 → sdf_uint8 저장. CLAUDE.md §4.3.

        에피소드 reset 시 1회 실행. float32 대신 uint8 로 양자화해 메모리 절감.

        scipy.ndimage.distance_transform_edt(input) 는
        "각 nonzero(True) cell 의 nearest zero(False) cell 까지의 거리" 를 반환한다.
        즉, True=foreground 가 거리를 받고, False=background 가 참조점이다.

        우리가 원하는 것: 각 free cell 이 nearest obstacle(또는 그리드 경계)까지의 거리.
        → ~occupancy 를 1셀 두께의 False(=obstacle) 경계로 패딩:
          True(free) → 거리 계산 대상, False(obstacle/border) → 참조 (거리=0).
        """
        from scipy.ndimage import distance_transform_edt  # 런타임 import (Colab 호환)

        # ~occupancy: True at free cells, False at obstacles
        # pad with False (= implicit boundary obstacle, 1-cell thick)
        padded = np.pad(~self.occupancy, pad_width=1, mode="constant", constant_values=False)
        sdf_full = distance_transform_edt(padded)
        # remove padding and clip to [0, SDF_MAX_CELLS]
        sdf_inner = sdf_full[1:-1, 1:-1, 1:-1]
        clipped = np.clip(sdf_inner / SDF_MAX_CELLS * 255.0, 0, 255)
        self.sdf_uint8 = clipped.astype(np.uint8)

    def _get_sdf(self, cell: npt.NDArray[np.int32]) -> float:
        """cell 의 SDF 값을 cell 단위 float 로 반환. 범위 밖이면 0.0."""
        if not self.in_bounds(cell):
            return 0.0
        return float(self.sdf_uint8[tuple(cell)]) / 255.0 * SDF_MAX_CELLS

    def _raycast(self, direction: npt.NDArray[np.int32]) -> float:
        """지정 방향으로 obstacle 까지의 step 수 반환 (최대 = max grid dim)."""
        pos = self.agent_cell.copy()
        max_dist = max(self.grid_shape)
        for steps in range(max_dist):
            next_pos = pos + direction
            if self.is_occupied(next_pos):
                return float(steps)
            pos = next_pos
        return float(max_dist)

    # ── Face 방향 제약 (CLAUDE.md §12.4) ──────────────────────────────────────

    @staticmethod
    def sample_start_goal_dirs(rng: np.random.Generator) -> tuple[int, int]:
        """에피소드 start/goal face 방향 인덱스 샘플링. CLAUDE.md §12.4.

        배관 공학 원칙: 동일 방향 및 정반대 방향 모두 금지.
        """
        start_dir_idx = int(rng.integers(0, N_FACE_DIRS))
        start_dir_vec = FACE_DIRS[start_dir_idx]
        valid_goal = [
            i
            for i in range(N_FACE_DIRS)
            if i != start_dir_idx and not np.array_equal(FACE_DIRS[i], -start_dir_vec)
        ]
        return start_dir_idx, int(rng.choice(valid_goal))

    # ── 경계/충돌 판정 ─────────────────────────────────────────────────────────

    def in_bounds(self, cell: npt.NDArray[np.int32]) -> bool:
        """Cell 이 grid 범위 [0, grid_shape) 안에 있는지 확인."""
        shape = np.array(self.grid_shape, dtype=np.int32)
        return bool(np.all(cell >= 0) and np.all(cell < shape))

    def is_occupied(self, cell: npt.NDArray[np.int32]) -> bool:
        """Cell 이 장애물로 점유됐는지 확인. 범위 밖은 occupied 로 취급."""
        if not self.in_bounds(cell):
            return True
        return bool(self.occupancy[tuple(cell)])

    # ── Gymnasium 인터페이스 ───────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
        """에피소드 초기화. CLAUDE.md §12.4 start/goal dir 샘플링 포함."""
        super().reset(seed=seed)
        if seed is not None:
            self._episode_rng = np.random.default_rng(seed)

        self._forced_scenario_seed = seed
        self._sample_scenario()
        self._forced_scenario_seed = None

        # episode reset 시 SDF 1회 계산 (CLAUDE.md §4.3)
        self._compute_full_sdf()

        # initial Manhattan distance (PBS 정규화용, CLAUDE.md §16.7.2)
        manhattan = float(np.abs(self.goal_cell - self.agent_cell).sum())
        self.d_initial = manhattan if manhattan > 0.0 else 1.0

        self.start_dir_idx, self.goal_dir_idx = self.sample_start_goal_dirs(
            self._episode_rng
        )
        self.path = [self.agent_cell.copy()]
        self._step_count = 0
        self._last_action = -1
        self._termination_reason = ""

        obs = self._get_obs()
        self._prev_obs = obs  # PBS shaping 초기화 (CLAUDE.md §16.7)
        return obs, {
            "start_cell": self.agent_cell.copy(),
            "goal_cell": self.goal_cell.copy(),
            "start_dir_idx": self.start_dir_idx,
            "goal_dir_idx": self.goal_dir_idx,
            "d_initial": self.d_initial,
        }

    def step(
        self,
        action: int,
    ) -> tuple[npt.NDArray[np.float64], float, bool, bool, dict[str, Any]]:
        """한 스텝 진행. CLAUDE.md §12.4~12.5."""
        self._step_count += 1
        terminated = False
        truncated = False
        self._termination_reason = ""

        if action < N_FACE_DIRS:
            next_cell = self.agent_cell + FACE_DIRS[action]
        else:
            # action 6: macro-action 등 하위 클래스에서 처리 (L-D3)
            next_cell = self.agent_cell.copy()

        if not self.in_bounds(next_cell):
            terminated = True
            self._termination_reason = "out_of_bounds"
        elif self.is_occupied(next_cell):
            terminated = True
            self._termination_reason = "collision"
        else:
            self.agent_cell = next_cell
            self.path.append(self.agent_cell.copy())
            if np.array_equal(self.agent_cell, self.goal_cell):
                terminated = True
                self._termination_reason = "goal_reached"

        if not terminated and self._step_count >= self.max_steps:
            truncated = True
            self._termination_reason = "timeout"

        self._last_action = action
        reward_baseline: float = self._calc_reward(action, terminated, truncated)
        obs = self._get_obs()
        is_terminal: bool = terminated or truncated
        reward_shape: float = self._calc_shape(self._prev_obs, obs, is_terminal)
        self._prev_obs = obs
        return obs, reward_baseline + reward_shape, terminated, truncated, {
            "termination": self._termination_reason,
            "step_count": self._step_count,
            "path_length": len(self.path),
            "reward_baseline": reward_baseline,   # wandb §16.7.8
            "reward_shape": reward_shape,
        }

    def render(self) -> None:
        """시각화는 matplotlib + IPython HTML (SKILL §3.2). 단계 4 이후 구현."""

    def close(self) -> None:
        pass

    # ── Public interface (sb3-contrib MaskablePPO 호환) ─────────────────────────

    def action_masks(self) -> npt.NDArray[np.bool_]:
        """MaskablePPO 호환 public interface. CLAUDE.md §12.5."""
        return self._get_action_mask()

    # ── Shaping (default no-op, Step1Env 에서 override) ─────────────────────────

    def _calc_shape(
        self,
        s: npt.NDArray[np.float64],
        s_next: npt.NDArray[np.float64],
        is_terminal: bool,
    ) -> float:
        """PBS shaping reward. 기본값 0.0 (shaping 없음). CLAUDE.md §16.7."""
        return 0.0

    # ── Abstract methods (하위 클래스 구현) ────────────────────────────────────

    @abstractmethod
    def _sample_scenario(self) -> None:
        """occupancy, agent_cell, goal_cell, pipe_* 을 초기화한다. 단계 3 에서 구현."""

    @abstractmethod
    def _get_obs(self) -> npt.NDArray[np.float64]:
        """150-dim observation 반환. CLAUDE.md §4.3. 단계 2 에서 구현."""

    @abstractmethod
    def _calc_reward(self, action: int, terminated: bool, truncated: bool) -> float:
        """Reward 계산. CLAUDE.md §16.3 + §16.7. 단계 2 에서 구현."""

    @abstractmethod
    def _get_action_mask(self) -> npt.NDArray[np.bool_]:
        """7-direction action mask 반환. CLAUDE.md §12.5. 단계 3 에서 구현."""
