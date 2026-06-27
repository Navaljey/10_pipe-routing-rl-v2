"""Base environment — CLAUDE.md §2.1, §4.3, §9, §12.4, §12.5.

Step 1~10 공통 환경 base class.
reward / observation / action_mask 는 하위 클래스에서 구현 (단계 2~3).
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

# Face 방향 인덱스 (CLAUDE.md §12.4)
# 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z
FACE_DIRS: npt.NDArray[np.int32] = np.array(
    [
        [1, 0, 0],  # 0: +X
        [-1, 0, 0],  # 1: -X
        [0, 1, 0],  # 2: +Y
        [0, -1, 0],  # 3: -Y
        [0, 0, 1],  # 4: +Z
        [0, 0, -1],  # 5: -Z
    ],
    dtype=np.int32,
)
N_FACE_DIRS: int = 6  # len(FACE_DIRS)

# 기본 공간 크기 (CLAUDE.md §10.1 예시값, 환경 생성 시 재지정 가능)
DEFAULT_SPACE_MM: npt.NDArray[np.float64] = np.array(
    [20_000.0, 15_000.0, 5_000.0], dtype=np.float64
)


# ─────────────────────────────────────────────────────────────────────────────
# 좌표 변환 함수 (CLAUDE.md §9, L-C4)
# ─────────────────────────────────────────────────────────────────────────────


def cell_to_mm(
    cell: npt.ArrayLike,
    cell_size_mm: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Cell 인덱스 → mm 좌표 (셀 중심점). CLAUDE.md §9.

    Args:
        cell: (3,) int-like, (ix, iy, iz) cell indices.
        cell_size_mm: (3,) float array, per-axis cell size in mm.

    Returns:
        (3,) float64 array, mm coordinates of cell center.
    """
    return (np.asarray(cell, dtype=np.float64) + 0.5) * cell_size_mm


def mm_to_cell(
    pos_mm: npt.ArrayLike,
    cell_size_mm: npt.NDArray[np.float64],
    grid_shape: tuple[int, int, int] = GRID_SHAPE,
) -> npt.NDArray[np.int32]:
    """mm 좌표 → Cell 인덱스. 범위 밖이면 ValueError. CLAUDE.md §9, L-C4.

    Args:
        pos_mm: (3,) float-like, mm coordinates.
        cell_size_mm: (3,) float array, per-axis cell size in mm.
        grid_shape: (nx, ny, nz) grid dimensions (default: GRID_SHAPE).

    Returns:
        (3,) int32 array, (ix, iy, iz) cell indices.

    Raises:
        ValueError: pos_mm 가 grid 밖 (음수 또는 grid_shape 이상).
    """
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
        _sample_scenario() : start/goal/occupancy grid 초기화 (단계 3, generator 연결).
        _get_obs()         : 150-dim observation 반환 (단계 2).
        _calc_reward()     : reward 계산 (단계 2, CLAUDE.md §16.3 + §16.7).
        _get_action_mask() : 7-direction mask 반환 (단계 3, CLAUDE.md §12.5).
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(
        self,
        space_size_mm: npt.ArrayLike | None = None,
        max_steps: int = 500,
        seed: int | None = None,
    ) -> None:
        """환경 초기화.

        Args:
            space_size_mm: (3,) 공간 크기 [mm]. None 이면 DEFAULT_SPACE_MM 사용.
            max_steps: 에피소드 최대 스텝 수 (timeout 조건).
            seed: 결정론적 재현을 위한 RNG seed. CLAUDE.md §11.0.6 PCG64 사용.
        """
        super().__init__()

        self.space_size_mm: npt.NDArray[np.float64] = (
            DEFAULT_SPACE_MM.copy()
            if space_size_mm is None
            else np.asarray(space_size_mm, dtype=np.float64)
        )
        if self.space_size_mm.shape != (3,):
            raise ValueError(f"space_size_mm must be shape (3,), got {self.space_size_mm.shape}")

        self.grid_shape: tuple[int, int, int] = GRID_SHAPE  # SKILL §3.1 절대 변경 금지
        self.cell_size_mm: npt.NDArray[np.float64] = self.space_size_mm / np.array(
            self.grid_shape, dtype=np.float64
        )
        self.max_steps: int = max_steps

        # observation_space / action_space (SKILL §3.1 절대 변경 금지)
        self.observation_space: spaces.Box = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBS_DIM,),
            dtype=np.float64,
        )
        self.action_space: spaces.Discrete = spaces.Discrete(N_ACTIONS)

        # 에피소드 상태 (reset() 이후 유효)
        self.occupancy: npt.NDArray[np.bool_] = np.zeros(self.grid_shape, dtype=np.bool_)
        self.agent_cell: npt.NDArray[np.int32] = np.zeros(3, dtype=np.int32)
        self.goal_cell: npt.NDArray[np.int32] = np.zeros(3, dtype=np.int32)
        self.start_dir_idx: int = 0
        self.goal_dir_idx: int = 2
        self.path: list[npt.NDArray[np.int32]] = []
        self._step_count: int = 0

        # CLAUDE.md §11.0.6: PCG64 (default_rng) 만 사용
        self._episode_rng: np.random.Generator = np.random.default_rng(seed)

        # 회귀 감시 callback 자리 (단계 4 에서 통합. SKILL §3.7, CLAUDE.md §12.3)
        self._regression_callback: Any | None = None

    # ── 좌표 변환 (module-level 함수 위임) ────────────────────────────────────

    def cell_to_mm(self, cell: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Instance 메서드 편의 래퍼. CLAUDE.md §9."""
        return cell_to_mm(cell, self.cell_size_mm)

    def mm_to_cell(self, pos_mm: npt.ArrayLike) -> npt.NDArray[np.int32]:
        """Instance 메서드 편의 래퍼. CLAUDE.md §9, L-C4."""
        return mm_to_cell(pos_mm, self.cell_size_mm, self.grid_shape)

    # ── Face 방향 제약 (CLAUDE.md §12.4) ──────────────────────────────────────

    @staticmethod
    def sample_start_goal_dirs(rng: np.random.Generator) -> tuple[int, int]:
        """에피소드 start/goal face 방향 인덱스를 랜덤 샘플링한다. CLAUDE.md §12.4.

        배관 공학 원칙: start 와 goal 이 동일하거나 정반대 방향이 되면 안 됨.

        Returns:
            (start_dir_idx, goal_dir_idx) — 0~5 범위.
        """
        start_dir_idx = int(rng.integers(0, N_FACE_DIRS))
        start_dir_vec = FACE_DIRS[start_dir_idx]
        valid_goal = [
            i
            for i in range(N_FACE_DIRS)
            if i != start_dir_idx and not np.array_equal(FACE_DIRS[i], -start_dir_vec)
        ]
        goal_dir_idx = int(rng.choice(valid_goal))
        return start_dir_idx, goal_dir_idx

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
        """에피소드 초기화. CLAUDE.md §12.4 start/goal dir 샘플링 포함.

        _sample_scenario() 는 하위 클래스 구현 (단계 3).
        """
        super().reset(seed=seed)
        if seed is not None:
            self._episode_rng = np.random.default_rng(seed)

        self._sample_scenario()

        self.start_dir_idx, self.goal_dir_idx = self.sample_start_goal_dirs(self._episode_rng)
        self.path = [self.agent_cell.copy()]
        self._step_count = 0

        obs = self._get_obs()
        info: dict[str, Any] = {
            "start_cell": self.agent_cell.copy(),
            "goal_cell": self.goal_cell.copy(),
            "start_dir_idx": self.start_dir_idx,
            "goal_dir_idx": self.goal_dir_idx,
        }
        return obs, info

    def step(
        self,
        action: int,
    ) -> tuple[npt.NDArray[np.float64], float, bool, bool, dict[str, Any]]:
        """한 스텝 진행. CLAUDE.md §12.4~12.5.

        Action 0~5 는 FACE_DIRS 이동. Action 6 은 하위 클래스 해석 (CLAUDE.md §2.5, L-D3).
        reward 계산은 _calc_reward() (하위 클래스, 단계 2).
        """
        self._step_count += 1
        terminated = False
        truncated = False
        info: dict[str, Any] = {}

        if action < N_FACE_DIRS:
            next_cell = self.agent_cell + FACE_DIRS[action]
        else:
            # action 6: macro-action 등 하위 클래스에서 처리 (L-D3)
            next_cell = self.agent_cell.copy()

        if not self.in_bounds(next_cell):
            terminated = True
            info["termination"] = "out_of_bounds"
        elif self.is_occupied(next_cell):
            terminated = True
            info["termination"] = "collision"
        else:
            self.agent_cell = next_cell
            self.path.append(self.agent_cell.copy())
            if np.array_equal(self.agent_cell, self.goal_cell):
                terminated = True
                info["termination"] = "goal_reached"

        if not terminated and self._step_count >= self.max_steps:
            truncated = True
            info["termination"] = "timeout"

        reward: float = self._calc_reward(action, terminated, truncated)
        obs = self._get_obs()
        info["step_count"] = self._step_count
        info["path_length"] = len(self.path)
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """시각화는 matplotlib + IPython HTML (SKILL §3.2). 구현은 단계 4 이후."""

    def close(self) -> None:
        pass

    # ── Abstract methods (하위 클래스 구현) ────────────────────────────────────

    @abstractmethod
    def _sample_scenario(self) -> None:
        """occupancy, agent_cell, goal_cell 을 초기화한다. 단계 3 에서 구현.

        CLAUDE.md §11.0.8 generator 를 호출해 occupancy grid, start cell,
        goal cell 을 설정해야 한다.
        """

    @abstractmethod
    def _get_obs(self) -> npt.NDArray[np.float64]:
        """150-dim observation 반환. CLAUDE.md §4.3. 단계 2 에서 구현."""

    @abstractmethod
    def _calc_reward(self, action: int, terminated: bool, truncated: bool) -> float:
        """Reward 계산. CLAUDE.md §16.3 + §16.7. 단계 2 에서 구현."""

    @abstractmethod
    def _get_action_mask(self) -> npt.NDArray[np.bool_]:
        """7-direction action mask 반환. CLAUDE.md §12.5. 단계 3 에서 구현."""
