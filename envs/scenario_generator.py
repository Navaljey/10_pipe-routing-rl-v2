"""Procedural Scenario Generator v1.0.0 — CLAUDE.md §11.0.8.

generator_version = "v1.0.0"
알고리즘:
  - 장애물 배치: Poisson disk sampling (min_dist=3) + Cuboid 합성 (CLAUDE.md §11.0.8.1)
  - Start/Goal: rejection sampling + BFS reachability 검증 (CLAUDE.md §11.0.8.2)
  - 난이도 파라미터: DIFFICULTY_PARAMS dict (CLAUDE.md §11.0.8.3)

Seed range (Step 1, CLAUDE.md §11.0.4):
  Screening:  110000~119999  (75개 사용)
  Regression: 120000~129999  (150개 사용)
  Final Eval: 130000~139999  (500개 사용)
  Train:      150000~199999  (50000슬롯, 순차 사용)
"""

from __future__ import annotations

from collections import deque
from typing import Final

import numpy as np
import numpy.typing as npt

from envs.base_env import FACE_DIRS, GRID_SHAPE

# ─────────────────────────────────────────────────────────────────────────────
# Generator version
# ─────────────────────────────────────────────────────────────────────────────
GENERATOR_VERSION: Final[str] = "v1.0.0"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 seed ranges (CLAUDE.md §11.0.4, N=1)
# ─────────────────────────────────────────────────────────────────────────────
STEP1_SCREENING_SEEDS: Final[list[int]] = list(range(110000, 110075))    # 75개
STEP1_REGRESSION_SEEDS: Final[list[int]] = list(range(120000, 120150))   # 150개
STEP1_FINAL_EVAL_SEEDS: Final[list[int]] = list(range(130000, 130500))   # 500개
STEP1_TRAIN_SEED_START: Final[int] = 150000
STEP1_TRAIN_SEED_END: Final[int] = 199999                                 # 50000슬롯

# ─────────────────────────────────────────────────────────────────────────────
# 난이도 파라미터 (CLAUDE.md §11.0.8.3)
# ─────────────────────────────────────────────────────────────────────────────
DIFFICULTY_PARAMS: Final[dict] = {
    "step1": {
        "easy": {
            "obstacle_density": 0.10,
            "dist_range": (5, 10),
            "ratio_range": (0.9, 1.0),
        },
        "medium": {
            "obstacle_density": 0.20,
            "dist_range": (10, 20),
            "ratio_range": (0.6, 0.9),
        },
        "hard": {
            "obstacle_density": 0.35,
            "dist_range": (20, 30),
            "ratio_range": (0.3, 0.6),
        },
    },
}

# 장애물 타입별 파라미터 (CLAUDE.md §11.0.8.1)
_OBS_TYPES: Final[dict] = {
    "tank":       {"size_range": (5, 12), "weight": 0.4},
    "structural": {"size_range": (2,  4), "weight": 0.3},
    "small_eq":   {"size_range": (2,  5), "weight": 0.3},
}


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────


class ScenarioGenerationFailure(Exception):  # noqa: N818
    """Rejection sampling 이 1000회 이내에 유효한 start/goal 을 찾지 못했을 때."""

    def __init__(self, seed: int, difficulty: str) -> None:
        super().__init__(
            f"seed={seed}, difficulty={difficulty}: "
            "1000 번 시도 후 valid start/goal 생성 실패."
        )
        self.seed = seed
        self.difficulty = difficulty


# ─────────────────────────────────────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────────────────────────────────────


def _poisson_disk_sample_3d(
    grid_size: tuple[int, int, int],
    min_dist: float,
    rng: np.random.Generator,
    max_centers: int = 200,
) -> list[npt.NDArray[np.int32]]:
    """Rejection-based Poisson disk sampling. 최소 거리 min_dist 보장."""
    nx, ny, nz = grid_size
    centers: list[npt.NDArray[np.int32]] = []
    max_tries = max_centers * 30

    for _ in range(max_tries):
        if len(centers) >= max_centers:
            break
        c = np.array(
            [rng.integers(0, nx), rng.integers(0, ny), rng.integers(0, nz)],
            dtype=np.int32,
        )
        if all(float(np.linalg.norm(c - e)) >= min_dist for e in centers):
            centers.append(c)

    return centers


def _place_cuboid(
    grid: npt.NDArray[np.bool_],
    center: npt.NDArray[np.int32],
    size: npt.NDArray[np.int32],
) -> int:
    """Cuboid 를 grid 에 배치하고 실제 점유 voxel 수를 반환."""
    nx, ny, nz = grid.shape
    half = size // 2
    x1 = max(0, int(center[0]) - int(half[0]))
    x2 = min(nx, int(center[0]) + int(size[0]) - int(half[0]))
    y1 = max(0, int(center[1]) - int(half[1]))
    y2 = min(ny, int(center[1]) + int(size[1]) - int(half[1]))
    z1 = max(0, int(center[2]) - int(half[2]))
    z2 = min(nz, int(center[2]) + int(size[2]) - int(half[2]))

    grid[x1:x2, y1:y2, z1:z2] = True
    return (x2 - x1) * (y2 - y1) * (z2 - z1)


def _bfs_reachable(
    occupancy: npt.NDArray[np.bool_],
    start: npt.NDArray[np.int32],
    goal: npt.NDArray[np.int32],
) -> bool:
    """6-connected BFS 도달 가능성 검증. CLAUDE.md §11.0.8.2 "A* reachable 검증 전용"."""
    if np.array_equal(start, goal):
        return True
    shape = occupancy.shape
    visited: set[tuple[int, int, int]] = set()
    queue: deque[tuple[int, int, int]] = deque([tuple(start)])  # type: ignore[arg-type]
    visited.add(tuple(start))  # type: ignore[arg-type]
    goal_t = tuple(goal)

    while queue:
        cx, cy, cz = queue.popleft()
        for d in FACE_DIRS:
            nx_, ny_, nz_ = cx + d[0], cy + d[1], cz + d[2]
            if not (0 <= nx_ < shape[0] and 0 <= ny_ < shape[1] and 0 <= nz_ < shape[2]):
                continue
            if occupancy[nx_, ny_, nz_]:
                continue
            pos = (nx_, ny_, nz_)
            if pos == goal_t:
                return True
            if pos not in visited:
                visited.add(pos)
                queue.append(pos)

    return False


def _n_free_face_neighbors(
    occupancy: npt.NDArray[np.bool_],
    cell: npt.NDArray[np.int32],
) -> int:
    """cell 의 6-face neighbor 중 자유(free) 셀 개수. start/goal 품질 보장용."""
    shape = occupancy.shape
    count = 0
    for d in FACE_DIRS:
        nx_, ny_, nz_ = int(cell[0]) + d[0], int(cell[1]) + d[1], int(cell[2]) + d[2]
        if (
            0 <= nx_ < shape[0] and 0 <= ny_ < shape[1] and 0 <= nz_ < shape[2]
            and not occupancy[nx_, ny_, nz_]
        ):
            count += 1
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Generator v1.0.0
# ─────────────────────────────────────────────────────────────────────────────


class ScenarioGeneratorV1:
    """Procedural Scenario Generator v1.0.0. CLAUDE.md §11.0.8.

    generate(seed, difficulty) → (occupancy, start_cell, goal_cell)

    Notes
    -----
    - 장애물 seed 와 start/goal seed 분리 (CLAUDE.md §11.0.8.2: seed+1 사용).
    - BFS 검증은 학습 신호가 아닌 시나리오 품질 보장 전용.
    - start cell 의 free face neighbor ≥ 2 보장 (임의 start_dir 방향으로 출발 가능).
    - Generator version 변경 규칙: §11.0.8.6 semver 참조.
    """

    version: str = GENERATOR_VERSION

    def generate(
        self,
        seed: int,
        difficulty: str = "medium",
        step: str = "step1",
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """시나리오 생성. (occupancy, start, goal) 반환.

        Parameters
        ----------
        seed :
            에피소드 고유 seed (CLAUDE.md §11.0.4 seed range 에서 선택).
        difficulty :
            "easy" | "medium" | "hard"
        step :
            "step1" (현재 구현, 추후 step2~6 확장 예정)
        """
        params = DIFFICULTY_PARAMS[step][difficulty]
        rng_obs = np.random.default_rng(seed)          # 장애물용 RNG
        rng_sg  = np.random.default_rng(seed + 1)      # start/goal 분리 (§11.0.8.2)

        grid = np.zeros(GRID_SHAPE, dtype=np.bool_)
        occupancy = self._generate_obstacles(grid, params, rng_obs)
        start, goal = self._sample_start_goal(occupancy, params, rng_sg, seed, difficulty)

        return occupancy, start, goal

    # ── obstacle generation (CLAUDE.md §11.0.8.1) ────────────────────────────

    def _generate_obstacles(
        self,
        grid: npt.NDArray[np.bool_],
        params: dict,
        rng: np.random.Generator,
    ) -> npt.NDArray[np.bool_]:
        density: float = params["obstacle_density"]
        budget: int = int(grid.size * density)

        obs_types = list(_OBS_TYPES.keys())
        weights = np.array([_OBS_TYPES[t]["weight"] for t in obs_types], dtype=np.float64)
        weights /= weights.sum()  # normalize (tank:0.4 structural:0.3 small_eq:0.3)

        # Poisson disk sampling (min_dist=3, 장애물 간 corridor 보장)
        centers = _poisson_disk_sample_3d(GRID_SHAPE, min_dist=3.0, rng=rng)

        used = 0
        rng.shuffle(centers)  # type: ignore[arg-type]
        for center in centers:
            if used >= budget:
                break
            obs_type = obs_types[int(rng.choice(len(obs_types), p=weights))]
            lo, hi = _OBS_TYPES[obs_type]["size_range"]
            size = rng.integers(lo, hi + 1, size=3).astype(np.int32)

            # center ± half → don't let the cuboid cover the grid boundary
            half = size // 2
            cx = int(np.clip(center[0], int(half[0]), GRID_SHAPE[0] - 1 - int(size[0] - half[0])))
            cy = int(np.clip(center[1], int(half[1]), GRID_SHAPE[1] - 1 - int(size[1] - half[1])))
            cz = int(np.clip(center[2], int(half[2]), GRID_SHAPE[2] - 1 - int(size[2] - half[2])))
            clipped_center = np.array([cx, cy, cz], dtype=np.int32)

            placed = _place_cuboid(grid, clipped_center, size)
            used += placed

        return grid

    # ── start/goal sampling (CLAUDE.md §11.0.8.2) ─────────────────────────────

    def _sample_start_goal(
        self,
        occupancy: npt.NDArray[np.bool_],
        params: dict,
        rng: np.random.Generator,
        seed: int,
        difficulty: str,
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        dist_lo, dist_hi = params["dist_range"]
        ratio_lo, ratio_hi = params["ratio_range"]

        free_cells = np.argwhere(~occupancy).astype(np.int32)
        if len(free_cells) < 2:
            raise ScenarioGenerationFailure(seed=seed, difficulty=difficulty)

        for _ in range(1000):
            si = int(rng.integers(0, len(free_cells)))
            gi = int(rng.integers(0, len(free_cells)))
            if si == gi:
                continue

            start = free_cells[si]
            goal  = free_cells[gi]

            d_manhattan = int(np.abs(goal - start).sum())
            d_euclidean = float(np.linalg.norm((goal - start).astype(np.float64)))
            ratio       = d_euclidean / d_manhattan if d_manhattan > 0 else 1.0

            if not (dist_lo <= d_manhattan <= dist_hi):
                continue
            if not (ratio_lo <= ratio <= ratio_hi):
                continue

            # start cell: free face neighbor ≥ 2 (임의 start_dir 보장)
            if _n_free_face_neighbors(occupancy, start) < 2:
                continue

            # goal cell: free face neighbor ≥ 1 (goal approach 방향 보장)
            if _n_free_face_neighbors(occupancy, goal) < 1:
                continue

            # BFS reachability (학습 신호 아님, 시나리오 검증 전용)
            if _bfs_reachable(occupancy, start, goal):
                return start, goal

        raise ScenarioGenerationFailure(seed=seed, difficulty=difficulty)
