"""pytest 공유 fixture — 고정 시나리오 Step1Env 구체 구현."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from envs.base_env import JIS_EFF_RADIUS_MM
from envs.step1_env import Step1Env


class ConcreteStep1Env(Step1Env):
    """테스트 전용 concrete Step1Env.

    _sample_scenario() 를 고정 설정으로 구현해 generator 없이 동작.
    장애물 없는 빈 grid, agent(5,5,5), goal(25,25,25).
    """

    def _sample_scenario(self) -> None:
        self.occupancy[:] = False
        self.agent_cell = np.array([5, 5, 5], dtype=np.int32)
        self.goal_cell = np.array([25, 25, 25], dtype=np.int32)
        # T1, 100A, 보온재 있음 (Phase 1 기본)
        self.pipe_nominal_size_idx = 8
        self.pipe_has_insulation = True
        self.pipe_effective_radius_mm = float(JIS_EFF_RADIUS_MM[8])
        self.pipe_type_idx = 0  # T1
        self.pipe_gravity = False

    def _get_action_mask(self) -> npt.NDArray[np.bool_]:
        return np.ones(7, dtype=np.bool_)


@pytest.fixture
def env() -> ConcreteStep1Env:
    return ConcreteStep1Env(seed=42)


@pytest.fixture
def env_with_wall() -> ConcreteStep1Env:
    """agent cell 의 +X 방향이 wall 로 막혀 있는 환경."""

    class _WallEnv(ConcreteStep1Env):
        def _sample_scenario(self) -> None:
            super()._sample_scenario()
            # agent(5,5,5) 의 +X neighbor(6,5,5) 를 obstacle 로
            self.occupancy[6, 5, 5] = True

    return _WallEnv(seed=42)
