"""Step 1 environment — CLAUDE.md §2.1, Step 1 (장애물 회피).

Phase 1 학습 대상. observation / reward / action_mask 는 단계 2~3 에서 구현.
시나리오 샘플링은 단계 3 (generator 연결) 에서 구현.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from envs.base_env import BaseEnv


class Step1Env(BaseEnv):
    """Step 1 (장애물 회피) 환경. CLAUDE.md Step 1 정의 (§11.2, §11.0.3).

    현재 skeleton 상태:
        _sample_scenario() : NotImplementedError — 단계 3, generator 연결.
        _get_obs()         : NotImplementedError — 단계 2, obs[0:98] 구조 (§4.3).
        _calc_reward()     : NotImplementedError — 단계 2, §16.3.1 + §16.7.
        _get_action_mask() : NotImplementedError — 단계 3, §12.5 마스킹 규칙.
    """

    def _sample_scenario(self) -> None:
        """Step 1 시나리오 초기화. CLAUDE.md §11.0.8 generator 연결 (단계 3).

        단계 3 에서 ScenarioGenerator 를 호출해 occupancy, agent_cell, goal_cell 설정.
        """
        raise NotImplementedError(
            "_sample_scenario() 는 단계 3 (generator 연결) 에서 구현 예정."
        )

    def _get_obs(self) -> npt.NDArray[np.float64]:
        """150-dim observation 반환. CLAUDE.md §4.3 obs[0:98]. 단계 2 에서 구현."""
        raise NotImplementedError(
            "_get_obs() 는 단계 2 (observation 구조 결정) 에서 구현 예정."
        )

    def _calc_reward(self, action: int, terminated: bool, truncated: bool) -> float:
        """Step 1 reward. CLAUDE.md §16.3.1 + §16.7. 단계 2 에서 구현."""
        raise NotImplementedError(
            "_calc_reward() 는 단계 2 (reward function 구현) 에서 구현 예정."
        )

    def _get_action_mask(self) -> npt.NDArray[np.bool_]:
        """7-direction action mask. CLAUDE.md §12.5 Step 1 마스킹 규칙. 단계 3 에서 구현."""
        raise NotImplementedError(
            "_get_action_mask() 는 단계 3 (action mask 구현) 에서 구현 예정."
        )
