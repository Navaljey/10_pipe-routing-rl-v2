"""_evaluate_screening / _evaluate_success_rate 평가 함수 테스트. CLAUDE.md §11.0, §11.2.

배경 (FAILURE_LOG.md 참조): Round 1 Stage 1/2 의 screening 평가가 §11.0 spec
(75개 고정, Easy:Medium:Hard=3:5:2)을 따르지 않고 final_eval 20-episode 를
호출했으며, 그마저도 seed 전달 버그로 20개가 전부 동일 시나리오였다. 본 파일은
그 대체 구현(_evaluate_screening)과 수정된 _evaluate_success_rate 를 검증한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from envs.base_env import FACE_DIRS, S_TO_GOAL_VEC
from training.train_step1 import (
    SCREENING_LENGTH_RATIO_CAP,
    SCREENING_LENGTH_RATIO_EPS,
    SCREENING_N_EPISODES,
    _evaluate_screening,
    _evaluate_success_rate,
)


class _GreedyToGoalModel:
    """테스트 전용 fake model — 허용된 action 중 목표 방향과 내적이 가장 큰 것을 선택.

    실제 정책 학습 없이도 대부분의 medium/easy 시나리오를 빠르게 성공시켜
    (deterministic=True 경로와 동일한 model.predict 인터페이스로) 평가 함수 자체의
    로직을 초 단위로 검증할 수 있게 한다.
    """

    def predict(self, obs, action_masks=None, deterministic=True):
        to_goal = obs[S_TO_GOAL_VEC]
        valid = np.where(action_masks)[0]
        dots = [float(np.dot(FACE_DIRS[a], to_goal)) for a in valid]
        return valid[int(np.argmax(dots))], None


# ─────────────────────────────────────────────────────────────────────────────
# composite score 안전장치 (lexicographic 순서 보존)
# ─────────────────────────────────────────────────────────────────────────────


def test_screening_score_eps_cap_invariant() -> None:
    """length_ratio tie-break 항이 success_rate 최소 격차(1/75)를 절대 못 넘는다."""
    assert SCREENING_LENGTH_RATIO_EPS * SCREENING_LENGTH_RATIO_CAP < 1.0 / SCREENING_N_EPISODES


def test_screening_score_never_flips_success_rate_ordering() -> None:
    """success_rate 가 다르면, length_ratio 가 극단이어도 score 순서가 뒤집히지 않는다."""

    def score(success_rate: float, mean_length_ratio: float) -> float:
        penalty = min(mean_length_ratio, SCREENING_LENGTH_RATIO_CAP)
        return success_rate - SCREENING_LENGTH_RATIO_EPS * penalty

    lower_success_worst_ratio = score(10 / SCREENING_N_EPISODES, 1.0)  # 최선의 length_ratio
    higher_success_best_ratio = score(11 / SCREENING_N_EPISODES, 1000.0)  # 최악의 length_ratio (CAP 이상)
    assert higher_success_best_ratio > lower_success_worst_ratio


# ─────────────────────────────────────────────────────────────────────────────
# _evaluate_screening — 75개 고정, 난이도 3:5:2, length_ratio
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_evaluate_screening_runs_75_episodes_with_difficulty_gradient() -> None:
    """75개(23/37/15) 를 전부 돌고, 난이도가 높을수록 성공률이 같거나 낮다 (건강한 신호)."""
    result = _evaluate_screening(_GreedyToGoalModel(), alpha=1.0, beta=0.1)

    assert result["by_difficulty"]["easy"]["n"] == 23
    assert result["by_difficulty"]["medium"]["n"] == 37
    assert result["by_difficulty"]["hard"]["n"] == 15
    total_success = sum(d["success"] for d in result["by_difficulty"].values())
    assert result["success_rate"] == pytest.approx(total_success / SCREENING_N_EPISODES)

    # greedy-to-goal 은 무작위 정책보다 훨씬 잘 풀지만, 실패 없는 완벽한 정책도 아니므로
    # 0~1 범위와 난이도 단조성(대략적 경향)만 검증한다.
    assert 0.0 <= result["success_rate"] <= 1.0
    easy_rate = result["by_difficulty"]["easy"]["success"] / 23
    hard_rate = result["by_difficulty"]["hard"]["success"] / 15
    assert easy_rate >= hard_rate, "easy 성공률이 hard 보다 낮으면 난이도 축 자체를 의심해야 함"


@pytest.mark.slow
def test_evaluate_screening_length_ratio_near_optimal_for_greedy_policy() -> None:
    """거의 최적 경로를 타는 정책이면 length_ratio 가 1.0 에 가까워야 한다 (BFS benchmark 정합성)."""
    result = _evaluate_screening(_GreedyToGoalModel(), alpha=1.0, beta=0.1)
    assert result["mean_length_ratio"] is not None
    assert 1.0 <= result["mean_length_ratio"] <= 1.5


# ─────────────────────────────────────────────────────────────────────────────
# _evaluate_success_rate — reset(seed=) 수정 이후에도 정상 동작
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_evaluate_success_rate_still_works_after_seed_fix() -> None:
    """수정된 _evaluate_success_rate 가 예외 없이 [0,1] success_rate 를 반환한다."""
    result = _evaluate_success_rate(
        _GreedyToGoalModel(), alpha=1.0, beta=0.1, difficulty="medium", n_episodes=10
    )
    assert 0.0 <= result["success_rate"] <= 1.0
    assert result["mean_ep_length"] > 0
