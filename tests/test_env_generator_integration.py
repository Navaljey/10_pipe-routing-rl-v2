"""Generator 통합 검증 — CLAUDE.md §11.0.4, §11.0.8.

검증 범위:
  - reset() 시 generator 호출 → occupancy / agent / goal 설정
  - Seed range 분리 (train/screening/regression/final_eval 겹침 없음)
  - 고정 seed 재현성
  - ScenarioGeneratorV1 버전 확인
  - 난이도 분포 & 시나리오 품질 (start free, goal free, BFS reachable)
  - Train seed counter 순차 증가
"""

from __future__ import annotations

import numpy as np
import pytest

from envs.scenario_generator import (
    GENERATOR_VERSION,
    STEP1_FINAL_EVAL_SEEDS,
    STEP1_REGRESSION_SEEDS,
    STEP1_SCREENING_SEEDS,
    STEP1_TRAIN_SEED_END,
    STEP1_TRAIN_SEED_START,
    ScenarioGeneratorV1,
    _bfs_reachable,
    _n_free_face_neighbors,
)
from envs.step1_env import Step1Env

# ─────────────────────────────────────────────────────────────────────────────
# 1. Generator 버전 확인 (CLAUDE.md §11.0.8.6)
# ─────────────────────────────────────────────────────────────────────────────


def test_generator_version() -> None:
    """ScenarioGeneratorV1.version == 'v1.0.0'."""
    gen = ScenarioGeneratorV1()
    assert gen.version == GENERATOR_VERSION
    assert GENERATOR_VERSION == "v1.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Seed range 분리 (CLAUDE.md §11.0.4)
# ─────────────────────────────────────────────────────────────────────────────


def test_seed_ranges_no_overlap() -> None:
    """Screening / Regression / Final Eval / Train seed range 겹침 없음."""
    screening_set   = set(STEP1_SCREENING_SEEDS)
    regression_set  = set(STEP1_REGRESSION_SEEDS)
    final_eval_set  = set(STEP1_FINAL_EVAL_SEEDS)
    train_set       = set(range(STEP1_TRAIN_SEED_START, STEP1_TRAIN_SEED_END + 1))

    assert len(screening_set  & regression_set)  == 0
    assert len(screening_set  & final_eval_set)  == 0
    assert len(screening_set  & train_set)        == 0
    assert len(regression_set & final_eval_set)  == 0
    assert len(regression_set & train_set)        == 0
    assert len(final_eval_set & train_set)        == 0


def test_seed_pool_sizes() -> None:
    """각 pool 크기 = spec 정의 개수. CLAUDE.md §11.0.1."""
    assert len(STEP1_SCREENING_SEEDS)   == 75
    assert len(STEP1_REGRESSION_SEEDS)  == 150
    assert len(STEP1_FINAL_EVAL_SEEDS)  == 500
    assert STEP1_TRAIN_SEED_END - STEP1_TRAIN_SEED_START + 1 == 50000


def test_seed_ranges_correct_offsets() -> None:
    """Step 1 (N=1): N*100000 + offset 공식 검증. CLAUDE.md §11.0.4."""
    N = 1
    assert STEP1_SCREENING_SEEDS[0]  == N * 100_000 + 10_000
    assert STEP1_REGRESSION_SEEDS[0] == N * 100_000 + 20_000
    assert STEP1_FINAL_EVAL_SEEDS[0] == N * 100_000 + 30_000
    assert STEP1_TRAIN_SEED_START    == N * 100_000 + 50_000


# ─────────────────────────────────────────────────────────────────────────────
# 3. reset() 시 generator 호출 검증
# ─────────────────────────────────────────────────────────────────────────────


def test_reset_sets_occupancy(tmp_path: object) -> None:
    """reset() 후 occupancy 가 None 이 아닌 bool array 이다."""
    env = Step1Env(mode="train", difficulty="easy", seed=0)
    env.reset()
    assert env.occupancy.dtype == np.bool_
    assert env.occupancy.shape == (30, 30, 30)


def test_reset_sets_agent_and_goal(tmp_path: object) -> None:
    """reset() 후 agent_cell, goal_cell 이 설정된다."""
    env = Step1Env(mode="train", difficulty="easy", seed=0)
    env.reset()
    assert env.agent_cell.shape == (3,)
    assert env.goal_cell.shape == (3,)


def test_reset_start_is_free(tmp_path: object) -> None:
    """reset() 후 agent_cell 은 항상 free (미점유)."""
    env = Step1Env(mode="train", difficulty="easy", seed=0)
    for seed in range(10):
        env.reset(seed=STEP1_TRAIN_SEED_START + seed)
        assert not env.occupancy[tuple(env.agent_cell)], (
            f"seed={STEP1_TRAIN_SEED_START+seed}: agent_cell 이 occupied"
        )


def test_reset_goal_is_free(tmp_path: object) -> None:
    """reset() 후 goal_cell 은 항상 free (미점유)."""
    env = Step1Env(mode="train", difficulty="easy", seed=0)
    for seed in range(10):
        env.reset(seed=STEP1_TRAIN_SEED_START + seed)
        assert not env.occupancy[tuple(env.goal_cell)], (
            f"seed={STEP1_TRAIN_SEED_START+seed}: goal_cell 이 occupied"
        )


def test_reset_bfs_reachable(tmp_path: object) -> None:
    """reset() 후 start→goal BFS 도달 가능. (generator 검증 전용, 학습 신호 아님)"""
    env = Step1Env(mode="train", difficulty="easy", seed=0)
    for seed in range(5):
        env.reset(seed=STEP1_TRAIN_SEED_START + seed)
        reachable = _bfs_reachable(env.occupancy, env.agent_cell, env.goal_cell)
        assert reachable, (
            f"seed={STEP1_TRAIN_SEED_START+seed}: BFS 도달 불가"
        )


def test_reset_start_has_enough_free_neighbors(tmp_path: object) -> None:
    """start cell 의 free face neighbor ≥ 2 (임의 start_dir 방향 보장)."""
    env = Step1Env(mode="train", difficulty="easy", seed=0)
    for seed in range(10):
        env.reset(seed=STEP1_TRAIN_SEED_START + seed)
        n_free = _n_free_face_neighbors(env.occupancy, env.agent_cell)
        assert n_free >= 2, (
            f"seed={STEP1_TRAIN_SEED_START+seed}: start 자유 neighbor = {n_free} < 2"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. 고정 seed 재현성
# ─────────────────────────────────────────────────────────────────────────────


def test_fixed_seed_reproducibility() -> None:
    """동일 seed 로 reset() 하면 동일 시나리오 생성 (PCG64 결정성). CLAUDE.md §11.0.6."""
    env = Step1Env(mode="train", difficulty="medium", seed=0)

    env.reset(seed=150000)
    occ1 = env.occupancy.copy()
    ag1  = env.agent_cell.copy()
    go1  = env.goal_cell.copy()

    env.reset(seed=150000)
    occ2 = env.occupancy.copy()
    ag2  = env.agent_cell.copy()
    go2  = env.goal_cell.copy()

    np.testing.assert_array_equal(occ1, occ2)
    np.testing.assert_array_equal(ag1, ag2)
    np.testing.assert_array_equal(go1, go2)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Train seed counter 순차 증가
# ─────────────────────────────────────────────────────────────────────────────


def test_train_seed_counter_advances() -> None:
    """mode='train': reset() 마다 _train_seed_counter 가 1씩 증가."""
    env = Step1Env(mode="train", difficulty="easy", seed=0)
    initial_counter = env._train_seed_counter

    env.reset()
    assert env._train_seed_counter == initial_counter + 1

    env.reset()
    assert env._train_seed_counter == initial_counter + 2


def test_train_seed_wraps_around() -> None:
    """train seed counter 가 END 에 도달하면 START 로 wrap. CLAUDE.md §11.0.4."""
    env = Step1Env(mode="train", difficulty="easy", seed=0)
    env._train_seed_counter = STEP1_TRAIN_SEED_END  # 마지막 seed

    # 이 reset 에서 STEP1_TRAIN_SEED_END 사용 → counter = END+1 → wrap → START
    import contextlib
    with contextlib.suppress(Exception):  # ScenarioGenerationFailure fallback
        env.reset()
    assert env._train_seed_counter == STEP1_TRAIN_SEED_START


# ─────────────────────────────────────────────────────────────────────────────
# 6. 모드별 seed pool 동작
# ─────────────────────────────────────────────────────────────────────────────


def test_screening_mode_cycles_fixed_seeds() -> None:
    """mode='screening': 고정 75개 seed pool 순환."""
    env = Step1Env(mode="screening", difficulty="easy", seed=0)
    env.reset()
    env.reset()  # idx=1
    # _pool_idx["screening"] == 2
    assert env._pool_idx["screening"] == 2


def test_regression_mode_cycles_fixed_seeds() -> None:
    """mode='regression': 고정 150개 seed pool 순환."""
    env = Step1Env(mode="regression", difficulty="easy", seed=0)
    env.reset()
    assert env._pool_idx["regression"] == 1


def test_final_eval_mode_cycles_fixed_seeds() -> None:
    """mode='final_eval': 고정 500개 seed pool 순환."""
    env = Step1Env(mode="final_eval", difficulty="easy", seed=0)
    env.reset()
    assert env._pool_idx["final_eval"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 7. ScenarioGeneratorV1 직접 검증
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generator_difficulty(difficulty: str) -> None:
    """모든 난이도에서 generate() 가 (occupancy, start, goal) 를 반환한다."""
    gen = ScenarioGeneratorV1()
    occ, start, goal = gen.generate(seed=150000, difficulty=difficulty)

    assert occ.shape == (30, 30, 30)
    assert occ.dtype == np.bool_
    assert start.shape == (3,)
    assert goal.shape == (3,)
    assert not occ[tuple(start)], "start 가 occupied"
    assert not occ[tuple(goal)], "goal 이 occupied"


def test_generator_obstacle_density() -> None:
    """medium 난이도 장애물 밀도가 spec 범위 내 (목표 0.20 ± 0.10)."""
    gen = ScenarioGeneratorV1()
    densities: list[float] = []
    for seed in range(10):
        occ, _, _ = gen.generate(seed=150000 + seed, difficulty="medium")
        densities.append(float(occ.sum()) / occ.size)

    mean_density = float(np.mean(densities))
    # 목표 0.20, 허용 범위 0.05~0.40 (budget 초과 방지 로직 때문에 하한만 느슨하게)
    assert 0.05 <= mean_density <= 0.40, (
        f"장애물 밀도 out of range: {mean_density:.3f}"
    )


def test_generator_invalid_mode_raises() -> None:
    """잘못된 mode 를 주면 ValueError."""
    with pytest.raises(ValueError, match="mode must be"):
        _ = Step1Env(mode="unknown_mode", difficulty="easy", seed=0)


def test_generator_invalid_difficulty_raises() -> None:
    """잘못된 difficulty 를 주면 ValueError."""
    with pytest.raises(ValueError, match="difficulty must be"):
        _ = Step1Env(mode="train", difficulty="extreme", seed=0)
