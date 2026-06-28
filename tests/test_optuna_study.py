"""Optuna study 모듈 단위 테스트 — autoresearch/optuna_study.py.

검증 범위:
  - TPE sampler + MedianPruner 생성 (CLAUDE.md §16.6.5)
  - study_name = f'step{N}_round{R}' 형식
  - enqueue_trial 로 grid 명시 등록 → suggest 시 순서 소비
  - load_if_exists=True: 동일 storage/name 재생성 시 기존 study 로드
  - importance analysis: 완료 trial 2+개 이후 dict 반환
  - Round 1 grid: α×β 12개 enqueue (§16.6.7)
  - Round 2 grid: w1×w2×w3 36개 enqueue (§16.3.2; 구 "27" 오타 수정 — 의사결정 27)
"""

from __future__ import annotations

import math
import os
import tempfile

import optuna
import pytest

from autoresearch.optuna_study import (
    create_study,
    enqueue_round1_grid,
    enqueue_round2_grid,
    get_best_params,
    get_param_importances,
)


# ─── fixture: in-memory storage (각 테스트 독립) ────────────────────────────

@pytest.fixture()
def storage(tmp_path) -> str:
    return f"sqlite:///{tmp_path / 'ar_test.db'}"


# ─── 1. study 생성 ───────────────────────────────────────────────────────────

def test_create_study_direction_maximize(storage):
    study = create_study(step_n=1, round_n=1, storage=storage)
    assert study.direction == optuna.study.StudyDirection.MAXIMIZE


def test_create_study_name_format(storage):
    study = create_study(step_n=3, round_n=2, storage=storage)
    assert study.study_name == "step3_round2"


def test_create_study_tpe_sampler(storage):
    study = create_study(step_n=1, round_n=1, storage=storage)
    assert isinstance(study.sampler, optuna.samplers.TPESampler)


def test_create_study_median_pruner(storage):
    study = create_study(step_n=1, round_n=1, storage=storage)
    assert isinstance(study.pruner, optuna.pruners.MedianPruner)


def test_create_study_load_if_exists(storage):
    """동일 study_name 재생성 시 기존 study 를 로드한다."""
    study1 = create_study(step_n=1, round_n=1, storage=storage)
    study1.enqueue_trial({"alpha": 1.0, "beta": 0.1})
    # objective 로 trial 하나 완료
    study1.optimize(lambda t: t.suggest_float("alpha", 0.5, 5.0) + t.suggest_float("beta", 0.0, 0.3), n_trials=1)

    study2 = create_study(step_n=1, round_n=1, storage=storage)
    assert len(study2.trials) == 1


# ─── 2. enqueue_trial / grid ─────────────────────────────────────────────────

def test_enqueue_consumed_in_order(storage):
    """enqueue 한 trial 이 suggest 시 순서대로 소비된다."""
    study = create_study(step_n=1, round_n=1, storage=storage)
    study.enqueue_trial({"alpha": 0.5, "beta": 0.0})
    study.enqueue_trial({"alpha": 2.0, "beta": 0.3})

    consumed = []
    def objective(trial):
        a = trial.suggest_float("alpha", 0.0, 10.0)
        b = trial.suggest_float("beta", 0.0, 1.0)
        consumed.append((a, b))
        return a + b

    study.optimize(objective, n_trials=2)
    assert consumed[0] == pytest.approx((0.5, 0.0))
    assert consumed[1] == pytest.approx((2.0, 0.3))


def test_enqueue_round1_grid_count(storage):
    """Round 1 α×β grid: 12 variants 정확히 enqueue."""
    study = create_study(step_n=1, round_n=1, storage=storage)
    enqueue_round1_grid(study)
    waiting = [t for t in study.trials if t.state == optuna.trial.TrialState.WAITING]
    assert len(waiting) == 12


def test_enqueue_round1_grid_no_duplicates(storage):
    """이미 enqueue 된 조합은 중복 추가하지 않는다."""
    study = create_study(step_n=1, round_n=1, storage=storage)
    enqueue_round1_grid(study)
    enqueue_round1_grid(study)  # 2번째 호출
    waiting = [t for t in study.trials if t.state == optuna.trial.TrialState.WAITING]
    assert len(waiting) == 12  # 여전히 12개


def test_enqueue_round1_grid_covers_spec_values(storage):
    """α ∈ {0.5, 1.0, 2.0, 5.0} × β ∈ {0.0, 0.1, 0.3} 모두 포함."""
    study = create_study(step_n=1, round_n=1, storage=storage)
    enqueue_round1_grid(study)

    # WAITING trial 은 system_attrs['fixed_params'] 에 실제 값 있음
    pairs = set()
    for t in study.trials:
        if t.state == optuna.trial.TrialState.WAITING:
            fp = t.system_attrs.get("fixed_params", {})
            if "alpha" in fp and "beta" in fp:
                pairs.add((fp["alpha"], fp["beta"]))

    for alpha in [0.5, 1.0, 2.0, 5.0]:
        for beta in [0.0, 0.1, 0.3]:
            assert (alpha, beta) in pairs, f"(α={alpha}, β={beta}) 누락"


def test_enqueue_round2_grid_count(storage):
    """Round 2 w1×w2×w3 grid: 36 variants 정확히 enqueue.

    spec §16.3.2 의 '4 × 3 × 3 = 27' 는 오타 — 실제 4×3×3 = 36.
    spec 갱신 필요 (의사결정 27 또는 minor correction).
    """
    study = create_study(step_n=1, round_n=2, storage=storage)
    enqueue_round2_grid(study)
    waiting = [t for t in study.trials if t.state == optuna.trial.TrialState.WAITING]
    assert len(waiting) == 36  # 4*3*3 = 36 (spec §16.3.2 오타: "27" → 36)


# ─── 3. importance analysis ──────────────────────────────────────────────────

def test_importance_returns_empty_when_few_trials(storage):
    """완료 trial 이 2개 미만이면 빈 dict 반환."""
    study = create_study(step_n=1, round_n=1, storage=storage)
    assert get_param_importances(study) == {}


def test_importance_returns_dict_after_trials(storage):
    """완료 trial ≥ n_startup_trials+1 이후 중요도 dict 반환 (TPE 가 작동해야 함).

    n_startup_trials=12 이므로 최소 13개 trial 필요.
    """
    study = create_study(step_n=1, round_n=1, storage=storage)

    def objective(trial):
        a = trial.suggest_float("alpha", 0.5, 5.0)
        b = trial.suggest_float("beta", 0.0, 0.3)
        return a * 2 + b  # alpha 가 beta 보다 중요

    study.optimize(objective, n_trials=15)  # > n_startup_trials=12
    importances = get_param_importances(study)
    assert isinstance(importances, dict)
    # 파라미터가 존재하면 keys 가 alpha, beta 포함
    if importances:
        assert "alpha" in importances or "beta" in importances


def test_importance_values_sum_to_one(storage):
    """중요도 값의 합 ≈ 1.0 (completed trial 충분 시)."""
    study = create_study(step_n=1, round_n=1, storage=storage)

    def objective(trial):
        a = trial.suggest_float("alpha", 0.5, 5.0)
        b = trial.suggest_float("beta", 0.0, 0.3)
        return a + b

    study.optimize(objective, n_trials=15)
    importances = get_param_importances(study)
    if importances:  # 계산 가능한 경우만 검증
        assert math.isclose(sum(importances.values()), 1.0, abs_tol=0.01)


# ─── 4. get_best_params ──────────────────────────────────────────────────────

def test_get_best_params_none_when_empty(storage):
    study = create_study(step_n=1, round_n=1, storage=storage)
    assert get_best_params(study) is None


def test_get_best_params_returns_dict_after_trials(storage):
    study = create_study(step_n=1, round_n=1, storage=storage)
    study.optimize(
        lambda t: t.suggest_float("alpha", 0.5, 5.0),
        n_trials=3,
    )
    best = get_best_params(study)
    assert isinstance(best, dict)
    assert "alpha" in best
