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
    ask_waiting_trials,
    create_study,
    enqueue_round1_grid,
    enqueue_round2_grid,
    get_best_params,
    get_param_importances,
    register_grid_params,
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


# ─── 5. ask_waiting_trials / register_grid_params (enqueue → tell 연동) ──────
#
# 회귀 배경: enqueue_trial() 로 등록한 WAITING trial 을 study.tell() 로 바로
# 완료 처리할 수 없다 (실측: ValueError "Cannot tell a WAITING trial.").
# ask_waiting_trials() 로 먼저 RUNNING 전환해야 하고, suggest_* 를 한 번도
# 안 부르면 trial.params 가 비어 get_param_importances() 가 조용히 빈 dict 를
# 반환한다 — register_grid_params() 가 이를 채운다.

def test_tell_on_waiting_trial_raises(storage):
    """WAITING trial 에 직접 study.tell() 하면 실패한다 (Optuna 실측 동작).

    ask_waiting_trials() 가 왜 필요한지에 대한 회귀 가드.
    """
    study = create_study(step_n=1, round_n=1, storage=storage)
    study.enqueue_trial({"alpha": 0.5})
    waiting = [t for t in study.trials if t.state == optuna.trial.TrialState.WAITING]
    with pytest.raises(ValueError):
        study.tell(waiting[0].number, 0.5)


def test_ask_waiting_trials_transitions_to_running(storage):
    study = create_study(step_n=1, round_n=1, storage=storage)
    enqueue_round1_grid(study)
    trials = ask_waiting_trials(study)
    assert len(trials) == 12
    assert all(
        study.trials[t.number].state == optuna.trial.TrialState.RUNNING
        for t in trials
    )
    assert not any(t.state == optuna.trial.TrialState.WAITING for t in study.trials)


def test_ask_waiting_trials_preserves_fixed_params(storage):
    study = create_study(step_n=1, round_n=1, storage=storage)
    study.enqueue_trial({"alpha": 2.0, "beta": 0.3})
    trials = ask_waiting_trials(study)
    assert trials[0].system_attrs["fixed_params"] == {"alpha": 2.0, "beta": 0.3}


def test_ask_then_tell_completes_trial(storage):
    """ask() 로 RUNNING 전환 후에는 tell() 이 정상적으로 COMPLETE 처리한다."""
    study = create_study(step_n=1, round_n=1, storage=storage)
    study.enqueue_trial({"alpha": 0.5})
    trial = ask_waiting_trials(study)[0]
    study.tell(trial.number, 0.75)
    assert study.trials[trial.number].state == optuna.trial.TrialState.COMPLETE
    assert study.trials[trial.number].value == pytest.approx(0.75)


def test_ask_then_tell_accepts_neg_inf(storage):
    """실패 variant 의 metric=-inf 도 tell() 이 그대로 받아들인다 (StageRunner 연동)."""
    study = create_study(step_n=1, round_n=1, storage=storage)
    study.enqueue_trial({"alpha": 0.5})
    trial = ask_waiting_trials(study)[0]
    study.tell(trial.number, float("-inf"))
    assert study.trials[trial.number].state == optuna.trial.TrialState.COMPLETE
    assert study.trials[trial.number].value == float("-inf")


def test_register_grid_params_populates_trial_params(storage):
    """register_grid_params() 이후 trial.params 에 enqueue 했던 값이 그대로 들어간다."""
    study = create_study(step_n=1, round_n=1, storage=storage)
    enqueue_round1_grid(study)
    trials = ask_waiting_trials(study)
    assert all(t.params == {} for t in trials)  # register 전에는 비어있음 (실측)

    register_grid_params(trials)
    for t in trials:
        fp = t.system_attrs["fixed_params"]
        assert t.params["alpha"] == fp["alpha"]
        assert t.params["beta"] == fp["beta"]


def test_register_grid_params_enables_importance(storage):
    """register_grid_params() 없이는 importance 가 항상 {} 이고, 있으면 계산된다.

    §16.6.5: Round 종료 시 어떤 인자가 결과를 가장 설명하는지 정량 추출해야
    Round 2 sweep 대상을 정할 수 있다. suggest_* 미호출 상태로는 이게
    구조적으로 불가능함을 회귀 가드로 남긴다.
    """
    def run(with_register: bool) -> dict:
        study = create_study(
            step_n=1, round_n=1 if with_register else 2, storage=storage,
        )
        enqueue_round1_grid(study)
        trials = ask_waiting_trials(study)
        if with_register:
            register_grid_params(trials)
        for t in trials:
            fp = t.system_attrs["fixed_params"]
            # alpha 가 beta 보다 압도적으로 중요하도록 결정적 점수 부여
            study.tell(t.number, fp["alpha"] * 10 + fp["beta"])
        return get_param_importances(study)

    without = run(with_register=False)
    assert without == {}, "suggest_* 없이는 importance 가 계산되면 안 됨 (실측 확인된 제약)"

    with_reg = run(with_register=True)
    assert with_reg, "register_grid_params() 이후에는 importance 가 계산돼야 함"
    assert "alpha" in with_reg and "beta" in with_reg
    assert with_reg["alpha"] > with_reg["beta"]


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
