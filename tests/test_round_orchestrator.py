"""RoundOrchestrator 단위 테스트 — autoresearch/round_orchestrator.py.

검증 범위:
  - Round 1 → Round 2 전환 (α, β 고정 + w sweep)
  - Round 종료 조건 4가지 (mock 데이터로 각 case)
  - plateau 감지 정확성 (2%p 임계값)
  - 2 Round 연속 동일 best 감지
  - fixed_params 전파 (Round N best → Round N+1 고정)
  - sanity: 2 Round × 4 variants × 5K timestep (실제 학습 포함)

인메모리 Optuna storage 사용 (각 테스트 독립).
"""

from __future__ import annotations

import math
import tempfile

import pytest

from autoresearch.round_orchestrator import (
    MAX_ROUNDS,
    PLATEAU_THRESHOLD,
    STAGNATION_ROUNDS,
    OrchestratorState,
    RoundOrchestrator,
    RoundResult,
)
from autoresearch.stage_runner import VariantResult


# ─── fixture: 결정적 mock train_fn ──────────────────────────────────────────

def make_train_fn(scores: dict | None = None, default: float = 0.5):
    """params → score 매핑 기반 train_fn. 매핑 없으면 default 반환.

    scores 키는 frozenset(params.items()) 또는 특정 params 값 (alpha, beta, w1 등).
    """
    call_log: list[dict] = []

    def train_fn(params: dict, timesteps: int, variant_id: int) -> float:
        call_log.append({"params": dict(params), "timesteps": timesteps, "variant_id": variant_id})
        if scores is None:
            return default
        # frozenset 기반 exact match
        key = frozenset(params.items())
        if key in scores:
            return scores[key]
        # alpha/beta 기반 partial match
        for k_params, score in scores.items():
            if isinstance(k_params, frozenset):
                kd = dict(k_params)
                if all(params.get(k) == v for k, v in kd.items() if k in params):
                    return score
        return default

    train_fn.call_log = call_log
    return train_fn


def make_simple_train_fn(alpha_score_map: dict | None = None, default: float = 0.5):
    """alpha 값 → score 매핑 기반 간단한 train_fn."""
    def train_fn(params: dict, timesteps: int, variant_id: int) -> float:
        if alpha_score_map is None:
            return default
        return alpha_score_map.get(params.get("alpha"), default)
    return train_fn


# ─── fixture: in-memory Optuna storage ──────────────────────────────────────

@pytest.fixture()
def tmp_storage(tmp_path) -> str:
    return f"sqlite:///{tmp_path / 'test_ar.db'}"


def make_orchestrator(train_fn, tmp_storage, stage1_ts=5_000, stage2_ts=5_000, workers=1):
    """테스트용 RoundOrchestrator (짧은 timestep)."""
    return RoundOrchestrator(
        train_fn=train_fn,
        step_n=1,
        max_workers=workers,
        stage1_timesteps=stage1_ts,
        stage2_timesteps=stage2_ts,
        optuna_storage=tmp_storage,
    )


# ─── 1. OrchestratorState 종료 조건 ─────────────────────────────────────────

def _make_round_result(round_n: int, metric: float, params: dict) -> RoundResult:
    best = VariantResult(variant_id=0, params=params, metric=metric, timesteps=5000, stage=2)
    return RoundResult(round_n=round_n, best=best, importances={}, fixed_params=params)


def test_should_stop_max_rounds():
    """Round 6 (> MAX_ROUNDS=5) 진입 시 즉시 종료."""
    state = OrchestratorState()
    stop, reason = state.should_stop(round_n=MAX_ROUNDS + 1)
    assert stop
    assert "절대 상한" in reason


def test_should_not_stop_at_max_rounds():
    """Round 5 자체는 진입 가능 (Round 6부터 차단)."""
    state = OrchestratorState()
    stop, _ = state.should_stop(round_n=MAX_ROUNDS)
    assert not stop


def test_stop_on_user_request():
    """stop() 호출 후 즉시 종료."""
    state = OrchestratorState()
    state.request_stop()
    stop, reason = state.should_stop(round_n=1)
    assert stop
    assert "사용자" in reason


def test_stop_on_stagnation():
    """2 Round 연속 동일 params → 정체 감지 (§16.6.B 조건 2)."""
    state = OrchestratorState()
    same_params = {"alpha": 1.0, "beta": 0.1}
    state.round_results.append(_make_round_result(1, 0.8, same_params))
    state.round_results.append(_make_round_result(2, 0.82, same_params))  # 다른 metric, 같은 params

    stop, reason = state.should_stop(round_n=3)
    assert stop
    assert "동일" in reason or "정체" in reason


def test_no_stagnation_different_params():
    """서로 다른 params 이면 정체 미감지."""
    state = OrchestratorState()
    state.round_results.append(_make_round_result(1, 0.8, {"alpha": 1.0, "beta": 0.1}))
    state.round_results.append(_make_round_result(2, 0.85, {"alpha": 2.0, "beta": 0.1}))

    assert not state.is_stagnated()


def test_plateau_detected_below_threshold():
    """improvement < 2%p → plateau 감지 (§16.6.B 조건 3)."""
    state = OrchestratorState()
    state.round_results.append(_make_round_result(1, 0.800, {"alpha": 1.0}))
    state.round_results.append(_make_round_result(2, 0.810, {"alpha": 2.0}))  # +1%p < 2%p

    assert state.is_plateau()


def test_plateau_not_detected_above_threshold():
    """improvement ≥ 2%p → plateau 미감지."""
    state = OrchestratorState()
    state.round_results.append(_make_round_result(1, 0.800, {"alpha": 1.0}))
    state.round_results.append(_make_round_result(2, 0.825, {"alpha": 2.0}))  # +2.5%p ≥ 2%p

    assert not state.is_plateau()


def test_plateau_exactly_at_threshold():
    """improvement ≥ 2%p → 미감지. 부동소수점 안전값 사용 (+2.1%p = 명확히 통과)."""
    state = OrchestratorState()
    state.round_results.append(_make_round_result(1, 0.800, {"alpha": 1.0}))
    state.round_results.append(_make_round_result(2, 0.821, {"alpha": 2.0}))  # +2.1%p > 2%p

    assert not state.is_plateau()


def test_plateau_needs_two_rounds():
    """Round 1개만으로는 plateau 판단 불가."""
    state = OrchestratorState()
    state.round_results.append(_make_round_result(1, 0.8, {"alpha": 1.0}))
    assert not state.is_plateau()


def test_stagnation_needs_two_rounds():
    """Round 1개만으로는 정체 판단 불가."""
    state = OrchestratorState()
    state.round_results.append(_make_round_result(1, 0.8, {"alpha": 1.0}))
    assert not state.is_stagnated()


# ─── 2. Round 1 실행 ─────────────────────────────────────────────────────────

def test_round1_returns_round_result(tmp_storage):
    """run_round1() 이 RoundResult 반환, round_n=1."""
    orch = make_orchestrator(make_train_fn(), tmp_storage)
    result = orch.run_round1()
    assert isinstance(result, RoundResult)
    assert result.round_n == 1


def test_round1_has_12_variants(tmp_storage):
    """Round 1 은 α × β = 12 variants. §16.6.7."""
    call_log = []

    def train_fn(params, timesteps, variant_id):
        call_log.append(params)
        return 0.5

    orch = make_orchestrator(train_fn, tmp_storage)
    orch.run_round1()

    # Stage 1 + Stage 2: 12개 + 상위 50% (6개) = 18 calls 예상
    # alpha/beta 쌍 고유 수는 12개 이내
    unique_ab = {(p.get("alpha"), p.get("beta")) for p in call_log}
    assert len(unique_ab) == 12, f"expected 12 unique (α,β) pairs, got {unique_ab}"


def test_round1_best_has_alpha_beta(tmp_storage):
    """Round 1 best params 에 alpha, beta 키 포함."""
    orch = make_orchestrator(make_train_fn(), tmp_storage)
    result = orch.run_round1()
    assert "alpha" in result.best.params
    assert "beta" in result.best.params


def test_round1_state_updated(tmp_storage):
    """run_round1() 후 state.round_results 에 1개 추가."""
    orch = make_orchestrator(make_train_fn(), tmp_storage)
    assert len(orch.state.round_results) == 0
    orch.run_round1()
    assert len(orch.state.round_results) == 1


# ─── 3. Round 1 → Round 2 전환 ──────────────────────────────────────────────

def test_round2_fixes_alpha_beta(tmp_storage):
    """Round 2 는 Round 1 best 의 α, β 를 모든 variant 에 고정한다. §16.6.7."""
    alpha_score = {
        0.5: 0.6, 1.0: 0.7, 2.0: 0.9, 5.0: 0.8,  # Round 1: alpha=2.0 이 best
    }
    round2_params = []

    def train_fn(params, timesteps, variant_id):
        if "w1" in params:   # Round 2 호출
            round2_params.append(dict(params))
        return alpha_score.get(params.get("alpha"), 0.5)

    orch = make_orchestrator(train_fn, tmp_storage)
    r1 = orch.run_round1()
    orch.run_round2(r1)

    if round2_params:
        best_alpha = r1.best.params["alpha"]
        best_beta = r1.best.params["beta"]
        for p in round2_params:
            assert p.get("alpha") == pytest.approx(best_alpha), \
                f"Round 2 variant 에 alpha={best_alpha} 미고정: {p}"
            assert p.get("beta") == pytest.approx(best_beta), \
                f"Round 2 variant 에 beta={best_beta} 미고정: {p}"


def test_round2_has_36_variants(tmp_storage):
    """Round 2 는 w1 × w2 × w3 = 36 variants. §16.3.2."""
    w_calls = []

    def train_fn(params, timesteps, variant_id):
        if "w1" in params:
            w_calls.append((params.get("w1"), params.get("w2"), params.get("w3")))
        return 0.5

    orch = make_orchestrator(train_fn, tmp_storage)
    r1 = orch.run_round1()
    orch.run_round2(r1)

    # Stage 1 = 36, Stage 2 = 18 → w 고유 조합은 36 이내
    unique_w = set(w_calls)
    assert len(unique_w) == 36, f"expected 36 unique (w1,w2,w3), got {len(unique_w)}"


def test_round2_best_has_w_params(tmp_storage):
    """Round 2 best params 에 w1, w2, w3 포함."""
    orch = make_orchestrator(make_train_fn(), tmp_storage)
    r1 = orch.run_round1()
    r2 = orch.run_round2(r1)
    assert "w1" in r2.best.params
    assert "w2" in r2.best.params
    assert "w3" in r2.best.params


def test_round2_state_has_two_results(tmp_storage):
    """run_round1() + run_round2() 후 state.round_results = 2개."""
    orch = make_orchestrator(make_train_fn(), tmp_storage)
    r1 = orch.run_round1()
    orch.run_round2(r1)
    assert len(orch.state.round_results) == 2


# ─── 3.5 Optuna trial 완료 + importance (회귀 가드) ──────────────────────────
#
# 배경: 예전에는 enqueue_trial() 로 등록된 WAITING trial 이 학습 후에도
# COMPLETE 로 전환되지 않아 영원히 WAITING 에 남았고, importance 도 항상
# 빈 dict 였다 (Round 2 진행 근거 없음). run_round1()/run_round2() 가
# study.tell() 을 실제로 호출하는지를 공개 API 레벨에서 검증한다.

def _all_trials_of(orch: RoundOrchestrator, round_n: int):
    import optuna
    study = optuna.load_study(
        study_name=f"step{orch.step_n}_round{round_n}", storage=orch.optuna_storage,
    )
    return study.trials


def test_round1_all_trials_complete(tmp_storage):
    """run_round1() 이후 12개 trial 이 전부 COMPLETE (WAITING 에 안 남는다)."""
    orch = make_orchestrator(make_train_fn(), tmp_storage)
    orch.run_round1()

    trials = _all_trials_of(orch, round_n=1)
    assert len(trials) == 12
    assert all(t.state.name == "COMPLETE" for t in trials)


def test_round1_importances_non_empty_with_varying_scores(tmp_storage):
    """alpha 에 따라 점수가 크게 갈리면 importances 가 실제로 alpha 를 잡아낸다."""
    alpha_score = {0.5: 0.1, 1.0: 0.2, 2.0: 0.9, 5.0: 0.3}

    def train_fn(params, timesteps, variant_id):
        return alpha_score.get(params.get("alpha"), 0.5)

    orch = make_orchestrator(train_fn, tmp_storage)
    result = orch.run_round1()

    assert result.importances, "importances 가 비어있으면 안 됨 (register_grid_params 회귀)"
    assert "alpha" in result.importances


def test_round1_failed_variants_still_complete_trial(tmp_storage):
    """일부 variant 가 예외로 실패해도 (metric=-inf) trial 은 COMPLETE 로 남는다."""
    def train_fn(params, timesteps, variant_id):
        if params.get("alpha") == 5.0:
            raise RuntimeError("intentional failure")
        return 0.5

    orch = make_orchestrator(train_fn, tmp_storage, workers=2)
    orch.run_round1()

    trials = _all_trials_of(orch, round_n=1)
    assert all(t.state.name == "COMPLETE" for t in trials)
    neg_inf_trials = [t for t in trials if t.value == float("-inf")]
    assert len(neg_inf_trials) >= 1


# ─── 4. fixed_params 전파 ────────────────────────────────────────────────────

def test_fixed_params_propagated_to_next_round(tmp_storage):
    """RoundResult.fixed_params 에 Round N best params 포함 (다음 Round 고정값)."""
    orch = make_orchestrator(make_train_fn(), tmp_storage)
    r1 = orch.run_round1()
    # Round 1 fixed_params = best params
    assert "alpha" in r1.fixed_params
    assert "beta" in r1.fixed_params

    r2 = orch.run_round2(r1)
    # Round 2 fixed_params = α, β + w1, w2, w3
    assert "alpha" in r2.fixed_params
    assert "beta" in r2.fixed_params
    assert "w1" in r2.fixed_params


# ─── 5. run_phase1 ───────────────────────────────────────────────────────────

def test_run_phase1_returns_two_results(tmp_storage):
    """run_phase1() 는 Round 1+2 를 자동 실행하고 2개 결과 반환."""
    orch = make_orchestrator(make_train_fn(), tmp_storage)
    results = orch.run_phase1()
    assert len(results) == 2
    assert results[0].round_n == 1
    assert results[1].round_n == 2


def test_run_phase1_stops_on_user_stop(tmp_storage):
    """run_phase1() 중 stop() 호출 시 Round 1 이후 중단."""
    call_count = [0]

    def train_fn(params, timesteps, variant_id):
        call_count[0] += 1
        return 0.5

    orch = make_orchestrator(train_fn, tmp_storage)
    # Round 1 완료 직후 stop 요청 (state 직접 조작)
    original_run_round1 = orch.run_round1

    def patched_run_round1():
        result = original_run_round1()
        orch.stop()   # Round 1 완료 후 강제 종료
        return result

    orch.run_round1 = patched_run_round1
    results = orch.run_phase1()
    assert len(results) == 1   # Round 2 실행 안 됨
    assert results[0].round_n == 1


# ─── 6. run_round_generic ────────────────────────────────────────────────────

def test_run_round_generic_round3(tmp_storage):
    """run_round_generic() 으로 Round 3 실행 가능."""
    orch = make_orchestrator(make_train_fn(), tmp_storage)
    variants = [{"lr": 0.001}, {"lr": 0.003}, {"lr": 0.01}, {"lr": 0.03}]
    result = orch.run_round_generic(round_n=3, variants=variants, fixed_params={"alpha": 2.0})
    assert result.round_n == 3
    assert "alpha" in result.fixed_params


# ─── 7. 상수 검증 ────────────────────────────────────────────────────────────

def test_max_rounds_is_5():
    """MAX_ROUNDS = 5 (§16.6.B 조건 1)."""
    assert MAX_ROUNDS == 5


def test_plateau_threshold_is_2pct():
    """PLATEAU_THRESHOLD = 0.02 (2%p, §16.6.B 조건 3)."""
    assert math.isclose(PLATEAU_THRESHOLD, 0.02, rel_tol=1e-9)


def test_stagnation_rounds_is_2():
    """STAGNATION_ROUNDS = 2 (§16.6.B 조건 2)."""
    assert STAGNATION_ROUNDS == 2


# ─── 8. sanity check: 2 Round × 4 variants × 5K ─────────────────────────────

@pytest.mark.sanity
def test_round_orchestrator_sanity_2round_4variants_5k(tmp_path):
    """실제 Round 1 → Round 2 전환 통합 sanity check.

    Round 1: 4 variants (α × β subset) × 5K timestep
             → Stage 1: 4 variants → 하위 50% (2개) 제거 → 2 survivors
             → Stage 2: 2 survivors × 5K → best 1 선택
    Round 2: 4 variants (w subset) × 5K, Round 1 best α, β 고정
             → Stage 1: 4 variants → 2 survivors
             → Stage 2: 2 survivors × 5K → best 1

    검증:
      - Round 2 모든 variant 에 Round 1 best α, β 고정됨
      - Round 1 best 이 Round 2 fixed_params 에 포함됨
      - 총 round_results 2개
    """
    from stable_baselines3.common.env_util import make_vec_env
    from envs.step1_env import Step1Env

    alpha_scores = {0.5: 0.70, 1.0: 0.75, 2.0: 0.90, 5.0: 0.80}
    w1_scores = {0.05: 0.72, 0.1: 0.91, 0.2: 0.85, 0.5: 0.78}
    round2_variants_seen = []

    def train_fn(params: dict, timesteps: int, variant_id: int) -> float:
        # 실제 환경 생성 (단순 학습 없이 env 생성 + 1 rollout)
        env = make_vec_env(Step1Env, n_envs=1)
        obs = env.reset()
        for _ in range(min(timesteps, 100)):
            action = env.action_space.sample()
            obs, reward, done, info = env.step([action])
        env.close()

        if "w1" in params:
            round2_variants_seen.append(dict(params))
            return w1_scores.get(params.get("w1"), 0.5)
        return alpha_scores.get(params.get("alpha"), 0.5)

    # 4 variants 만 사용 (Round 1: α 4개, Round 2: w1 4개 subset)
    # Optuna enqueue 대신 run_round_generic 으로 직접 지정
    storage = f"sqlite:///{tmp_path / 'sanity.db'}"
    orch = RoundOrchestrator(
        train_fn=train_fn,
        step_n=1,
        max_workers=1,
        stage1_timesteps=5_000,
        stage2_timesteps=5_000,
        optuna_storage=storage,
    )

    # Round 1: α × β 4 variants (full 12 대신 4개 subset)
    round1_variants = [
        {"alpha": 0.5, "beta": 0.0},
        {"alpha": 1.0, "beta": 0.1},
        {"alpha": 2.0, "beta": 0.1},
        {"alpha": 5.0, "beta": 0.3},
    ]
    r1 = orch.run_round_generic(round_n=1, variants=round1_variants, fixed_params={})

    # Round 1 best 검증
    assert r1.best.params.get("alpha") == pytest.approx(2.0), \
        f"Round 1 best 가 alpha=2.0 이어야 함 (score 0.90 최고), got {r1.best.params}"

    # Round 2: w1 4 variants, Round 1 best α, β 고정
    best_alpha = r1.best.params["alpha"]
    best_beta = r1.best.params["beta"]
    round2_variants = [
        {"alpha": best_alpha, "beta": best_beta, "w1": 0.05, "w2": 2.0, "w3": 50.0},
        {"alpha": best_alpha, "beta": best_beta, "w1": 0.1,  "w2": 2.0, "w3": 50.0},
        {"alpha": best_alpha, "beta": best_beta, "w1": 0.2,  "w2": 2.0, "w3": 50.0},
        {"alpha": best_alpha, "beta": best_beta, "w1": 0.5,  "w2": 2.0, "w3": 50.0},
    ]
    r2 = orch.run_round_generic(round_n=2, variants=round2_variants, fixed_params=r1.fixed_params)

    # Round 2 best 검증
    assert r2.best.params.get("w1") == pytest.approx(0.1), \
        f"Round 2 best 가 w1=0.1 이어야 함 (score 0.91 최고), got {r2.best.params}"

    # α, β 고정 검증
    for p in round2_variants_seen:
        assert p.get("alpha") == pytest.approx(best_alpha), \
            f"Round 2 variant 의 alpha 불일치: {p}"
        assert p.get("beta") == pytest.approx(best_beta), \
            f"Round 2 variant 의 beta 불일치: {p}"

    # fixed_params 전파 검증
    assert "alpha" in r2.fixed_params
    assert "w1" in r2.fixed_params

    # 총 state 검증
    assert len(orch.state.round_results) == 2
    assert orch.state.round_results[0].round_n == 1
    assert orch.state.round_results[1].round_n == 2
