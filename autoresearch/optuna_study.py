"""Optuna study 생성/관리 — CLAUDE.md §16.6.5.

운영 spec:
  - TPE sampler: n_startup_trials=12 (Stage 1 grid 12개 보장)
  - MedianPruner: plateau 자동 차단
  - sqlite storage: 'sqlite:///autoresearch.db' (영구 보관)
  - study_name: f'step{N}_round{R}'
  - direction: maximize (metric = success_rate)
  - load_if_exists=True: Round 재시작 시 기존 study 이어받기

Round 1 α×β grid (§16.6.7):
  α ∈ {0.5, 1.0, 2.0, 5.0} × β ∈ {0.0, 0.1, 0.3} = 12 variants
  → study.enqueue_trial(params) 로 명시 등록 (grid 보장)
  Stage 2 이후: TPE 가 local search

Importance analysis (§16.6.5):
  Round 종료 시 get_param_importances(study) 로
  "어떤 인자가 결과 분산을 가장 설명하는지" 정량 추출 → 다음 Round sweep 대상 결정
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

logger = logging.getLogger(__name__)

# Stage 1 grid 12개 = TPE 가 grid 소진 전까지 random-like 동작해야 하므로
# n_startup_trials ≥ 12 로 설정 (§16.6.5 spec)
_N_STARTUP_TRIALS = 12

# optuna 기본 INFO 로그가 너무 많으면 WARNING 으로 억제 가능
optuna.logging.set_verbosity(optuna.logging.WARNING)


def create_study(
    step_n: int,
    round_n: int,
    storage: str = "sqlite:///autoresearch.db",
) -> optuna.Study:
    """CLAUDE.md §16.6.5 spec 정합 Optuna study 생성/로드.

    Parameters
    ----------
    step_n  : 현재 Step 번호 (1~10)
    round_n : 현재 Round 번호 (1~5)
    storage : sqlite URL (절대경로 권장: 'sqlite:////abs/path/autoresearch.db')

    Returns
    -------
    study : optuna.Study — direction=maximize, TPE+MedianPruner
    """
    study_name = f"step{step_n}_round{round_n}"
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(n_startup_trials=_N_STARTUP_TRIALS, seed=42),
        pruner=MedianPruner(),
        storage=storage,
        study_name=study_name,
        load_if_exists=True,  # Round 재시작 시 기존 이어받기
    )
    logger.info("[optuna] study '%s' ready (%d trials so far)", study_name, len(study.trials))
    return study


def enqueue_round1_grid(study: optuna.Study) -> None:
    """Round 1 α×β grid 12개를 enqueue. §16.6.7 Phase 1 autoresearch 운영 순서.

    grid:
      α ∈ {0.5, 1.0, 2.0, 5.0}
      β ∈ {0.0, 0.1, 0.3}
      → 4 × 3 = 12 variants

    이미 enqueue / 완료된 trial 이 있으면 중복 enqueue 하지 않는다.
    """
    alpha_values = [0.5, 1.0, 2.0, 5.0]
    beta_values = [0.0, 0.1, 0.3]

    # WAITING trial 은 t.params 가 비어있음 → system_attrs['fixed_params'] 에 실제 값 있음
    def _get_alpha_beta(t) -> tuple | None:
        if t.state == optuna.trial.TrialState.WAITING:
            fp = t.system_attrs.get("fixed_params", {})
            if "alpha" in fp and "beta" in fp:
                return (fp["alpha"], fp["beta"])
        elif "alpha" in t.params and "beta" in t.params:
            return (t.params["alpha"], t.params["beta"])
        return None

    existing_params = {ab for t in study.trials if (ab := _get_alpha_beta(t)) is not None}

    queued = 0
    for alpha in alpha_values:
        for beta in beta_values:
            if (alpha, beta) not in existing_params:
                study.enqueue_trial({"alpha": alpha, "beta": beta})
                queued += 1

    logger.info("[optuna] enqueued %d / 12 Round 1 grid trials", queued)


def enqueue_round2_grid(study: optuna.Study) -> None:
    """Round 2 w1×w2×w3 grid 36개를 enqueue. §16.3.2.

    grid:
      w1 ∈ {0.05, 0.1, 0.2, 0.5}  → 4 values
      w2 ∈ {1.0, 2.0, 5.0}         → 3 values
      w3 ∈ {20, 50, 100}            → 3 values
      → 4 × 3 × 3 = 36 variants (α, β는 Round 1 best 고정)

    Note: 구 spec §16.3.2 는 "4 × 3 × 3 = 27" 로 기록되어 있었으나 오타.
    올바른 값은 36. 의사결정 27 로 모든 관련 파일 수정 완료.
    """
    w1_values = [0.05, 0.1, 0.2, 0.5]
    w2_values = [1.0, 2.0, 5.0]
    w3_values = [20.0, 50.0, 100.0]

    def _get_w_triple(t) -> tuple | None:
        if t.state == optuna.trial.TrialState.WAITING:
            fp = t.system_attrs.get("fixed_params", {})
            if "w1" in fp and "w2" in fp and "w3" in fp:
                return (fp["w1"], fp["w2"], fp["w3"])
        elif "w1" in t.params and "w2" in t.params and "w3" in t.params:
            return (t.params["w1"], t.params["w2"], t.params["w3"])
        return None

    existing_params = {w for t in study.trials if (w := _get_w_triple(t)) is not None}

    queued = 0
    for w1 in w1_values:
        for w2 in w2_values:
            for w3 in w3_values:
                if (w1, w2, w3) not in existing_params:
                    study.enqueue_trial({"w1": w1, "w2": w2, "w3": w3})
                    queued += 1

    logger.info("[optuna] enqueued %d / 36 Round 2 grid trials", queued)


def get_param_importances(study: optuna.Study) -> dict[str, float]:
    """인자 중요도 추출 (Round 종료 시). §16.6.5.

    Returns {'alpha': 0.42, 'beta': 0.28, ...}
    완료된 trial 이 2개 미만이면 빈 dict 반환 (importance 계산 불가).
    """
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < 2:
        logger.warning("[optuna] importance 계산 불가: completed trials=%d (최소 2 필요)", len(completed))
        return {}
    try:
        return optuna.importance.get_param_importances(study)
    except Exception as e:
        logger.warning("[optuna] importance 계산 실패: %s", e)
        return {}


def get_best_params(study: optuna.Study) -> dict[str, Any] | None:
    """study 의 best trial params 반환. 완료된 trial 없으면 None."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return None
    return study.best_trial.params
