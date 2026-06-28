"""wandb naming 체계 + sb3 콜백 — CLAUDE.md §16.6.6, SKILL §3.6.

naming 체계:
  project : 'pipe-routing-rl'  (전 Step / 전 Phase 공통)
  name    : f'step{N}_round{R}_stage{S}_var{V:03d}'
  tags    : [stepN, roundR, stageS, autoresearch, phase1 or phase2plus]
  config  : 모든 sweep 대상 + git_commit + generator_version

§16.6.6 진단 panel logging (§16.7.8):
  r_baseline_mean, r_shape_mean, r_shape_to_baseline_ratio
  phi_goal_at_step0, phi_goal_at_terminal, phi_cong_distribution
  → WandbCallback 이 자동 logging 하는 항목 외 추가 metric 은
    학습 loop 에서 직접 wandb.log() 로 보내는 구조.

오프라인 모드:
  WANDB_MODE=offline 또는 mode="offline" 파라미터로 제어.
  Colab 에서 --no-wandb 사용 시 mode="disabled" 로 init → noop.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Any

import wandb

logger = logging.getLogger(__name__)

_PROJECT = "pipe-routing-rl"


# ─── pure naming helpers (테스트 가능, wandb.init 불필요) ─────────────────────

def build_run_name(step_n: int, round_n: int, stage_n: int, variant_id: int) -> str:
    """CLAUDE.md §16.6.6 run name 규칙: step{N}_round{R}_stage{S}_var{V:03d}."""
    return f"step{step_n}_round{round_n}_stage{stage_n}_var{variant_id:03d}"


def build_run_tags(step_n: int, round_n: int, stage_n: int) -> list[str]:
    """CLAUDE.md §16.6.6 tags: [stepN, roundR, stageS, autoresearch, phase1/phase2plus]."""
    return [
        f"step{step_n}",
        f"round{round_n}",
        f"stage{stage_n}",
        "autoresearch",
        "phase1" if step_n == 1 else "phase2plus",
    ]


def build_run_config(
    step_n: int,
    round_n: int,
    stage_n: int,
    variant_id: int,
    params: dict[str, Any],
    git_commit: str,
    generator_version: str,
) -> dict[str, Any]:
    """CLAUDE.md §16.6.6 config: 모든 sweep 대상 + git_commit + generator_version."""
    return {
        "step": step_n,
        "round": round_n,
        "stage": stage_n,
        "variant_id": variant_id,
        **params,
        "git_commit": git_commit,
        "generator_version": generator_version,
    }


def _get_git_commit() -> str:
    """현재 git HEAD commit hash (재현성 용). 실패 시 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def init_wandb_run(
    step_n: int,
    round_n: int,
    stage_n: int,
    variant_id: int,
    params: dict[str, Any],
    generator_version: str,
    *,
    mode: str = "online",
    notes: str = "",
) -> "wandb.sdk.wandb_run.Run":
    """CLAUDE.md §16.6.6 naming 체계로 wandb.init().

    Parameters
    ----------
    step_n, round_n, stage_n, variant_id :
        run name 구성 요소. name = f'step{N}_round{R}_stage{S}_var{V:03d}'
    params :
        sweep 대상 hyperparameter dict — config 에 전체 포함.
        최소 포함 권장: alpha, beta (Round 1) / w1, w2, w3 (Round 2)
    generator_version :
        GENERATOR_VERSION (시나리오 호환성 추적)
    mode :
        'online' | 'offline' | 'disabled'
        offline: Colab 세션 끊김 대비 로컬 저장 후 나중에 sync
        disabled: --no-wandb 시 noop (모든 wandb 호출 무시)
    notes :
        run 메모 (선택)

    Returns
    -------
    wandb.Run — wandb.finish() 는 호출자 책임.
    """
    git_commit = _get_git_commit()
    run_name = build_run_name(step_n, round_n, stage_n, variant_id)
    tags = build_run_tags(step_n, round_n, stage_n)
    config = build_run_config(step_n, round_n, stage_n, variant_id, params, git_commit, generator_version)

    run = wandb.init(
        project=_PROJECT,
        name=run_name,
        config=config,
        tags=tags,
        notes=notes,
        mode=mode,
        reinit=True,
    )
    logger.info("[wandb] init: %s (mode=%s)", run_name, mode)
    return run


def make_wandb_sb3_callback():
    """sb3 WandbCallback 반환. SKILL §3.6 의무.

    wandb.init() 이 먼저 호출된 상태에서 사용.
    wandb.init(mode='disabled') 일 때도 안전하게 noop 동작.
    """
    from wandb.integration.sb3 import WandbCallback

    return WandbCallback(
        gradient_save_freq=0,   # gradient logging 비활성 (속도 우선)
        verbose=0,
    )


def log_stage1_summary(
    step_n: int,
    round_n: int,
    all_results: "list",
    survivors: "list",
) -> None:
    """Stage 1 종료 시점 — 12 variant scores + 생존자 list → wandb summary.

    §16.6.6 진단 panel: Parallel Coordinates Plot 가 의미 있으려면
    모든 variant 의 Stage 1 metric 이 동일 run 에 기록돼야 함.
    오케스트레이터 run (round{R}_orch) 에 summary 로 기록.
    """
    if wandb.run is None:
        return

    # 전체 variant 점수 테이블
    table = wandb.Table(columns=["variant_id", "params", "metric", "survived"])
    survivor_ids = {r.variant_id for r in survivors}
    for r in sorted(all_results, key=lambda x: x.metric, reverse=True):
        table.add_data(
            r.variant_id,
            str(r.params),
            r.metric,
            r.variant_id in survivor_ids,
        )

    wandb.log({
        f"stage1/all_scores_table": table,
        f"stage1/n_variants": len(all_results),
        f"stage1/n_survivors": len(survivors),
        f"stage1/best_metric": survivors[0].metric if survivors else float("-inf"),
        f"stage1/worst_survivor_metric": survivors[-1].metric if survivors else float("-inf"),
    })

    # summary 에도 고정 (run 종료 후 wandb UI 에서 바로 보임)
    wandb.run.summary.update({
        "stage1_best_metric": survivors[0].metric if survivors else float("-inf"),
        "stage1_n_survivors": len(survivors),
    })
    logger.info("[wandb] Stage 1 summary logged to run '%s'", wandb.run.name)


def log_stage2_summary(
    step_n: int,
    round_n: int,
    all_results: "list",
    best: "object",
) -> None:
    """Stage 2 종료 시점 — 6 candidate scores + best 1 → wandb summary."""
    if wandb.run is None:
        return

    table = wandb.Table(columns=["variant_id", "params", "metric", "is_best"])
    for r in sorted(all_results, key=lambda x: x.metric, reverse=True):
        table.add_data(
            r.variant_id,
            str(r.params),
            r.metric,
            r.variant_id == best.variant_id,
        )

    wandb.log({
        "stage2/all_scores_table": table,
        "stage2/n_candidates": len(all_results),
        "stage2/best_metric": best.metric,
        "stage2/best_variant_id": best.variant_id,
        "stage2/best_params": str(best.params),
    })

    wandb.run.summary.update({
        "stage2_best_metric": best.metric,
        "stage2_best_variant_id": best.variant_id,
        "stage2_best_params": str(best.params),
    })
    logger.info(
        "[wandb] Stage 2 summary logged: best var%03d metric=%.4f",
        best.variant_id, best.metric,
    )


def log_round_summary(
    round_n: int,
    best_params: dict,
    importances: dict,
) -> None:
    """Round 종료 시점 — best params + importances → wandb summary.

    best params 는 다음 Round 의 고정값이므로 summary 에 명시적으로 기록.
    """
    if wandb.run is None:
        return

    payload: dict[str, Any] = {
        f"round{round_n}/best_params": str(best_params),
        f"round{round_n}/n_importances": len(importances),
    }
    # 각 best param 값도 개별 key 로 (wandb Parallel Coordinates 호환)
    for k, v in best_params.items():
        payload[f"round{round_n}/best_{k}"] = v
    for k, v in importances.items():
        payload[f"round{round_n}/importance_{k}"] = v

    wandb.log(payload)
    wandb.run.summary.update({
        f"round{round_n}_best_params": str(best_params),
    })
    logger.info("[wandb] Round %d summary logged: best=%s", round_n, best_params)


def init_orchestrator_run(
    step_n: int,
    round_n: int,
    *,
    mode: str = "online",
) -> "wandb.sdk.wandb_run.Run":
    """Round 단위 오케스트레이터 wandb run.

    개별 variant run (step{N}_round{R}_stage{S}_var{V}) 과 별도로,
    Stage 1/2 집계 summary 를 기록하는 long-lived run.
    name: step{N}_round{R}_orch
    tags: [stepN, roundR, orchestrator]
    """
    git_commit = _get_git_commit()
    run = wandb.init(
        project=_PROJECT,
        name=f"step{step_n}_round{round_n}_orch",
        config={"step": step_n, "round": round_n, "git_commit": git_commit},
        tags=[f"step{step_n}", f"round{round_n}", "orchestrator"],
        notes=f"Round {round_n} Stage 1/2 집계 summary",
        mode=mode,
        reinit=True,
    )
    logger.info("[wandb] orchestrator run init: step%d_round%d_orch", step_n, round_n)
    return run


def log_pbs_diagnostics(
    r_baseline_mean: float,
    r_shape_mean: float,
    phi_goal_start: float,
    phi_goal_end: float,
    step: int | None = None,
) -> None:
    """§16.7.8 wandb 진단 panel 항목 logging.

    r_shape_to_baseline_ratio 정상 범위: 0.1 ~ 1.0
      > 2.0: shape 가 baseline 압도 → 정책 왜곡 위험 경고
      < 0.05: shape 가 학습 신호로 미작동 경고
    """
    if wandb.run is None:
        return

    ratio = abs(r_shape_mean) / (abs(r_baseline_mean) + 1e-8)
    payload = {
        "pbs/r_baseline_mean": r_baseline_mean,
        "pbs/r_shape_mean": r_shape_mean,
        "pbs/r_shape_to_baseline_ratio": ratio,
        "pbs/phi_goal_at_step0": phi_goal_start,
        "pbs/phi_goal_at_terminal": phi_goal_end,
    }
    if step is not None:
        wandb.log(payload, step=step)
    else:
        wandb.log(payload)

    if ratio > 2.0:
        logger.warning("[wandb] PBS ratio=%.3f > 2.0 — shape 가 baseline 압도. 정책 왜곡 위험.", ratio)
    elif ratio < 0.05:
        logger.warning("[wandb] PBS ratio=%.3f < 0.05 — shape 가 학습 신호로 미작동.", ratio)
