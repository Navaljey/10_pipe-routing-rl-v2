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
