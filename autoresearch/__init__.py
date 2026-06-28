"""autoresearch — Optuna + wandb + 2-Stage screening + Round 운영. CLAUDE.md §16.5~§16.6.7."""

from autoresearch.optuna_study import create_study, get_param_importances
from autoresearch.round_orchestrator import (
    MAX_ROUNDS,
    PLATEAU_THRESHOLD,
    RoundOrchestrator,
    RoundResult,
)
from autoresearch.stage_runner import StageRunner, STAGE1_TIMESTEPS, STAGE2_TIMESTEPS
from autoresearch.wandb_callback import (
    build_run_config,
    build_run_name,
    build_run_tags,
    init_orchestrator_run,
    init_wandb_run,
    log_round_summary,
    log_stage1_summary,
    log_stage2_summary,
    make_wandb_sb3_callback,
)

__all__ = [
    "create_study",
    "get_param_importances",
    "RoundOrchestrator",
    "RoundResult",
    "MAX_ROUNDS",
    "PLATEAU_THRESHOLD",
    "StageRunner",
    "STAGE1_TIMESTEPS",
    "STAGE2_TIMESTEPS",
    "init_wandb_run",
    "init_orchestrator_run",
    "log_stage1_summary",
    "log_stage2_summary",
    "log_round_summary",
    "make_wandb_sb3_callback",
    "build_run_name",
    "build_run_tags",
    "build_run_config",
]
