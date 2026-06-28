"""autoresearch — Optuna + wandb + 2-Stage screening. CLAUDE.md §16.5~§16.6.7."""

from autoresearch.optuna_study import create_study, get_param_importances
from autoresearch.stage_runner import StageRunner, STAGE1_TIMESTEPS, STAGE2_TIMESTEPS
from autoresearch.wandb_callback import (
    build_run_config,
    build_run_name,
    build_run_tags,
    init_wandb_run,
    make_wandb_sb3_callback,
)

__all__ = [
    "create_study",
    "get_param_importances",
    "StageRunner",
    "STAGE1_TIMESTEPS",
    "STAGE2_TIMESTEPS",
    "init_wandb_run",
    "make_wandb_sb3_callback",
    "build_run_name",
    "build_run_tags",
    "build_run_config",
]
