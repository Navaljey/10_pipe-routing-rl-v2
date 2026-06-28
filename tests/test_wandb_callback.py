"""wandb_callback 모듈 단위 테스트 — autoresearch/wandb_callback.py.

검증 범위:
  - build_run_name: step{N}_round{R}_stage{S}_var{V:03d} 형식 (순수 함수, wandb 불필요)
  - build_run_tags: phase1/phase2plus 구분 (순수 함수)
  - build_run_config: 필수 키 포함 (순수 함수)
  - log_pbs_diagnostics: ratio 경계 경고 (wandb.run=None 시 noop)
  - make_wandb_sb3_callback: WandbCallback 인스턴스 반환 (disabled run 필요)
  - init_wandb_run: 통합 smoke (disabled 모드)

순수 함수 테스트는 wandb 불필요. wandb 필요 테스트만 disabled 모드로 실행.
"""

from __future__ import annotations

import logging

import pytest
import wandb

from autoresearch.wandb_callback import (
    build_run_config,
    build_run_name,
    build_run_tags,
    init_wandb_run,
    log_pbs_diagnostics,
    make_wandb_sb3_callback,
)


# ─── fixture: wandb disabled ─────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def wandb_disabled(monkeypatch):
    """wandb 가 필요한 테스트는 disabled 모드로 강제."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    yield
    if wandb.run is not None:
        wandb.finish()


# ─── 1. build_run_name (순수 함수) ───────────────────────────────────────────

def test_run_name_format_basic():
    """name = f'step{N}_round{R}_stage{S}_var{V:03d}' 형식."""
    assert build_run_name(1, 1, 1, 7) == "step1_round1_stage1_var007"


def test_run_name_variant_id_zero_padded():
    """variant_id 가 3자리 zero-padding: var001, var012, var100."""
    assert build_run_name(1, 1, 1, 1).endswith("var001")
    assert build_run_name(1, 1, 1, 12).endswith("var012")
    assert build_run_name(1, 1, 1, 100).endswith("var100")


def test_run_name_step2_round3_stage2():
    """다른 step/round/stage 조합도 올바른 형식."""
    assert build_run_name(2, 3, 2, 5) == "step2_round3_stage2_var005"


def test_run_name_all_digits_correct():
    """step/round/stage 숫자가 정확히 반영됨."""
    name = build_run_name(step_n=4, round_n=2, stage_n=1, variant_id=0)
    assert "step4" in name
    assert "round2" in name
    assert "stage1" in name
    assert "var000" in name


# ─── 2. build_run_tags (순수 함수) ───────────────────────────────────────────

def test_tags_contain_step_round_stage():
    """tags 에 step{N}, round{R}, stage{S}, autoresearch 포함."""
    tags = build_run_tags(step_n=1, round_n=2, stage_n=1)
    assert "step1" in tags
    assert "round2" in tags
    assert "stage1" in tags
    assert "autoresearch" in tags


def test_tags_phase1_for_step1():
    """step_n=1 → 'phase1' 태그, 'phase2plus' 없음."""
    tags = build_run_tags(step_n=1, round_n=1, stage_n=1)
    assert "phase1" in tags
    assert "phase2plus" not in tags


def test_tags_phase2plus_for_step2():
    """step_n=2 → 'phase2plus' 태그, 'phase1' 없음."""
    tags = build_run_tags(step_n=2, round_n=1, stage_n=1)
    assert "phase2plus" in tags
    assert "phase1" not in tags


def test_tags_phase2plus_for_step6():
    """step_n=6 → 'phase2plus' 태그."""
    tags = build_run_tags(step_n=6, round_n=3, stage_n=2)
    assert "phase2plus" in tags


# ─── 3. build_run_config (순수 함수) ─────────────────────────────────────────

def test_config_contains_structural_keys():
    """config 에 step, round, stage, variant_id 포함."""
    cfg = build_run_config(1, 1, 1, 0, {}, "abc123", "v1.0.0")
    for key in ("step", "round", "stage", "variant_id", "git_commit", "generator_version"):
        assert key in cfg, f"config 에 '{key}' 없음"


def test_config_contains_params():
    """params dict 가 config 에 포함된다."""
    params = {"alpha": 2.0, "beta": 0.3, "w1": 0.1}
    cfg = build_run_config(1, 1, 1, 0, params, "abc", "v1.0.0")
    for k, v in params.items():
        assert k in cfg
        assert cfg[k] == pytest.approx(v)


def test_config_generator_version():
    """generator_version 이 config 에 포함된다."""
    cfg = build_run_config(1, 1, 1, 0, {}, "abc", "v1.2.3")
    assert cfg["generator_version"] == "v1.2.3"


def test_config_step_n_matches():
    """config['step'] 가 step_n 과 일치한다."""
    cfg = build_run_config(3, 2, 1, 5, {}, "abc", "v1.0.0")
    assert cfg["step"] == 3
    assert cfg["round"] == 2
    assert cfg["stage"] == 1
    assert cfg["variant_id"] == 5


# ─── 4. log_pbs_diagnostics ──────────────────────────────────────────────────

def test_pbs_diagnostics_noop_when_no_run():
    """wandb.run=None 이면 예외 없이 noop."""
    assert wandb.run is None
    log_pbs_diagnostics(-10.0, -5.0, -0.9, -0.1)  # should not raise


def test_pbs_ratio_over_2_triggers_warning(caplog):
    """ratio > 2.0: 정책 왜곡 경고 발생."""
    run = init_wandb_run(1, 1, 1, 0, {}, "v1.0.0", mode="disabled")
    with caplog.at_level(logging.WARNING, logger="autoresearch.wandb_callback"):
        log_pbs_diagnostics(
            r_baseline_mean=-1.0,   # |baseline| = 1
            r_shape_mean=-5.0,      # |shape| = 5, ratio = 5 > 2
            phi_goal_start=-0.9,
            phi_goal_end=-0.1,
        )
    assert "PBS ratio" in caplog.text and "2.0" in caplog.text
    wandb.finish()


def test_pbs_ratio_under_0_05_triggers_warning(caplog):
    """ratio < 0.05: shape 미작동 경고 발생."""
    run = init_wandb_run(1, 1, 1, 0, {}, "v1.0.0", mode="disabled")
    with caplog.at_level(logging.WARNING, logger="autoresearch.wandb_callback"):
        log_pbs_diagnostics(
            r_baseline_mean=-100.0,   # |baseline| = 100
            r_shape_mean=-0.001,      # ratio ≈ 0 < 0.05
            phi_goal_start=-0.9,
            phi_goal_end=-0.1,
        )
    assert "PBS ratio" in caplog.text and "0.05" in caplog.text
    wandb.finish()


def test_pbs_ratio_normal_range_no_warning(caplog):
    """ratio=0.5 (정상 범위 0.1~1.0): 경고 없음."""
    run = init_wandb_run(1, 1, 1, 0, {}, "v1.0.0", mode="disabled")
    with caplog.at_level(logging.WARNING, logger="autoresearch.wandb_callback"):
        log_pbs_diagnostics(
            r_baseline_mean=-10.0,
            r_shape_mean=-5.0,  # ratio=0.5
            phi_goal_start=-0.9,
            phi_goal_end=-0.1,
        )
    assert "PBS ratio" not in caplog.text
    wandb.finish()


# ─── 5. make_wandb_sb3_callback ──────────────────────────────────────────────

def test_make_wandb_sb3_callback_returns_callback():
    """make_wandb_sb3_callback() 가 WandbCallback 인스턴스 반환."""
    from wandb.integration.sb3 import WandbCallback

    run = init_wandb_run(1, 1, 1, 0, {}, "v1.0.0", mode="disabled")
    cb = make_wandb_sb3_callback()
    assert isinstance(cb, WandbCallback)
    wandb.finish()


# ─── 6. init_wandb_run 통합 smoke ────────────────────────────────────────────

def test_init_wandb_run_smoke_disabled():
    """init_wandb_run(mode='disabled') 가 예외 없이 실행된다."""
    run = init_wandb_run(
        step_n=1, round_n=1, stage_n=1, variant_id=3,
        params={"alpha": 1.0, "beta": 0.1},
        generator_version="v1.0.0",
        mode="disabled",
    )
    assert run is not None
    wandb.finish()
