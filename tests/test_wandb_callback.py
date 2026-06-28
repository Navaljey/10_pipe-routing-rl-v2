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


# ─── 7. Stage/Round summary logging (disabled 모드) ─────────────────────────

from autoresearch.wandb_callback import (
    init_orchestrator_run,
    log_stage1_summary,
    log_stage2_summary,
    log_round_summary,
)
from autoresearch.stage_runner import VariantResult


def _make_variant(vid, metric, stage=1):
    return VariantResult(variant_id=vid, params={"alpha": 1.0}, metric=metric,
                         timesteps=100, stage=stage)


def test_log_stage1_summary_noop_when_no_run():
    """wandb.run=None 이면 log_stage1_summary 가 예외 없이 noop."""
    import wandb
    assert wandb.run is None
    results = [_make_variant(i, 0.5 + i * 0.05) for i in range(4)]
    survivors = results[:2]
    log_stage1_summary(1, 1, results, survivors)   # 예외 없어야 함


def test_log_stage2_summary_noop_when_no_run():
    """wandb.run=None 이면 log_stage2_summary 가 예외 없이 noop."""
    import wandb
    assert wandb.run is None
    results = [_make_variant(i, 0.6 + i * 0.05, stage=2) for i in range(2)]
    log_stage2_summary(1, 1, results, results[0])


def test_log_round_summary_noop_when_no_run():
    """wandb.run=None 이면 log_round_summary 가 예외 없이 noop."""
    import wandb
    assert wandb.run is None
    log_round_summary(1, {"alpha": 2.0, "beta": 0.1}, {"alpha": 0.7, "beta": 0.3})


def test_init_orchestrator_run_disabled():
    """init_orchestrator_run(mode='disabled') 가 예외 없이 run 반환."""
    import wandb
    run = init_orchestrator_run(step_n=1, round_n=1, mode="disabled")
    assert run is not None
    wandb.finish()


def test_log_stage1_summary_with_active_run():
    """활성 disabled run 에서 log_stage1_summary 예외 없이 실행."""
    import wandb
    run = init_orchestrator_run(step_n=1, round_n=1, mode="disabled")
    results = [_make_variant(i, 0.5 + i * 0.1) for i in range(6)]
    survivors = results[:3]
    log_stage1_summary(1, 1, results, survivors)   # 예외 없어야 함
    wandb.finish()


def test_log_stage2_summary_with_active_run():
    """활성 disabled run 에서 log_stage2_summary 예외 없이 실행."""
    import wandb
    run = init_orchestrator_run(step_n=1, round_n=1, mode="disabled")
    results = [_make_variant(i, 0.7 + i * 0.05, stage=2) for i in range(3)]
    best = results[0]
    log_stage2_summary(1, 1, results, best)
    wandb.finish()


def test_on_stage1_callback_called_in_stage_runner():
    """StageRunner.on_stage1_complete 콜백이 Stage 1 완료 시 호출된다."""
    from autoresearch.stage_runner import StageRunner

    called_with = []

    def cb(all_results, survivors):
        called_with.append((len(all_results), len(survivors)))

    runner = StageRunner(
        train_fn=lambda p, t, v: 0.5 + v * 0.1,
        stage1_timesteps=10,
        stage2_timesteps=20,
        on_stage1_complete=cb,
    )
    runner.run_stage1([{"v": i} for i in range(4)])
    assert len(called_with) == 1
    assert called_with[0] == (4, 2)   # 4 → 상위 50% = 2 생존


def test_on_stage2_callback_called_in_stage_runner():
    """StageRunner.on_stage2_complete 콜백이 Stage 2 완료 시 호출된다."""
    from autoresearch.stage_runner import StageRunner, VariantResult

    called_with = []

    def cb(all_results, best):
        called_with.append(best.metric)

    survivors = [VariantResult(i, {"v": i}, 0.6 + i * 0.1, 100, 1) for i in range(2)]
    runner = StageRunner(
        train_fn=lambda p, t, v: 0.8 + v * 0.05,
        stage1_timesteps=10,
        stage2_timesteps=20,
        on_stage2_complete=cb,
    )
    runner.run_stage2(survivors)
    assert len(called_with) == 1


def test_callback_exception_does_not_abort_stage():
    """콜백이 예외를 던져도 Stage 결과 자체는 정상 반환된다."""
    from autoresearch.stage_runner import StageRunner

    def bad_cb(all_results, survivors):
        raise RuntimeError("intentional callback error")

    runner = StageRunner(
        train_fn=lambda p, t, v: 0.5,
        stage1_timesteps=10,
        stage2_timesteps=20,
        on_stage1_complete=bad_cb,
    )
    survivors = runner.run_stage1([{"v": 0}, {"v": 1}])
    assert len(survivors) >= 1   # 예외에도 결과 반환됨
