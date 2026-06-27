"""핸드오프 패키지 검증 — CLAUDE.md §13.1.

검증 범위:
  - save_handoff() → 기대 파일 구조 생성
  - load_handoff() → HandoffResult 스키마 정합
  - make_regression_report() Phase 1 빈 report / Step 2+ 정상 report
  - wandb URL 저장/로드 왕복
  - HandoffConfig 직렬화/역직렬화 왕복
  - 전이학습 인터페이스: load_handoff 로 model_path 획득
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.handoff import (
    HandoffConfig,
    HandoffResult,
    load_handoff,
    make_regression_report,
    save_handoff,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. make_regression_report()
# ─────────────────────────────────────────────────────────────────────────────


def test_make_regression_report_phase1_empty() -> None:
    """Phase 1 (metrics_by_step=None): note 와 빈 metrics_by_step."""
    report = make_regression_report(step_n=1)
    assert report["step"] == 1
    assert report["metrics_by_step"] == {}
    assert "Phase 1" in report["note"] or "no previous" in report["note"]


def test_make_regression_report_step2_has_data() -> None:
    """Step 2: metrics_by_step={1: [...]} 가 report 에 포함된다."""
    history = [{"success_rate": 0.90, "timestamp_steps": 500_000}]
    report = make_regression_report(step_n=2, metrics_by_step={1: history})
    assert report["step"] == 2
    assert "1" in report["metrics_by_step"]
    assert report["metrics_by_step"]["1"] == history


def test_make_regression_report_has_generator_version() -> None:
    """report 에 generator_version 이 포함된다. CLAUDE.md §11.0.6."""
    from envs.scenario_generator import GENERATOR_VERSION

    report = make_regression_report(step_n=1)
    assert report["generator_version"] == GENERATOR_VERSION


# ─────────────────────────────────────────────────────────────────────────────
# 2. HandoffConfig 직렬화/역직렬화
# ─────────────────────────────────────────────────────────────────────────────


def test_handoff_config_to_dict_roundtrip() -> None:
    """HandoffConfig.to_dict() → HandoffConfig.from_dict() 왕복."""
    cfg = HandoffConfig(step_n=1, gamma_ppo=0.99)
    d = cfg.to_dict()
    cfg2 = HandoffConfig.from_dict(d)

    assert cfg2.step_n == cfg.step_n
    assert cfg2.obs_dim == cfg.obs_dim
    assert cfg2.hidden_dim == cfg.hidden_dim
    assert cfg2.n_actions == cfg.n_actions
    assert cfg2.gamma_ppo == pytest.approx(cfg.gamma_ppo)


def test_handoff_config_defaults_match_spec() -> None:
    """HandoffConfig 기본값이 CLAUDE.md §2.1, §4.3 spec 과 일치한다."""
    cfg = HandoffConfig(step_n=1)
    assert cfg.obs_dim == 150    # §4.3 관측 150-dim 고정
    assert cfg.hidden_dim == 256  # §2.1 backbone hidden_dim=256
    assert cfg.n_actions == 7    # §2.1 7-direction discrete


def test_handoff_config_extra_preserved() -> None:
    """extra 필드가 직렬화/역직렬화 왕복에서 보존된다."""
    cfg = HandoffConfig(step_n=1, extra={"alpha": 1.0, "beta": 0.1})
    cfg2 = HandoffConfig.from_dict(cfg.to_dict())
    assert cfg2.extra.get("alpha") == pytest.approx(1.0)
    assert cfg2.extra.get("beta") == pytest.approx(0.1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. save_handoff() 파일 구조
# ─────────────────────────────────────────────────────────────────────────────


def test_save_handoff_creates_directory(tmp_path: Path) -> None:
    """save_dir 이 없어도 save_handoff() 가 자동 생성한다."""
    save_dir = tmp_path / "new_subdir" / "step1_handoff"
    result_dir = save_handoff(step_n=1, save_dir=save_dir)
    assert result_dir.exists()
    assert result_dir.is_dir()


def test_save_handoff_creates_model_meta(tmp_path: Path) -> None:
    """save_handoff() 가 best_model_meta.json 을 생성한다."""
    save_handoff(step_n=1, save_dir=tmp_path)
    # best_model_meta.json (step-independent)
    assert (tmp_path / "best_model_meta.json").exists()


def test_save_handoff_creates_regression_report(tmp_path: Path) -> None:
    """save_handoff() 가 stepN_regression_report.json 을 생성한다."""
    save_handoff(step_n=1, save_dir=tmp_path)
    assert (tmp_path / "step1_regression_report.json").exists()


def test_save_handoff_creates_wandb_url_file(tmp_path: Path) -> None:
    """save_handoff() 가 stepN_wandb_run_url.txt 를 생성한다."""
    url = "https://wandb.ai/navaljey/pipe-routing-rl/runs/abc123"
    save_handoff(step_n=1, save_dir=tmp_path, wandb_url=url)
    txt_path = tmp_path / "step1_wandb_run_url.txt"
    assert txt_path.exists()
    assert txt_path.read_text(encoding="utf-8").strip() == url


def test_save_handoff_creates_env_config(tmp_path: Path) -> None:
    """env_config 제공 시 stepN_env_config.json 생성."""
    env_cfg = {"grid_shape": [30, 30, 30], "max_steps": 500}
    save_handoff(step_n=1, save_dir=tmp_path, env_config=env_cfg)
    path = tmp_path / "step1_env_config.json"
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["grid_shape"] == [30, 30, 30]


def test_save_handoff_creates_reward_config(tmp_path: Path) -> None:
    """reward_config 제공 시 stepN_reward_config.json 생성."""
    rwd_cfg = {"w1": 0.1, "w2": 2.0, "w3": 50.0, "alpha": 1.0, "beta": 0.1}
    save_handoff(step_n=1, save_dir=tmp_path, reward_config=rwd_cfg)
    path = tmp_path / "step1_reward_config.json"
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["w3"] == pytest.approx(50.0)


def test_save_handoff_skips_optional_if_none(tmp_path: Path) -> None:
    """env_config=None 이면 해당 파일 미생성."""
    save_handoff(step_n=1, save_dir=tmp_path, env_config=None, reward_config=None)
    assert not (tmp_path / "step1_env_config.json").exists()
    assert not (tmp_path / "step1_reward_config.json").exists()


def test_save_handoff_model_none_no_zip(tmp_path: Path) -> None:
    """model=None 이면 best_model.zip 미생성."""
    save_handoff(step_n=1, save_dir=tmp_path, model=None)
    assert not (tmp_path / "best_model.zip").exists()


# ─────────────────────────────────────────────────────────────────────────────
# 4. load_handoff() 스키마 정합
# ─────────────────────────────────────────────────────────────────────────────


def test_load_handoff_returns_handoff_result(tmp_path: Path) -> None:
    """load_handoff() 가 HandoffResult 를 반환한다."""
    save_handoff(step_n=1, save_dir=tmp_path)
    result = load_handoff(step_n=1, save_dir=tmp_path)
    assert isinstance(result, HandoffResult)


def test_load_handoff_config_roundtrip(tmp_path: Path) -> None:
    """save/load HandoffConfig 왕복."""
    cfg = HandoffConfig(step_n=1, gamma_ppo=0.99)
    save_handoff(step_n=1, save_dir=tmp_path, config=cfg)
    result = load_handoff(step_n=1, save_dir=tmp_path)
    assert result.config.step_n == 1
    assert result.config.gamma_ppo == pytest.approx(0.99)


def test_load_handoff_wandb_url_roundtrip(tmp_path: Path) -> None:
    """wandb URL save/load 왕복."""
    url = "https://wandb.ai/navaljey/pipe-routing-rl/runs/xyz"
    save_handoff(step_n=1, save_dir=tmp_path, wandb_url=url)
    result = load_handoff(step_n=1, save_dir=tmp_path)
    assert result.wandb_url == url


def test_load_handoff_regression_report_roundtrip(tmp_path: Path) -> None:
    """regression_report save/load 왕복."""
    report = make_regression_report(step_n=1)
    save_handoff(step_n=1, save_dir=tmp_path, regression_report=report)
    result = load_handoff(step_n=1, save_dir=tmp_path)
    assert result.regression_report["step"] == 1
    assert "metrics_by_step" in result.regression_report


def test_load_handoff_model_path_none_when_no_zip(tmp_path: Path) -> None:
    """best_model.zip 없으면 model_path=None 반환."""
    save_handoff(step_n=1, save_dir=tmp_path, model=None)
    result = load_handoff(step_n=1, save_dir=tmp_path)
    assert result.model_path is None


def test_load_handoff_missing_dir_raises(tmp_path: Path) -> None:
    """존재하지 않는 디렉터리 → FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_handoff(step_n=1, save_dir=tmp_path / "nonexistent")


def test_load_handoff_missing_meta_raises(tmp_path: Path) -> None:
    """model_meta 없으면 FileNotFoundError."""
    tmp_path.mkdir(exist_ok=True)
    with pytest.raises(FileNotFoundError, match="model_meta"):
        load_handoff(step_n=1, save_dir=tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# 5. 전이학습 인터페이스: 다음 Step 초기화용 model_path 획득
# ─────────────────────────────────────────────────────────────────────────────


def test_transfer_interface_model_path_usable(tmp_path: Path) -> None:
    """load_handoff() 로 획득한 model_path 가 사용 가능한 경로 객체이다.

    실제 모델 없이 경로 구조만 검증 (model=None 케이스).
    """
    save_handoff(step_n=1, save_dir=tmp_path)
    result = load_handoff(step_n=1, save_dir=tmp_path)
    # model=None 이면 model_path=None (다음 step 초기화 불가 → 호출자가 체크)
    assert result.model_path is None or isinstance(result.model_path, Path)


def test_handoff_step_numbering(tmp_path: Path) -> None:
    """step_n=6 핸드오프 파일명이 step6_ 접두사를 가진다."""
    save_handoff(step_n=6, save_dir=tmp_path)
    assert (tmp_path / "step6_regression_report.json").exists()
    assert (tmp_path / "step6_wandb_run_url.txt").exists()
