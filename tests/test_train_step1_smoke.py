"""Sub-단계 5.1 smoke test — 1K timestep 통합 학습.

검증 범위:
  - 학습이 NaN/inf 없이 완료
  - 핸드오프 4개 필수 파일 모두 생성 (§13.1)
  - PBS ratio 비이상 (∞, NaN 아님)
  - success_rate 0~1 범위
  - HandoffConfig 스키마 정합 (hidden_dim=256, obs_dim=150, n_actions=7)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from training.handoff import HandoffConfig, load_handoff


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def smoke_result(tmp_path_factory):
    """1K timestep 학습 실행 → (result_dict, step_dir) 반환."""
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from training.train_step1 import parse_args, run_training

    tmp = tmp_path_factory.mktemp("smoke_handoff")
    sys.argv = [
        "train_step1.py",
        "--timesteps", "1000",
        "--n-envs", "1",
        "--no-wandb",
        "--offline",
        "--handoff-dir", str(tmp),
        "--difficulty", "easy",
        "--seed", "150000",
    ]
    result = run_training(parse_args())
    return result, tmp


# ─────────────────────────────────────────────────────────────────────────────
# 1. 학습 완료 & NaN/inf 없음
# ─────────────────────────────────────────────────────────────────────────────


def test_smoke_no_nan_inf(smoke_result) -> None:
    """학습 결과에 NaN / inf 없음."""
    result, _ = smoke_result
    assert np.isfinite(result["success_rate"]), "success_rate NaN/inf"
    assert np.isfinite(result["mean_ep_length"]), "mean_ep_length NaN/inf"
    assert np.isfinite(result["elapsed_sec"]), "elapsed_sec NaN/inf"


def test_smoke_success_rate_range(smoke_result) -> None:
    """success_rate 가 [0, 1] 범위."""
    result, _ = smoke_result
    assert 0.0 <= result["success_rate"] <= 1.0


def test_smoke_elapsed_positive(smoke_result) -> None:
    """학습 시간 > 0."""
    result, _ = smoke_result
    assert result["elapsed_sec"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# 2. 핸드오프 파일 구조 (CLAUDE.md §13.1, 의사결정 25)
# ─────────────────────────────────────────────────────────────────────────────


def test_smoke_step_dir_created(smoke_result) -> None:
    """handoff/step1/ 디렉터리 생성됨."""
    _, tmp = smoke_result
    assert (tmp / "step1").is_dir()


def test_smoke_best_model_meta_exists(smoke_result) -> None:
    """best_model_meta.json 생성됨 (§13.1 필수 파일 1)."""
    _, tmp = smoke_result
    assert (tmp / "step1" / "best_model_meta.json").exists()


def test_smoke_regression_report_exists(smoke_result) -> None:
    """step1_regression_report.json 생성됨 (§13.1 필수 파일 2)."""
    _, tmp = smoke_result
    assert (tmp / "step1" / "step1_regression_report.json").exists()


def test_smoke_wandb_url_file_exists(smoke_result) -> None:
    """step1_wandb_run_url.txt 생성됨 (§13.1 필수 파일 3)."""
    _, tmp = smoke_result
    assert (tmp / "step1" / "step1_wandb_run_url.txt").exists()


def test_smoke_best_model_zip_exists(smoke_result) -> None:
    """best_model.zip 생성됨 (§13.1 필수 파일 4 — model weights)."""
    _, tmp = smoke_result
    assert (tmp / "step1" / "best_model.zip").exists()


def test_smoke_env_config_exists(smoke_result) -> None:
    """step1_env_config.json 생성됨."""
    _, tmp = smoke_result
    assert (tmp / "step1" / "step1_env_config.json").exists()


def test_smoke_reward_config_exists(smoke_result) -> None:
    """step1_reward_config.json 생성됨."""
    _, tmp = smoke_result
    assert (tmp / "step1" / "step1_reward_config.json").exists()


def test_smoke_kpi_report_exists(smoke_result) -> None:
    """step1_kpi_report.json 생성됨."""
    _, tmp = smoke_result
    assert (tmp / "step1" / "step1_kpi_report.json").exists()


# ─────────────────────────────────────────────────────────────────────────────
# 3. HandoffConfig 스키마 정합 (SKILL §3.1)
# ─────────────────────────────────────────────────────────────────────────────


def test_smoke_handoff_config_spec_values(smoke_result) -> None:
    """HandoffConfig 가 SKILL §3.1 절대 변경 금지 값과 일치한다."""
    _, tmp = smoke_result
    result = load_handoff(step_n=1, save_dir=tmp)
    cfg = result.config
    assert cfg.obs_dim == 150, f"obs_dim={cfg.obs_dim} (spec: 150)"
    assert cfg.hidden_dim == 256, f"hidden_dim={cfg.hidden_dim} (spec: 256)"
    assert cfg.n_actions == 7, f"n_actions={cfg.n_actions} (spec: 7)"
    assert cfg.gamma_ppo == pytest.approx(0.99), "gamma_ppo spec: 0.99"


def test_smoke_reward_config_initial_values(smoke_result) -> None:
    """reward_config 가 §16.3.1 초기값과 일치한다."""
    _, tmp = smoke_result
    result = load_handoff(step_n=1, save_dir=tmp)
    rc = result.reward_config
    assert rc is not None
    assert rc["w1"] == pytest.approx(0.1)
    assert rc["w2"] == pytest.approx(2.0)
    assert rc["w3"] == pytest.approx(50.0)
    assert rc["alpha"] == pytest.approx(1.0)
    assert rc["beta"] == pytest.approx(0.1)


def test_smoke_kpi_report_has_success_rate(smoke_result) -> None:
    """kpi_report 에 success_rate 가 있다."""
    _, tmp = smoke_result
    result = load_handoff(step_n=1, save_dir=tmp)
    assert result.kpi_report is not None
    assert "success_rate" in result.kpi_report
    assert 0.0 <= result.kpi_report["success_rate"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. wandb offline 모드 (--no-wandb 시 URL = "")
# ─────────────────────────────────────────────────────────────────────────────


def test_smoke_no_wandb_url_empty(smoke_result) -> None:
    """--no-wandb 실행 시 wandb_url 빈 문자열."""
    result, _ = smoke_result
    assert result["wandb_url"] == ""


# ─────────────────────────────────────────────────────────────────────────────
# 5. load_handoff() 로 model_path 획득 (전이학습 인터페이스)
# ─────────────────────────────────────────────────────────────────────────────


def test_smoke_model_path_loadable(smoke_result) -> None:
    """load_handoff() 로 얻은 model_path 가 MaskablePPO.load 가능하다."""
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    _, tmp = smoke_result
    result = load_handoff(step_n=1, save_dir=tmp)
    assert result.model_path is not None

    from training.train_step1 import make_env_fn
    vec_env = DummyVecEnv([make_env_fn(150000, 1.0, 0.1, "easy")])
    model = MaskablePPO.load(str(result.model_path.with_suffix("")), vec_env)
    assert model is not None
    vec_env.close()
