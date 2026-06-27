"""핸드오프 패키지 저장/로드 — CLAUDE.md §13.1.

핸드오프 패키지 구성:
  best_model.zip              sb3 모델 가중치 (MaskablePPO.save)
  best_model_meta.json        아키텍처 정의
  stepN_normalizer.json       정규화 파라미터 (VecNormalize)
  stepN_kpi_report.json       최종 평가 metric
  stepN_training_log.csv      학습 이력 (wandb 보조)
  stepN_action_space_map.json 행동 공간
  stepN_reward_config.json    채택된 reward 설정
  stepN_env_config.json       환경 설정
  stepN_validation_set.json   평가 시나리오 + benchmark 결과
  stepN_regression_report.json  과거 Step 회귀 검증 결과 (Step 2 이후)
  stepN_wandb_run_url.txt     wandb run URL

전이학습 인터페이스:
  load_policy_for_transfer() → dict with model_path for next Step init.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from envs.scenario_generator import GENERATOR_VERSION

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 핸드오프 파일명 템플릿
# ─────────────────────────────────────────────────────────────────────────────

_FILES = {
    "model":           "best_model.zip",
    "model_meta":      "best_model_meta.json",
    "normalizer":      "step{N}_normalizer.json",
    "kpi_report":      "step{N}_kpi_report.json",
    "training_log":    "step{N}_training_log.csv",
    "action_space":    "step{N}_action_space_map.json",
    "reward_config":   "step{N}_reward_config.json",
    "env_config":      "step{N}_env_config.json",
    "validation_set":  "step{N}_validation_set.json",
    "regression":      "step{N}_regression_report.json",
    "wandb_url":       "step{N}_wandb_run_url.txt",
}


def _fname(key: str, step_n: int) -> str:
    return _FILES[key].replace("{N}", str(step_n))


# ─────────────────────────────────────────────────────────────────────────────
# HandoffConfig: 핸드오프 패키지의 메타데이터 (저장/로드 양쪽에서 사용)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class HandoffConfig:
    """핸드오프 패키지 메타데이터. CLAUDE.md §13.2."""

    step_n: int
    obs_dim: int = 150
    hidden_dim: int = 256
    n_actions: int = 7
    gamma_ppo: float = 0.99
    generator_version: str = GENERATOR_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "step_n": self.step_n,
            "obs_dim": self.obs_dim,
            "hidden_dim": self.hidden_dim,
            "n_actions": self.n_actions,
            "gamma_ppo": self.gamma_ppo,
            "generator_version": self.generator_version,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict) -> HandoffConfig:
        known = {k: d[k] for k in ("step_n", "obs_dim", "hidden_dim", "n_actions",
                                    "gamma_ppo", "generator_version") if k in d}
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(**known, extra=extra)


# ─────────────────────────────────────────────────────────────────────────────
# make_regression_report: §13.1 stepN_regression_report.json 생성
# ─────────────────────────────────────────────────────────────────────────────


def make_regression_report(
    step_n: int,
    metrics_by_step: dict[int, list[dict]] | None = None,
) -> dict:
    """CLAUDE.md §13.1 stepN_regression_report.json 형식 dict 생성.

    Parameters
    ----------
    step_n :
        현재 학습 Step (N).
    metrics_by_step :
        {step_id: [StepMetrics.to_dict(), ...]} — 시간순 평가 이력.
        Phase 1 (Step 1): None 또는 {} → note 와 함께 빈 report.
    """
    if not metrics_by_step:
        return {
            "step": step_n,
            "generator_version": GENERATOR_VERSION,
            "note": (
                f"Phase 1 (Step {step_n}): "
                "no previous steps to regress on. "
                "Regression callback inactive."
            ),
            "metrics_by_step": {},
        }

    return {
        "step": step_n,
        "generator_version": GENERATOR_VERSION,
        "metrics_by_step": {
            str(sid): history for sid, history in metrics_by_step.items()
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# save_handoff: 핸드오프 패키지 저장
# ─────────────────────────────────────────────────────────────────────────────


def save_handoff(
    step_n: int,
    save_dir: Path | str,
    *,
    model=None,
    config: HandoffConfig | None = None,
    regression_report: dict | None = None,
    wandb_url: str = "",
    env_config: dict | None = None,
    reward_config: dict | None = None,
    kpi_report: dict | None = None,
) -> Path:
    """CLAUDE.md §13.1 핸드오프 패키지 저장.

    Parameters
    ----------
    step_n :
        현재 Step 번호.
    save_dir :
        저장 디렉터리. 없으면 자동 생성.
    model :
        MaskablePPO 인스턴스 (None 이면 best_model.zip 생략).
    config :
        HandoffConfig (None 이면 기본값).
    regression_report :
        make_regression_report() 결과 dict (None 이면 Phase 1 빈 report 자동 생성).
    wandb_url :
        wandb run URL 문자열.
    env_config, reward_config, kpi_report :
        각 JSON 설정 dict (None 이면 해당 파일 생략).

    Returns
    -------
    save_dir : Path — 저장된 디렉터리.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # model weights
    if model is not None:
        model_path = save_dir / _fname("model", step_n)
        model.save(str(model_path.with_suffix("")))  # sb3 .save() 가 .zip 자동 추가
        logger.info("[handoff] model saved: %s", model_path)

    # best_model_meta.json
    if config is None:
        config = HandoffConfig(step_n=step_n)
    _write_json(save_dir / _fname("model_meta", step_n), config.to_dict())

    # regression report
    if regression_report is None:
        regression_report = make_regression_report(step_n)
    _write_json(save_dir / _fname("regression", step_n), regression_report)

    # wandb URL
    wandb_path = save_dir / _fname("wandb_url", step_n)
    wandb_path.write_text(wandb_url, encoding="utf-8")
    logger.info("[handoff] wandb_url saved: %s", wandb_path)

    # optional configs
    if env_config is not None:
        _write_json(save_dir / _fname("env_config", step_n), env_config)
    if reward_config is not None:
        _write_json(save_dir / _fname("reward_config", step_n), reward_config)
    if kpi_report is not None:
        _write_json(save_dir / _fname("kpi_report", step_n), kpi_report)

    logger.info("[handoff] step %d handoff saved to %s", step_n, save_dir)
    return save_dir


# ─────────────────────────────────────────────────────────────────────────────
# load_handoff: 핸드오프 패키지 로드
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class HandoffResult:
    """load_handoff() 반환값."""

    step_n: int
    save_dir: Path
    config: HandoffConfig
    regression_report: dict
    wandb_url: str
    model_path: Path | None = None      # best_model.zip 경로 (존재 시)
    env_config: dict | None = None
    reward_config: dict | None = None
    kpi_report: dict | None = None


def load_handoff(step_n: int, save_dir: Path | str) -> HandoffResult:
    """CLAUDE.md §12.2 전이학습용 핸드오프 패키지 로드.

    다음 Step 초기화 시:
        result = load_handoff(step_n=1, save_dir="handoff/step1")
        model = MaskablePPO.load(result.model_path)
    """
    save_dir = Path(save_dir)
    if not save_dir.exists():
        raise FileNotFoundError(f"handoff directory not found: {save_dir}")

    # model_meta (필수)
    meta_path = save_dir / _fname("model_meta", step_n)
    if not meta_path.exists():
        raise FileNotFoundError(f"model_meta not found: {meta_path}")
    config = HandoffConfig.from_dict(_read_json(meta_path))

    # regression report (필수)
    reg_path = save_dir / _fname("regression", step_n)
    regression_report: dict = _read_json(reg_path) if reg_path.exists() else {}

    # wandb URL (선택)
    url_path = save_dir / _fname("wandb_url", step_n)
    wandb_url = url_path.read_text(encoding="utf-8").strip() if url_path.exists() else ""

    # model zip (선택)
    model_path_candidate = save_dir / _fname("model", step_n)
    model_path = model_path_candidate if model_path_candidate.exists() else None

    # optional configs
    env_cfg_path = save_dir / _fname("env_config", step_n)
    rwd_cfg_path = save_dir / _fname("reward_config", step_n)
    kpi_cfg_path = save_dir / _fname("kpi_report", step_n)

    return HandoffResult(
        step_n=step_n,
        save_dir=save_dir,
        config=config,
        regression_report=regression_report,
        wandb_url=wandb_url,
        model_path=model_path,
        env_config=_read_json(env_cfg_path) if env_cfg_path.exists() else None,
        reward_config=_read_json(rwd_cfg_path) if rwd_cfg_path.exists() else None,
        kpi_report=_read_json(kpi_cfg_path) if kpi_cfg_path.exists() else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────────────────────────────────────


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.debug("[handoff] wrote %s", path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
