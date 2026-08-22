"""5.3.a dry-run: Round 1 12 variants × 5K timestep (Stage 1 + Stage 2 실제 경로).

목적:
  - 12개 variant 모두 정상 init 확인
  - NaN / inf metric 없음 확인
  - 캐시 시스템 (개별 variant + survivors) 정상 동작 확인
  - Optuna trial 이 실제로 COMPLETE 되는지 확인 (WAITING 에 멈추지 않는지)
  - 예상 소요: ~3~8분 (로컬 CPU, sequential, Stage 1 12개 + Stage 2 생존자)

RoundOrchestrator.run_round1() 을 직접 호출한다 — 프로덕션에서 실제로 쓰이는
경로 그대로를 5K timestep 축소판으로 검증하기 위함. 예전 버전은 이 경로를
우회한 자체 루프를 돌려서 캐시/Optuna 연동이 전혀 검증되지 않는 문제가 있었다.

매 실행마다 CACHE_DIR / OPTUNA_DB 를 초기화한다 (이전 실행의 잔재 캐시로 인해
"이번에 진짜 무슨 일이 일어났는지"가 가려지는 것을 방지).

wandb_mode='disabled' (로컬 검증용. Colab 실제 실행 시 'online' 또는 'offline').
"""

from __future__ import annotations

import logging
import math
import shutil
import sys
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import optuna
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from autoresearch.optuna_study import create_study, enqueue_round1_grid
from autoresearch.round_orchestrator import RoundOrchestrator
from training.train_step1 import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_ENT_COEF,
    DEFAULT_LR,
    DEFAULT_N_STEPS,
    _evaluate_success_rate,
    build_model,
    make_env_fn,
)

DRY_TIMESTEPS = 5_000
CACHE_DIR = _ROOT / "cache" / "dryrun_round1"
OPTUNA_DB_PATH = _ROOT / "autoresearch_dryrun.db"
OPTUNA_DB = f"sqlite:///{OPTUNA_DB_PATH}"

logger = logging.getLogger("dryrun_round1")


def make_train_fn():
    """12 variant 각각을 Step1Env 로 DRY_TIMESTEPS 학습 후 success_rate 반환."""

    def train_fn(params: dict, timesteps: int, variant_id: int) -> float:
        alpha = params.get("alpha", 1.0)
        beta = params.get("beta", 0.1)
        seed_base = 150000 + variant_id * 100

        vec_env = DummyVecEnv([make_env_fn(seed_base, alpha, beta, "medium")])
        vec_env = VecMonitor(vec_env)

        # dry-run: n_steps 를 timesteps/2 로 축소 (5K × 2 rollout 이면 충분)
        dry_n_steps = min(DEFAULT_N_STEPS, max(64, timesteps // 2))
        model = build_model(
            vec_env,
            lr=DEFAULT_LR,
            n_steps=dry_n_steps,
            batch_size=min(DEFAULT_BATCH_SIZE, dry_n_steps),
            ent_coef=DEFAULT_ENT_COEF,
        )

        model.learn(total_timesteps=timesteps)

        result = _evaluate_success_rate(
            model, alpha, beta, "medium", n_episodes=10  # 빠른 평가
        )
        vec_env.close()

        return result["success_rate"]

    return train_fn


def _reset_state() -> None:
    """이전 실행 잔재 제거 — 매 dry-run 이 처음부터 깨끗하게 시작하도록."""
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    if OPTUNA_DB_PATH.exists():
        OPTUNA_DB_PATH.unlink()


def _preflight_check_grid() -> list[dict]:
    """Round 1 α×β grid 를 enqueue 만 해두고 (아직 ask() 하지 않음) 12개/값 검증.

    실행 전에 grid 구성이 깨졌으면 5~8분짜리 dry-run을 시작하기 전에 바로 실패시킨다.
    RoundOrchestrator.run_round1() 이 뒤에서 다시 enqueue_round1_grid() 를 호출해도
    이미 등록된 조합은 중복 enqueue 되지 않는다 (dedup 내장).
    """
    study = create_study(step_n=1, round_n=1, storage=OPTUNA_DB)
    enqueue_round1_grid(study)

    waiting = [t for t in study.trials if t.state == optuna.trial.TrialState.WAITING]
    variants = [dict(t.system_attrs.get("fixed_params", {})) for t in waiting]

    print(f"\n[확인 1] enqueue 된 variants: {len(variants)}개 (기대: 12)")
    assert len(variants) == 12, f"variants 수 불일치: {len(variants)} ≠ 12"

    alphas = sorted({v["alpha"] for v in variants})
    betas = sorted({v["beta"] for v in variants})
    print(f"[확인 2] α 값: {alphas} (기대: [0.5, 1.0, 2.0, 5.0])")
    print(f"[확인 3] β 값: {betas}  (기대: [0.0, 0.1, 0.3])")
    assert alphas == [0.5, 1.0, 2.0, 5.0], f"α 불일치: {alphas}"
    assert betas == [0.0, 0.1, 0.3], f"β 불일치: {betas}"

    return variants


def _postflight_checks() -> tuple[list[str], list[float]]:
    """run_round1() 이후 — Optuna study + 캐시 파일을 단일 source of truth 로 검증.

    Returns
    -------
    (errors, completed_metrics) — errors 가 비어 있으면 전부 정상.
    """
    errors: list[str] = []

    study = create_study(step_n=1, round_n=1, storage=OPTUNA_DB)
    trials = study.trials
    n_total = len(trials)
    if n_total != 12:
        errors.append(f"Optuna trial 총 개수 불일치: {n_total} ≠ 12")

    not_complete = [t for t in trials if t.state != optuna.trial.TrialState.COMPLETE]
    if not_complete:
        detail = ", ".join(f"trial#{t.number}={t.state.name}" for t in not_complete)
        errors.append(f"COMPLETE 아닌 trial {len(not_complete)}개 존재: {detail}")

    completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed_metrics: list[float] = []
    for t in completed:
        value = t.value
        params = t.system_attrs.get("fixed_params", {})
        if value is None or math.isnan(value) or math.isinf(value):
            errors.append(
                f"trial#{t.number} params={params} metric={value} — 학습 실패"
                " (자세한 traceback 은 위 로그의 '[stage_runner] var... 학습 실패' 라인 참조)"
            )
        else:
            completed_metrics.append(value)

    # 캐시는 "성공한(finite metric)" variant 에 대해서만 쓰인다 — 실패 variant 는
    # 다음 실행에서 재시도되도록 의도적으로 캐싱하지 않는다
    # (StageRunner._run_one_safe 참조). 그래서 정확한 개수(예: 12)를 강제하지 않고,
    # "성공이 있는데 캐시가 완전히 비어있다" 만 캐시 메커니즘 고장으로 취급한다.
    stage1_cache = sorted(CACHE_DIR.glob("stage1_var*.json"))
    stage2_cache = sorted(CACHE_DIR.glob("stage2_var*.json"))
    if completed_metrics and not stage1_cache and not stage2_cache:
        errors.append(
            f"성공한 variant 가 {len(completed_metrics)}개 있는데 개별 캐시 파일이 "
            f"하나도 없음 (경로: {CACHE_DIR}) — 캐시 저장 로직 확인 필요"
        )

    return errors, completed_metrics


def main() -> bool:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # autoresearch 패키지만 DEBUG 로 올려서 variant 단위 진행상황을 노출한다
    # (다른 라이브러리 DEBUG 로그로 화면이 뒤덮이는 것 방지).
    logging.getLogger("autoresearch").setLevel(logging.DEBUG)

    print("=" * 60)
    print("5.3.a dry-run: Round 1 (12 variants × 5K timestep, Stage 1+2)")
    print(f"cache_dir : {CACHE_DIR}")
    print(f"optuna_db : {OPTUNA_DB}")
    print("wandb     : disabled (로컬 검증)")
    print("=" * 60)

    _reset_state()

    variants = _preflight_check_grid()

    print(f"\n[Round 1 시작] {len(variants)} variants × {DRY_TIMESTEPS} timestep "
          "(Stage 1 → 상위 50% → Stage 2)")
    t0 = time.time()

    errors: list[str] = []
    completed_metrics: list[float] = []
    try:
        orch = RoundOrchestrator(
            train_fn=make_train_fn(),
            step_n=1,
            max_workers=1,               # 로컬: sequential
            stage1_timesteps=DRY_TIMESTEPS,
            stage2_timesteps=DRY_TIMESTEPS,
            optuna_storage=OPTUNA_DB,
            cache_dir=str(CACHE_DIR),
            wandb_mode="disabled",
        )
        result = orch.run_round1()
        print(f"\n[Round 1 완료] best var{result.best.variant_id:03d} "
              f"metric={result.best.metric:.4f} params={result.best.params}")
    except Exception:
        tb = traceback.format_exc()
        logger.error("[dryrun] Round 1 실행 중 예상치 못한 예외:\n%s", tb)
        errors.append(f"Round 1 실행 자체가 실패함 — 아래 traceback 참조:\n{tb}")

    elapsed_total = time.time() - t0

    # 캐시/Optuna study 는 실제로 무슨 일이 일어났는지에 대한 단일 source of truth.
    # (errors 가 이미 있어도 — 즉 run_round1() 이 크래시했어도 — 상태 점검은
    #  그 자체로 유용하므로 계속 진행한다.)
    postflight_errors, completed_metrics = _postflight_checks()
    errors.extend(postflight_errors)

    all_ok = len(errors) == 0

    print("\n" + "=" * 60)
    print("dry-run 결과 요약")
    print("=" * 60)
    print(f"정상 완료 (Optuna COMPLETE, NaN/inf 없음) : {len(completed_metrics)} / 12")
    print(f"오류 항목 수: {len(errors)}")
    print(f"총 소요   : {elapsed_total:.1f}초 ({elapsed_total/60:.1f}분)")

    if completed_metrics:
        print("\nmetric 분포 (Optuna COMPLETE trial 기준):")
        print(f"  min  = {min(completed_metrics):.4f}")
        print(f"  max  = {max(completed_metrics):.4f}")
        print(f"  mean = {sum(completed_metrics) / len(completed_metrics):.4f}")

    if errors:
        print("\n⚠️  오류 목록:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("\n✅ 모든 12개 variant 정상 완료 — NaN/inf 없음, "
              "Optuna 12/12 COMPLETE, 캐시 정상")

    # 250K / 2M 외삽 (Stage 1 12개 + Stage 2 생존자 수 기준)
    n_stage2 = len(sorted(CACHE_DIR.glob("stage2_var*.json"))) or 6
    n_variant_runs = 12 + n_stage2
    if n_variant_runs > 0 and elapsed_total > 0:
        per_var_5k = elapsed_total / n_variant_runs
        stage1_250k_est = per_var_5k * 50 * 12 / 2 / 60  # 50배 × 12개 / worker 2 / 분
        stage2_2m_est = per_var_5k * 400 * n_stage2 / 2 / 60  # 400배 / worker 2 / 분
        print("\nColab T4 예상 (GPU ≈ 10~15× 가속, worker 2):")
        print(f"  Stage 1 (12 × 250K): ~{stage1_250k_est/15:.0f}~{stage1_250k_est/10:.0f}분")
        print(f"  Stage 2  ({n_stage2} × 2M) : ~{stage2_2m_est/15:.0f}~{stage2_2m_est/10:.0f}분")

    return all_ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
