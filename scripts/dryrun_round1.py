"""5.3.a dry-run: Round 1 12 variants × 5K timestep.

목적:
  - 12개 variant 모두 정상 init 확인
  - NaN / inf metric 없음 확인
  - 캐시 시스템 정상 동작 확인
  - 예상 소요: ~2~5분 (로컬 CPU, sequential)

wandb_mode='disabled' (로컬 검증용. Colab 실제 실행 시 'online' 또는 'offline').
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from autoresearch.optuna_study import create_study, enqueue_round1_grid
from autoresearch.round_orchestrator import RoundOrchestrator
from autoresearch.stage_runner import StageRunner
from training.train_step1 import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_ENT_COEF,
    DEFAULT_LR,
    DEFAULT_N_STEPS,
    DEFAULT_W1,
    DEFAULT_W2,
    DEFAULT_W3,
    GAMMA_PPO,
    HIDDEN_DIM,
    _evaluate_success_rate,
    build_model,
    make_env_fn,
)

DRY_TIMESTEPS = 5_000
CACHE_DIR = _ROOT / "cache" / "dryrun_round1"
OPTUNA_DB = f"sqlite:///{_ROOT}/autoresearch_dryrun.db"


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

        metric = result["success_rate"]
        return metric

    return train_fn


def main():
    print("=" * 60)
    print("5.3.a dry-run: Round 1 (12 variants × 5K timestep)")
    print(f"cache_dir : {CACHE_DIR}")
    print(f"optuna_db : {OPTUNA_DB}")
    print("wandb     : disabled (로컬 검증)")
    print("=" * 60)

    # Optuna study + 12 variants enqueue
    study = create_study(step_n=1, round_n=1, storage=OPTUNA_DB)
    enqueue_round1_grid(study)

    orch = RoundOrchestrator(
        train_fn=make_train_fn(),
        step_n=1,
        max_workers=1,              # 로컬: sequential
        stage1_timesteps=DRY_TIMESTEPS,
        stage2_timesteps=DRY_TIMESTEPS,
        optuna_storage=OPTUNA_DB,
        cache_dir=str(CACHE_DIR),
        wandb_mode="disabled",
    )

    variants = orch._waiting_params(study)
    print(f"\n[확인 1] enqueue 된 variants: {len(variants)}개 (기대: 12)")
    assert len(variants) == 12, f"variants 수 불일치: {len(variants)} ≠ 12"

    # α × β 조합 검증
    alphas = sorted({v["alpha"] for v in variants})
    betas  = sorted({v["beta"]  for v in variants})
    print(f"[확인 2] α 값: {alphas} (기대: [0.5, 1.0, 2.0, 5.0])")
    print(f"[확인 3] β 값: {betas}  (기대: [0.0, 0.1, 0.3])")
    assert alphas == [0.5, 1.0, 2.0, 5.0], f"α 불일치: {alphas}"
    assert betas  == [0.0, 0.1, 0.3],      f"β 불일치: {betas}"

    # Stage 1 dry-run (12 × 5K)
    print(f"\n[Stage 1 시작] {len(variants)} variants × {DRY_TIMESTEPS} timestep")
    t0 = time.time()

    runner = StageRunner(
        train_fn=make_train_fn(),
        max_workers=1,
        stage1_timesteps=DRY_TIMESTEPS,
        stage2_timesteps=DRY_TIMESTEPS,
        cache_dir=str(CACHE_DIR),
    )

    results_all: list = []
    errors: list[str] = []

    # 개별 variant 를 직접 실행하여 NaN/inf 체크
    from autoresearch.stage_runner import VariantResult
    for i, params in enumerate(variants):
        t_var = time.time()
        try:
            metric = make_train_fn()(params, DRY_TIMESTEPS, i)
            elapsed_var = time.time() - t_var

            nan_flag = math.isnan(metric) or math.isinf(metric)
            status = "❌ NaN/inf" if nan_flag else "✅"
            print(f"  var{i:03d} α={params['alpha']:.1f} β={params['beta']:.1f}"
                  f"  metric={metric:.4f}  {elapsed_var:.1f}s  {status}")

            if nan_flag:
                errors.append(f"var{i:03d}: metric={metric}")
            results_all.append(VariantResult(i, params, metric, DRY_TIMESTEPS, 1))
        except Exception as e:
            elapsed_var = time.time() - t_var
            print(f"  var{i:03d} α={params['alpha']:.1f} β={params['beta']:.1f}"
                  f"  ERROR: {e}  {elapsed_var:.1f}s  ❌")
            errors.append(f"var{i:03d}: {e}")
            results_all.append(VariantResult(i, params, float("-inf"), DRY_TIMESTEPS, 1))

    elapsed_total = time.time() - t0

    # 결과 요약
    print("\n" + "=" * 60)
    print("dry-run 결과 요약")
    print("=" * 60)
    valid = [r for r in results_all if not math.isinf(r.metric) and not math.isnan(r.metric)]
    print(f"정상 완료 : {len(valid)} / {len(results_all)}")
    print(f"오류      : {len(errors)}")
    print(f"총 소요   : {elapsed_total:.1f}초 ({elapsed_total/60:.1f}분)")
    print(f"variant 당 평균: {elapsed_total/len(results_all):.1f}초")

    if valid:
        metrics = [r.metric for r in valid]
        print(f"\nmetric 분포:")
        print(f"  min  = {min(metrics):.4f}")
        print(f"  max  = {max(metrics):.4f}")
        print(f"  mean = {sum(metrics)/len(metrics):.4f}")

    if errors:
        print(f"\n⚠️  오류 목록:")
        for e in errors:
            print(f"  {e}")
    else:
        print("\n✅ 모든 12개 variant 정상 완료 — NaN/inf 없음")

    # 250K 외삽
    per_var_5k = elapsed_total / len(results_all)
    stage1_250k_est = per_var_5k * 50 * 12 / 2 / 60  # 50배 × 12개 / worker 2 / 분
    stage2_2m_est  = per_var_5k * 400 * 6  / 2 / 60   # 400배 × 6개 / worker 2 / 분
    print(f"\nColab T4 예상 (GPU ≈ 10~15× 가속, worker 2):")
    print(f"  Stage 1 (12 × 250K): ~{stage1_250k_est/10:.0f}~{stage1_250k_est/15:.0f}분")
    print(f"  Stage 2  (6 × 2M) : ~{stage2_2m_est/10:.0f}~{stage2_2m_est/15:.0f}분")

    return len(errors) == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
