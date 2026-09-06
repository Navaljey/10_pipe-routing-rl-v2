"""Phase 1 Step 1 학습 entry point — baseline 1회 학습. CLAUDE.md §16, SKILL §3.6.

Usage
-----
  # sanity check (50K, offline wandb)
  python training/train_step1.py --timesteps 50000 --offline

  # 정상 학습 (2M, wandb online)
  python training/train_step1.py --timesteps 2000000

  # 특정 α/β 지정 (autoresearch 호환)
  python training/train_step1.py --alpha 2.0 --beta 0.1 --timesteps 250000

SKILL §3.1 절대 변경 금지 항목:
  hidden_dim=256, obs_dim=150, n_actions=7, MaskablePPO, γ_PBS==γ_PPO
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ─── 프로젝트 루트를 sys.path 에 추가 (Colab / CLI 양쪽 호환) ───────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from envs.step1_env import Step1Env  # noqa: E402
from training.handoff import (  # noqa: E402
    HandoffConfig,
    make_regression_report,
    save_handoff,
)

# ─── 학습 필수 상수 (SKILL §3.1 절대 변경 금지) ─────────────────────────────
HIDDEN_DIM = 256        # 전이학습 보장 — Step 1~10 동일
OBS_DIM = 150           # zero-padding 고정
N_ACTIONS = 7           # 7-direction discrete
GAMMA_PPO = 0.99        # γ_PBS == γ_PPO 조건 (§16.7.4)

# ─── baseline reward 초기값 (CLAUDE.md §16.3.1 L-16.3-w 해제) ───────────────
DEFAULT_W1 = 0.1    # length penalty
DEFAULT_W2 = 2.0    # collision penalty
DEFAULT_W3 = 50.0   # goal bonus
# w4=5.0, w5=15.0 은 §12.4 고정값, Step1Env 내부 상수

# ─── PBS 초기값 (CLAUDE.md §16.7.6) ────────────────────────────────────────
DEFAULT_ALPHA = 1.0   # Φ_goal 가중치
DEFAULT_BETA = 0.1    # Φ_cong 가중치

# ─── MaskablePPO sane defaults (autoresearch 전) ────────────────────────────
DEFAULT_N_STEPS = 2048
DEFAULT_BATCH_SIZE = 64
DEFAULT_ENT_COEF = 0.01
DEFAULT_LR = 3e-4
DEFAULT_N_ENVS = 1      # sanity check 기준. 정상 학습 시 증가 가능


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 1 Step 1 baseline 학습. CLAUDE.md §16, SKILL §3.6."
    )
    # 학습 규모
    p.add_argument("--timesteps", type=int, default=50_000,
                   help="총 학습 timestep (sanity=50K, 정상=2M)")
    p.add_argument("--n-envs", type=int, default=DEFAULT_N_ENVS,
                   help="병렬 환경 수")
    # PBS 가중치 (CLAUDE.md §16.7.6)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                   help="PBS Φ_goal 가중치 α (초기값 1.0)")
    p.add_argument("--beta", type=float, default=DEFAULT_BETA,
                   help="PBS Φ_cong 가중치 β (초기값 0.1)")
    # baseline reward 가중치 (CLAUDE.md §16.3.1)
    p.add_argument("--w1", type=float, default=DEFAULT_W1)
    p.add_argument("--w2", type=float, default=DEFAULT_W2)
    p.add_argument("--w3", type=float, default=DEFAULT_W3)
    # PPO hyperparameter
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--ent-coef", type=float, default=DEFAULT_ENT_COEF)
    # 실험 메타
    p.add_argument("--seed", type=int, default=150000,
                   help="학습용 시작 seed (§11.0.4 train range: 150000~199999)")
    p.add_argument("--difficulty", type=str, default="medium",
                   choices=["easy", "medium", "hard"])
    p.add_argument("--handoff-dir", type=str, default="handoff",
                   help="핸드오프 base 디렉터리 (step1/ 자동 생성됨)")
    # wandb 설정
    p.add_argument("--offline", action="store_true",
                   help="wandb offline 모드 (WANDB_MODE=offline)")
    p.add_argument("--run-name", type=str, default="step1_baseline_test",
                   help="wandb run name (CLAUDE.md §16.6.6)")
    p.add_argument("--no-wandb", action="store_true",
                   help="wandb 비활성화 (tests 전용)")
    return p.parse_args()


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_ROOT), stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
    except Exception:
        return "unknown"


def make_env_fn(seed: int, alpha: float, beta: float,
                difficulty: str, mode: str = "train",
                w1: float = DEFAULT_W1, w2: float = DEFAULT_W2, w3: float = DEFAULT_W3):
    """DummyVecEnv / SubprocVecEnv 용 env factory.

    w1/w2/w3 는 기본값이 Step1Env 의 모듈 상수와 동일하므로, 호출부에서 명시
    안 하면 기존과 동일하게 동작한다 (2026-09 FAILURE_LOG: w1/w2/w3 미배선 발견).
    """
    def _make():
        return Step1Env(
            mode=mode,
            difficulty=difficulty,
            seed=seed,
            alpha=alpha,
            beta=beta,
            w1=w1,
            w2=w2,
            w3=w3,
        )
    return _make


def build_model(vec_env, *, lr: float, n_steps: int, batch_size: int,
                ent_coef: float) -> "MaskablePPO":
    """SKILL §3.1: hidden_dim=256 절대 고정, MaskablePPO, γ=GAMMA_PPO."""
    from sb3_contrib import MaskablePPO

    return MaskablePPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=dict(net_arch=[HIDDEN_DIM, HIDDEN_DIM]),
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        ent_coef=ent_coef,
        gamma=GAMMA_PPO,      # γ_PBS == γ_PPO 조건 (CLAUDE.md §16.7.4)
        verbose=1,
    )


def run_training(args: argparse.Namespace) -> dict:
    """학습 실행 + 핸드오프 저장. 결과 요약 dict 반환."""
    import wandb
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    from wandb.integration.sb3 import WandbCallback

    # wandb 설정
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"

    git_commit = _get_git_commit()
    from envs.scenario_generator import GENERATOR_VERSION

    wandb_config = {
        "step": 1,
        "run_type": "baseline_sanity" if args.timesteps <= 100_000 else "baseline",
        "alpha": args.alpha,
        "beta": args.beta,
        "w1": args.w1,
        "w2": args.w2,
        "w3": args.w3,
        "learning_rate": args.lr,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "ent_coef": args.ent_coef,
        "gamma": GAMMA_PPO,
        "hidden_dim": HIDDEN_DIM,
        "obs_dim": OBS_DIM,
        "n_actions": N_ACTIONS,
        "timesteps": args.timesteps,
        "difficulty": args.difficulty,
        "git_commit": git_commit,
        "generator_version": GENERATOR_VERSION,
    }

    if not args.no_wandb:
        run = wandb.init(
            project="pipe-routing-rl",
            name=args.run_name,
            config=wandb_config,
            tags=["step1", "baseline", "phase1"],
            sync_tensorboard=False,
        )
        wandb_url = run.url or ""
    else:
        run = None
        wandb_url = ""

    # 환경 생성
    vec_env = DummyVecEnv([
        make_env_fn(args.seed + i, args.alpha, args.beta, args.difficulty,
                    w1=args.w1, w2=args.w2, w3=args.w3)
        for i in range(args.n_envs)
    ])
    vec_env = VecMonitor(vec_env)

    # 모델 생성 (SKILL §3.1)
    model = build_model(
        vec_env,
        lr=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
    )

    callbacks = []
    if not args.no_wandb and run is not None:
        callbacks.append(WandbCallback(
            gradient_save_freq=0,
            verbose=0,
        ))

    # 학습 실행
    t0 = time.time()
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks if callbacks else None,
    )
    elapsed = time.time() - t0

    # 평가: 50 episode 로 success_rate 측정
    eval_result = _evaluate_success_rate(model, args.alpha, args.beta,
                                         args.difficulty, n_episodes=50)

    # PBS ratio 별도 rollout 측정 (§16.7.8)
    pbs_ratio = _measure_pbs_ratio(model, args.alpha, args.beta, args.difficulty)

    # wandb 최종 metric logging
    if not args.no_wandb and run is not None:
        wandb.log({
            "eval/success_rate": eval_result["success_rate"],
            "eval/mean_episode_length": eval_result["mean_ep_length"],
            "pbs/r_shape_to_baseline_ratio_mean": pbs_ratio,
            "train/elapsed_sec": elapsed,
        })

    # 핸드오프 저장 (CLAUDE.md §13.1, 의사결정 25)
    env_config = {
        "grid_shape": [30, 30, 30],
        "max_steps": 500,
        "difficulty": args.difficulty,
        "obs_dim": OBS_DIM,
    }
    reward_config = {
        "w1": args.w1, "w2": args.w2, "w3": args.w3,
        "alpha": args.alpha, "beta": args.beta,
        "gamma": GAMMA_PPO,
    }
    kpi_report = {
        "step": 1,
        "timesteps": args.timesteps,
        "success_rate": eval_result["success_rate"],
        "mean_episode_length": eval_result["mean_ep_length"],
        "pbs_ratio_mean": pbs_ratio,
        "elapsed_sec": elapsed,
        "note": "baseline sanity run" if args.timesteps <= 100_000 else "baseline run",
    }

    handoff_cfg = HandoffConfig(
        step_n=1,
        gamma_ppo=GAMMA_PPO,
        generator_version=GENERATOR_VERSION,
        extra={
            "alpha": args.alpha,
            "beta": args.beta,
            "w1": args.w1, "w2": args.w2, "w3": args.w3,
        },
    )

    step_dir = save_handoff(
        step_n=1,
        save_dir=args.handoff_dir,
        model=model,
        config=handoff_cfg,
        regression_report=make_regression_report(step_n=1),
        wandb_url=wandb_url,
        env_config=env_config,
        reward_config=reward_config,
        kpi_report=kpi_report,
    )

    if not args.no_wandb and run is not None:
        wandb.finish()

    vec_env.close()

    return {
        "success_rate": eval_result["success_rate"],
        "mean_ep_length": eval_result["mean_ep_length"],
        "pbs_ratio_mean": pbs_ratio,
        "elapsed_sec": elapsed,
        "step_dir": str(step_dir),
        "wandb_url": wandb_url,
    }


def _evaluate_success_rate(model, alpha: float, beta: float,
                            difficulty: str, n_episodes: int = 50) -> dict:
    """학습 후 greedy evaluation. §11.2 success_rate 측정."""
    from sb3_contrib import MaskablePPO
    from envs.step1_env import Step1Env

    eval_seed = 130000  # §11.0.4 final_eval range 첫 seed
    successes = 0
    ep_lengths = []

    for i in range(n_episodes):
        env = Step1Env(
            mode="final_eval",
            difficulty=difficulty,
            alpha=alpha,
            beta=beta,
        )
        # 주의: seed 는 반드시 reset() 에 넘겨야 시나리오(occupancy/start/goal)가
        # 실제로 바뀐다. 생성자의 seed= 는 start_dir/goal_dir 추첨용 RNG만
        # 초기화할 뿐 _forced_scenario_seed 를 세팅하지 않아, 매 episode 마다
        # 새 인스턴스를 만들면 _next_seed() 폴백이 항상 pool[0] 하나로 고정된다
        # (FAILURE_LOG.md 참조 — Round 1 Stage 1/2 20-episode 평가가 전부 동일
        # 시나리오였던 원인).
        obs, _ = env.reset(seed=eval_seed + i)
        done = False
        length = 0
        while not done:
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            length += 1
        if info.get("termination") == "goal_reached":
            successes += 1
        ep_lengths.append(length)
        env.close()

    return {
        "success_rate": successes / n_episodes,
        "mean_ep_length": float(np.mean(ep_lengths)),
    }


# screening 순위용 composite score 안전장치 (CLAUDE.md §11.0.2, §11.2).
# score = success_rate - SCREENING_LENGTH_RATIO_EPS * min(mean_length_ratio, CAP)
# success_rate 는 75개 episode 기준 1/75 단위로만 변하므로, length_ratio 항이
# EPS*CAP < 1/75 를 만족하면 절대 success_rate 순위를 뒤집지 못하고 동률일 때만
# tie-break 역할을 한다 (lexicographic 정렬과 동일한 순서를 하나의 float 로 인코딩).
SCREENING_N_EPISODES = 75
SCREENING_LENGTH_RATIO_CAP = 10.0
SCREENING_LENGTH_RATIO_EPS = 1e-3
assert SCREENING_LENGTH_RATIO_EPS * SCREENING_LENGTH_RATIO_CAP < 1.0 / SCREENING_N_EPISODES


def _evaluate_screening(model, alpha: float, beta: float) -> dict:
    """Stage 1/2 screening 평가. CLAUDE.md §11.0(75개 고정, Easy:Medium:Hard=3:5:2) + §11.2(length_ratio).

    §11.2 는 "success_rate AND length_ratio" 채택 조건(둘 다 임계값 통과)을 정의하지만,
    screening 은 순위가 필요하므로 lexicographic 정렬(success_rate 우선, length_ratio
    tie-break)을 단일 float score 로 인코딩해 반환한다 — 근거는 위 상수 블록 주석 참조.

    L_astar 는 §12.4 의 강제 첫 스텝(start_dir_idx)을 반영해, "start_dir_idx 방향
    이웃 cell"을 기준점으로 BFS 최단거리 + 1 로 계산한다 (raw start 기준이면 물리
    제약 때문에 돌아간 거리를 RL 에게 부당하게 불리하게 매기게 된다).
    """
    from envs.base_env import FACE_DIRS
    from envs.scenario_generator import (
        STEP1_SCREENING_SEEDS_EASY,
        STEP1_SCREENING_SEEDS_HARD,
        STEP1_SCREENING_SEEDS_MEDIUM,
        bfs_shortest_path_length,
    )
    from envs.step1_env import Step1Env

    seed_pools = {
        "easy": STEP1_SCREENING_SEEDS_EASY,
        "medium": STEP1_SCREENING_SEEDS_MEDIUM,
        "hard": STEP1_SCREENING_SEEDS_HARD,
    }

    successes = 0
    length_ratios: list[float] = []
    by_difficulty: dict[str, dict[str, int]] = {
        d: {"n": len(seeds), "success": 0} for d, seeds in seed_pools.items()
    }

    for difficulty, seeds in seed_pools.items():
        for seed in seeds:
            env = Step1Env(mode="screening", difficulty=difficulty, alpha=alpha, beta=beta)
            obs, _ = env.reset(seed=seed)

            start_neighbor = env.agent_cell + FACE_DIRS[env.start_dir_idx]
            if env.is_occupied(start_neighbor):
                a_star_length = None  # 강제 첫 스텝 자체가 막힘 — benchmark 정의 불가
            else:
                d_rest = bfs_shortest_path_length(env.occupancy, start_neighbor, env.goal_cell)
                a_star_length = None if d_rest is None else d_rest + 1

            done = False
            length = 0
            while not done:
                mask = env.action_masks()
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                obs, _, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated
                length += 1

            if info.get("termination") == "goal_reached":
                successes += 1
                by_difficulty[difficulty]["success"] += 1
                if a_star_length is not None and a_star_length > 0:
                    length_ratios.append(length / a_star_length)
            env.close()

    n_total = sum(len(seeds) for seeds in seed_pools.values())
    success_rate = successes / n_total
    mean_length_ratio = float(np.mean(length_ratios)) if length_ratios else None
    length_penalty = (
        min(mean_length_ratio, SCREENING_LENGTH_RATIO_CAP)
        if mean_length_ratio is not None
        else 0.0
    )
    score = success_rate - SCREENING_LENGTH_RATIO_EPS * length_penalty

    return {
        "success_rate": success_rate,
        "mean_length_ratio": mean_length_ratio,
        "score": score,
        "by_difficulty": by_difficulty,
    }


class _PBSMonitorCallback:
    """PBS r_shape / r_baseline ratio 수집용 placeholder.

    실제 ratio 는 학습 후 _measure_pbs_ratio() 에서 별도 rollout 으로 측정.
    run_training() 에서 callback list 에 포함되지 않음 — 별도 사용.
    """

    def __init__(self):
        self.ratio_mean = 0.0


# PBS ratio 를 env info 에서 수집하려면 VecEnv step 후처리가 필요하지만
# DummyVecEnv 는 info 를 직접 노출하지 않는다.
# 대신 별도 단기 rollout 으로 측정한다.
def _measure_pbs_ratio(model, alpha: float, beta: float,
                       difficulty: str, n_steps: int = 500) -> float:
    """학습 후 단기 rollout 으로 r_shape/r_baseline ratio 측정. §16.7.8."""
    from envs.step1_env import Step1Env
    env = Step1Env(mode="train", difficulty=difficulty,
                   seed=150000, alpha=alpha, beta=beta)
    obs, _ = env.reset()
    ratios = []
    for _ in range(n_steps):
        mask = env.action_masks()
        action, _ = model.predict(obs, action_masks=mask, deterministic=False)
        obs, _, terminated, truncated, info = env.step(int(action))
        rb = info.get("reward_baseline", 0.0) or 0.0
        rs = info.get("reward_shape", 0.0) or 0.0
        denom = abs(rb) if abs(rb) > 1e-8 else 1e-8
        ratios.append(abs(rs) / denom)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    return float(np.mean(ratios)) if ratios else 0.0


def main() -> None:
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"Phase 1 Step 1 baseline 학습")
    print(f"  timesteps : {args.timesteps:,}")
    print(f"  alpha     : {args.alpha}  beta: {args.beta}")
    print(f"  w1/w2/w3  : {args.w1}/{args.w2}/{args.w3}")
    print(f"  offline   : {args.offline}")
    print(f"  handoff   : {args.handoff_dir}/step1/")
    print(f"{'='*60}\n")

    result = run_training(args)
    pbs_ratio = result["pbs_ratio_mean"]

    print(f"\n{'='*60}")
    print("Sub-단계 5.1 결과 보고")
    print(f"{'='*60}")
    print(f"  success_rate     : {result['success_rate']:.1%}")
    print(f"  mean_ep_length   : {result['mean_ep_length']:.1f} steps")
    print(f"  PBS ratio (mean) : {pbs_ratio:.3f}  (정상 범위: 0.1~1.0)")
    print(f"  elapsed          : {result['elapsed_sec']:.1f}s")
    print(f"  handoff dir      : {result['step_dir']}")
    print(f"  wandb URL        : {result['wandb_url'] or '(offline/no-wandb)'}")

    # 통과 기준 체크 (CLAUDE.md §11.2)
    sr = result["success_rate"]
    ratio_ok = 0.05 <= pbs_ratio <= 2.0  # §16.7.8 정상 범위 (느슨한 하한)
    passed = sr > 0.05 and ratio_ok

    print(f"\n[통과 기준]")
    print(f"  success_rate > 5% : {'✅' if sr > 0.05 else '❌'}  ({sr:.1%})")
    print(f"  PBS ratio 0.05~2.0: {'✅' if ratio_ok else '❌'}  ({pbs_ratio:.3f})")
    print(f"\n→ Sub-단계 5.1 {'통과 ✅ — 5.2 진입 가능' if passed else '실패 ❌ — FAILURE_LOG 작성 후 재검토'}")
    print(f"{'='*60}\n")

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
