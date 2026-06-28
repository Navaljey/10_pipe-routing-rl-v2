# Colab 실행 가이드 — Phase 1 Round 1 (Sub-단계 5.3)

> 무료 Colab T4 (12시간 세션) 기준. worker=2 (안전), 내결함성 캐시 활성.

---

## 0. 사전 조건

- Google Colab 접속 (무료 버전 T4 GPU)
- Google Drive 마운트 (H: 드라이브 = Google Drive 동기화 경로)
- `wandb login` (처음 1회)

---

## 1. 셀 1 — 환경 설정

```python
# Colab 환경 설정 + 의존성 설치
import subprocess, sys

# Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 프로젝트 경로
PROJECT = '/content/drive/MyDrive/Papers/10_pipe-routing-rl-v2'

# 의존성 설치 (최초 1회, 이후 세션에서도 재실행 필요)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'sb3-contrib', 'stable-baselines3', 'optuna', 'wandb',
    'gymnasium', 'scipy',
], check=True)

import sys
sys.path.insert(0, PROJECT)
print("환경 설정 완료")
```

---

## 2. 셀 2 — GPU 확인

```python
import torch
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# 속도 측정 (10K step)
import time
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import MaskablePPO
from envs.step1_env import Step1Env

env = make_vec_env(Step1Env, n_envs=1)
m = MaskablePPO('MlpPolicy', env,
    policy_kwargs=dict(net_arch=[256, 256]), verbose=0, gamma=0.99)
t0 = time.time()
m.learn(10_000)
elapsed = time.time() - t0
env.close()

print(f"\n10K steps = {elapsed:.1f}s")
print(f"250K ≈ {elapsed*25/60:.0f}분 | 2M ≈ {elapsed*200/3600:.1f}시간")
print(f"Stage 1 예상 (12× 250K, worker 2): {elapsed*25*6/3600:.1f}시간")
print(f"Stage 2 예상 (6× 2M, worker 2): {elapsed*200*3/3600:.1f}시간")
```

---

## 3. 셀 3 — wandb 로그인

```python
import wandb
wandb.login()   # API key 입력 (처음 1회)
```

---

## 4. 셀 4 — dry-run (12 variants × 5K, ~3분)

```python
"""
dry-run: 12개 variant 모두 정상 init + 단기 학습 확인.
Stage 1/2 본 실행 전 필수.
"""
import os
os.chdir(PROJECT)

from training.train_step1 import run_training, parse_args, make_env_fn, build_model
from autoresearch.round_orchestrator import RoundOrchestrator

CACHE_DIR = f"{PROJECT}/cache/round1"   # 내결함성 캐시

def make_train_fn_real(n_envs=1, handoff_dir=None):
    """실제 Step1Env 기반 train_fn."""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

    def train_fn(params: dict, timesteps: int, variant_id: int) -> float:
        alpha = params.get("alpha", 1.0)
        beta  = params.get("beta", 0.1)
        w1    = params.get("w1", 0.1)
        w2    = params.get("w2", 2.0)
        w3    = params.get("w3", 50.0)

        vec_env = DummyVecEnv([
            make_env_fn(150000 + variant_id * 100 + i, alpha, beta, "medium")
            for i in range(n_envs)
        ])
        vec_env = VecMonitor(vec_env)

        model = build_model(vec_env, lr=3e-4, n_steps=2048, batch_size=64, ent_coef=0.01)

        # wandb 연동 (autoresearch 명명 체계 §16.6.6)
        import wandb
        from autoresearch.wandb_callback import build_run_name, build_run_config, build_run_tags
        from wandb.integration.sb3 import WandbCallback
        round_n = 1
        stage_n = 1 if timesteps <= 250_000 else 2

        run = wandb.init(
            project="pipe-routing-rl",
            name=build_run_name(1, round_n, stage_n, variant_id),
            config=build_run_config(1, round_n, stage_n, variant_id, params,
                                    "unknown", "v1.0.0"),
            tags=build_run_tags(1, round_n, stage_n),
            reinit=True,
        )

        model.learn(total_timesteps=timesteps,
                    callback=WandbCallback(gradient_save_freq=0, verbose=0))

        # 평가: 20 episode
        from training.train_step1 import _evaluate_success_rate
        result = _evaluate_success_rate(model, alpha, beta, "medium", n_episodes=20)
        success_rate = result["success_rate"]

        wandb.log({"eval/success_rate": success_rate})
        wandb.finish()
        vec_env.close()
        return success_rate

    return train_fn


# dry-run: 5K timestep
orch_dry = RoundOrchestrator(
    train_fn=make_train_fn_real(n_envs=1),
    step_n=1,
    max_workers=2,           # 무료 Colab 안전값
    stage1_timesteps=5_000,  # dry-run 전용
    stage2_timesteps=5_000,
    optuna_storage=f"sqlite:///{PROJECT}/autoresearch_dry.db",
    cache_dir=f"{CACHE_DIR}_dry",
)

print("=== dry-run 시작 (12 variants × 5K) ===")
r1_dry = orch_dry.run_round1()
print(f"\n=== dry-run 완료 ===")
print(f"best: {r1_dry.best.params}  metric={r1_dry.best.metric:.4f}")
print(f"importances: {r1_dry.importances}")
print("\n✅ 12개 variant 모두 정상 — Stage 1 본 실행 진행 가능")
```

---

## 5. 셀 5 — Stage 1 본 실행 (12 × 250K)

```python
"""
Stage 1: 12 variants × 250K timestep.
무료 Colab T4 + worker 2 기준 예상 ~2~3시간.
세션 끊기면 같은 셀 재실행 → 완료된 variant 캐시 재사용.
"""

orch_r1 = RoundOrchestrator(
    train_fn=make_train_fn_real(n_envs=1),
    step_n=1,
    max_workers=2,
    stage1_timesteps=250_000,   # §16.5.B Stage 1
    stage2_timesteps=2_000_000, # §16.5.B Stage 2
    optuna_storage=f"sqlite:///{PROJECT}/autoresearch_round1.db",
    cache_dir=CACHE_DIR,        # 내결함성 캐시
)

from autoresearch.optuna_study import create_study, enqueue_round1_grid
from autoresearch.stage_runner import StageRunner
import optuna

study = create_study(step_n=1, round_n=1,
                     storage=f"sqlite:///{PROJECT}/autoresearch_round1.db")
enqueue_round1_grid(study)
variants = orch_r1._waiting_params(study)

print(f"=== Stage 1 시작: {len(variants)} variants × 250K ===")
runner = StageRunner(
    train_fn=make_train_fn_real(n_envs=1),
    max_workers=2,
    stage1_timesteps=250_000,
    stage2_timesteps=2_000_000,
    cache_dir=CACHE_DIR,
)
survivors = runner.run_stage1(variants)

print(f"\n=== Stage 1 완료: {len(variants)} → {len(survivors)} 생존 ===")
for s in sorted(survivors, key=lambda r: r.metric, reverse=True):
    print(f"  var{s.variant_id:03d} {s.params}  metric={s.metric:.4f}")

# 결과 저장 (Stage 2 전 세션 끊김 대비)
import json
(Path(PROJECT) / "cache/stage1_survivors.json").write_text(
    json.dumps([s._asdict() for s in survivors]), encoding="utf-8"
)
print("\n✅ Stage 1 결과 저장 완료 — 사용자 confirm 후 Stage 2 진행")
```

---

## 6. 셀 6 — Stage 2 본 실행 (6 × 2M)

> **Stage 1 결과 검토 후 실행.** 생존자 6개의 metric 분포를 확인하고 이상 없으면 진행.

```python
"""
Stage 2: Stage 1 생존자 × 2M timestep.
무료 Colab T4 + worker 2 기준 예상 ~3~6시간.
세션 끊기면 같은 셀 재실행 → 완료된 variant 캐시 재사용.
"""
from pathlib import Path
import json
from autoresearch.stage_runner import StageRunner, VariantResult

# Stage 1 결과 복구 (세션 끊김 후 재시작 시)
survivors_path = Path(PROJECT) / "cache/stage1_survivors.json"
if survivors_path.exists():
    survivors = [VariantResult(**d) for d in json.loads(survivors_path.read_text())]
    print(f"Stage 1 결과 복구: {len(survivors)}개")
# else: 셀 5 에서 이어서 실행 시 survivors 변수 그대로 사용

runner2 = StageRunner(
    train_fn=make_train_fn_real(n_envs=1),
    max_workers=2,
    stage1_timesteps=250_000,
    stage2_timesteps=2_000_000,
    cache_dir=CACHE_DIR,
)

print(f"=== Stage 2 시작: {len(survivors)} 후보 × 2M ===")
best = runner2.run_stage2(survivors)

print(f"\n=== Stage 2 완료 ===")
print(f"best variant: {best.params}")
print(f"best metric : {best.metric:.4f}")
print(f"\n✅ Round 1 완료 — best (α={best.params.get('alpha')}, β={best.params.get('beta')})")
```

---

## 7. 셀 7 — Round 1 결과 분석

```python
"""Round 1 결과 → Optuna importance + wandb 링크 확인."""
from autoresearch.optuna_study import create_study, get_param_importances

study = create_study(step_n=1, round_n=1,
                     storage=f"sqlite:///{PROJECT}/autoresearch_round1.db")
importances = get_param_importances(study)

print("=== Optuna Importance ===")
for k, v in sorted(importances.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v:.3f}")

print(f"\n=== Round 1 결과 요약 ===")
print(f"best α = {best.params.get('alpha')}")
print(f"best β = {best.params.get('beta')}")
print(f"best success_rate = {best.metric:.4f}")
print(f"\n→ Round 2 (w1 × w2 × w3 sweep, 36 variants) 진행 가능")
print(f"→ wandb: https://wandb.ai/[user]/pipe-routing-rl")
```

---

## 8. 세션 끊김 복구 체크리스트

세션이 끊겼을 때:

1. **셀 1** 재실행 (Drive 마운트 + 의존성)
2. **셀 3** 재실행 (wandb login)
3. 끊긴 셀부터 재실행 → 캐시 히트로 완료된 variant 자동 스킵
4. `cache/round1/` 에 `stage1_var000.json` ~ `stage1_var011.json` 로 진행 상황 확인

```bash
# 진행 상황 확인
ls cache/round1/ | wc -l   # 완료된 variant 수
```

---

## 타임라인 예상 (무료 Colab T4, worker 2)

| 단계 | 예상 시간 | 세션 내 완료 여부 |
|------|---------|----------------|
| dry-run (12 × 5K) | ~3~5분 | ✅ |
| Stage 1 (12 × 250K, worker 2) | ~2~3시간 | ✅ (12시간 내) |
| Stage 2 (6 × 2M, worker 2) | ~4~8시간 | ⚠️ 세션 끊기면 캐시로 복구 |
| **Round 1 합계** | **~6~11시간** | ⚠️ 무료 12시간 아슬아슬 |

> **팁**: Stage 1 완료 후 결과 저장 확인 (셀 5 마지막 줄) → Stage 2는 새 세션에서 시작해도 캐시로 복구됩니다.
