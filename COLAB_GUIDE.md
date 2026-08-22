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
학습 "도중" 끊긴 경우는 §8 의 복구 절차를 먼저 실행할 것.
"""
import optuna
from autoresearch.optuna_study import (
    create_study, enqueue_round1_grid, ask_waiting_trials, register_grid_params,
)
from autoresearch.stage_runner import StageRunner

OPTUNA_DB = f"sqlite:///{PROJECT}/autoresearch_round1.db"

study = create_study(step_n=1, round_n=1, storage=OPTUNA_DB)
enqueue_round1_grid(study)

# WAITING trial 을 ask() 로 pop (RUNNING 전환) 하며 params 추출 +
# suggest_categorical() 으로 분포를 등록해야 셀 7 의 importance 분석이 가능해진다.
# (study.tell() 은 RUNNING trial 에만 가능 — WAITING 에 직접 tell() 하면 실패한다.)
trials = ask_waiting_trials(study)
register_grid_params(trials)
variants = [dict(t.system_attrs.get("fixed_params", {})) for t in trials]

print(f"=== 확인: enqueue 된 variants: {len(variants)}개 (기대: 12) ===")
assert len(variants) == 12, (
    f"variants 수가 12가 아닙니다 ({len(variants)}개). "
    "Stage 1 학습 도중 세션이 끊겼을 가능성이 큽니다 — "
    "§8 '세션 끊김 복구 체크리스트'의 복구 절차를 먼저 실행하세요."
)

def _tell_stage1(all_results, survivors):
    """Stage 1 탈락 variant 를 Stage 1 metric 으로 즉시 study.tell().
    이걸 안 하면 그 trial 은 RUNNING 에 멈춰 셀 7 importance 분석이 비어버린다.
    이미 COMPLETE 인 trial(재실행 시 캐시 히트한 variant)은 조용히 건너뛴다 —
    COMPLETE trial 에 다시 tell() 하면 ValueError 가 나기 때문."""
    survivor_ids = {r.variant_id for r in survivors}
    for r in all_results:
        if r.variant_id not in survivor_ids:
            trial = study.trials[r.variant_id]
            if trial.state == optuna.trial.TrialState.COMPLETE:
                continue
            try:
                study.tell(trial.number, r.metric)
            except Exception as e:
                print(f"  ⚠️ var{r.variant_id:03d} study.tell 실패: {e}")

runner = StageRunner(
    train_fn=make_train_fn_real(n_envs=1),
    max_workers=2,
    stage1_timesteps=250_000,
    stage2_timesteps=2_000_000,
    cache_dir=CACHE_DIR,        # 내결함성 캐시
    on_stage1_complete=_tell_stage1,
)

print(f"=== Stage 1 시작: {len(variants)} variants × 250K ===")
survivors = runner.run_stage1(variants)

print(f"\n=== Stage 1 완료: {len(variants)} → {len(survivors)} 생존 ===")
for s in sorted(survivors, key=lambda r: r.metric, reverse=True):
    print(f"  var{s.variant_id:03d} {s.params}  metric={s.metric:.4f}")

# StageRunner 가 {CACHE_DIR}/stage1_survivors.json 에 자동 저장한다 (Stage 2 재개용).
print(f"\n✅ Stage 1 결과 저장 완료: {CACHE_DIR}/stage1_survivors.json")
print("→ 위 생존자 metric 분포를 확인하고 이상 없으면 셀 6(Stage 2)으로 진행하세요.")
```

---

## 6. 셀 6 — Stage 2 본 실행 (6 × 2M)

> **Stage 1 결과 검토 후 실행.** 생존자 6개의 metric 분포를 확인하고 이상 없으면 진행.

```python
"""
Stage 2: Stage 1 생존자 × 2M timestep.
무료 Colab T4 + worker 2 기준 예상 ~3~6시간.
세션 끊기면 같은 셀 재실행 → 완료된 variant 캐시 재사용.
학습 "도중" 끊긴 경우는 §8 의 복구 절차를 먼저 실행할 것.
"""
from pathlib import Path
import json
import optuna
from autoresearch.optuna_study import create_study
from autoresearch.stage_runner import StageRunner, VariantResult

OPTUNA_DB = f"sqlite:///{PROJECT}/autoresearch_round1.db"
study = create_study(step_n=1, round_n=1, storage=OPTUNA_DB)  # 기존 study 를 그대로 로드 (load_if_exists)

# Stage 1 결과 복구 (새 세션에서 셀 6부터 이어서 실행하는 경우).
# 셀 5 에서 바로 이어서 실행 중이면 survivors 변수가 이미 있으므로 이 블록은 건너뛴다.
survivors_path = Path(CACHE_DIR) / "stage1_survivors.json"
if "survivors" not in globals():
    assert survivors_path.exists(), (
        "survivors 를 찾을 수 없습니다 — 셀 5(Stage 1)를 먼저 실행/완료하세요."
    )
    survivors = [VariantResult(**d) for d in json.loads(survivors_path.read_text())]
    print(f"Stage 1 결과 복구: {len(survivors)}개 (경로: {survivors_path})")

def _tell_stage2(all_results, best):
    """Stage 2 까지 도달한 variant 를 Stage 2 metric 으로 study.tell() — 셀 7 대비.
    이미 COMPLETE 인 trial(재실행 시 캐시 히트한 variant)은 조용히 건너뛴다."""
    for r in all_results:
        trial = study.trials[r.variant_id]
        if trial.state == optuna.trial.TrialState.COMPLETE:
            continue
        try:
            study.tell(trial.number, r.metric)
        except Exception as e:
            print(f"  ⚠️ var{r.variant_id:03d} study.tell 실패: {e}")

runner2 = StageRunner(
    train_fn=make_train_fn_real(n_envs=1),
    max_workers=2,
    stage1_timesteps=250_000,
    stage2_timesteps=2_000_000,
    cache_dir=CACHE_DIR,
    on_stage2_complete=_tell_stage2,
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
4. `cache/round1/` 에 `stage1_var000.json` ~ `stage1_var011.json` (Stage 1), `stage2_var{V:03d}.json` (Stage 2, 생존자만) 로 진행 상황 확인

```bash
# 진행 상황 확인
ls cache/round1/*.json | wc -l
```

이 방식(3~4번)은 **variant 학습이 완전히 끝난 뒤** 끊긴 경우에만 통합니다. **학습 도중** 끊긴 경우는 아래 별도 절차가 필요합니다.

### ⚠️ 알려진 한계: variant 학습 "도중" 끊기면 셀 재실행만으로 복구 안 됨

**증상**: 셀 5(또는 6)가 `ask_waiting_trials()` 로 Optuna trial 을 RUNNING 으로 전환한 뒤, `study.tell()` 로 완료 처리되기 전에 세션이 끊기면, 그 trial 들은 Optuna DB 안에 **RUNNING 상태로 영구히 멈춘 채** 남습니다. 이 상태에서 셀을 그냥 재실행하면:

- `enqueue_round1_grid()` 자체는 이미 RUNNING/COMPLETE 인 조합을 정상적으로 인식해 중복 enqueue 하지 않습니다 (문제 없음).
- 하지만 `ask_waiting_trials()` 는 **WAITING 상태 trial만** pop 하므로, 이미 RUNNING 으로 넘어간 trial 은 다시 가져오지 못하고 **0개**를 반환합니다.
- 결과적으로 `variants` 가 빈 리스트가 되어 `runner.run_stage1([])` 이 `ValueError: variants 가 비어 있습니다.` 로 즉시 실패합니다.
- 셀 5의 `assert len(variants) == 12` 가 이 상황을 이 에러보다 먼저, 더 명확한 메시지로 잡아줍니다.

**복구 절차 (실측 검증됨)** — Optuna DB 파일만 삭제하고 **셀 5부터** 다시 실행합니다 (Stage 2 도중 끊긴 경우도 동일하게 셀 5부터):

```python
import os
os.remove(f"{PROJECT}/autoresearch_round1.db")   # Optuna DB만 삭제. cache/round1/ 은 절대 지우지 말 것.
```

그 다음 **셀 5 → (필요 시) 셀 6** 순서로 그대로 재실행합니다. 이게 안전하고 완전한 이유:

- Round 1 grid(α×β)는 매번 같은 순서로 결정적(deterministic)으로 생성되므로, Optuna DB 를 지우고 새 study 를 만들어도 variant 순서/번호가 이전과 완전히 동일하게 재현됩니다.
- `StageRunner` 의 개별 variant 캐시(`stage1_var{V:03d}.json`, `stage2_var{V:03d}.json`)는 Optuna DB 와 무관하게 파일로 저장되므로 DB 를 지워도 그대로 남아있습니다.
- 따라서 재실행 시 **이미 완료된 variant 는 캐시를 히트해 즉시 스킵**되고, 중단됐던 variant 만 실제로 다시 학습됩니다.
- Stage 1 이 이미 끝난 뒤 Stage 2 도중 끊긴 경우에도 마찬가지입니다 — 셀 5 재실행은 12개 전부 캐시 히트로 몇 초 안에 통과하고, 셀 6에서 Stage 2 캐시가 없는 남은 생존자만 이어서 학습합니다.
- 실측: 12개 중 5개만 완료한 채 강제 중단 → DB 삭제 후 셀 5 재실행 → 정확히 나머지 7개만 재학습됨(캐시된 5개는 재학습 0회, train_fn 미호출). Stage 2 도중 강제 중단(생존자 6개 중 3개만 완료) 후 동일 절차로도 나머지 3개만 재학습되는 것을 확인함.

**주의 — 이 복구가 통하지 않는 경우**: `cache/round1/` 디렉터리 자체를 지우면 캐시가 없으므로 12개(또는 남은 생존자) 전부 처음부터 재학습됩니다 — 이 경우 복구가 아니라 사실상 재시작입니다. **DB 삭제는 안전하지만 캐시 디렉터리 삭제는 그렇지 않습니다** — 절대 함께 지우지 마세요.

---

## 타임라인 예상 (무료 Colab T4, worker 2)

| 단계 | 예상 시간 | 세션 내 완료 여부 |
|------|---------|----------------|
| dry-run (12 × 5K) | ~3~5분 | ✅ |
| Stage 1 (12 × 250K, worker 2) | ~2~3시간 | ✅ (12시간 내) |
| Stage 2 (6 × 2M, worker 2) | ~4~8시간 | ⚠️ 세션 끊기면 캐시로 복구 |
| **Round 1 합계** | **~6~11시간** | ⚠️ 무료 12시간 아슬아슬 |

> **팁**: Stage 1 완료 후 결과 저장 확인 (셀 5 마지막 줄) → Stage 2는 새 세션에서 시작해도 캐시로 복구됩니다.
