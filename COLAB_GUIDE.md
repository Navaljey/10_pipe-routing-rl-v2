# Colab 실행 가이드 — Phase 1 Round 1 (Sub-단계 5.3)

> 무료 Colab T4 (12시간 세션) 기준. worker=1 (순차 실행), 내결함성 캐시 활성.
>
> **worker=1 인 이유**: `StageRunner`/`RoundOrchestrator` 의 병렬 실행은 `ThreadPoolExecutor`
> (OS 스레드) 기반이다. 학습 루프가 `n_envs=1` 단일 env + 작은 MLP(256×256)로 대부분
> CPU 바운드(Python GIL 이 자주 안 풀림)이고, 무료 Colab 은 vCPU 2개 + T4 1개를 공유하므로
> `max_workers=2` 는 실제로는 GIL/CPU 경합만 늘려 **순차 실행보다 느려질 수 있다** (실측:
> 70분 동안 12개 중 1개도 완료 못 함). §16.5.B 의 "동시 학습 2~3개" 는 진짜 병렬(프로세스
> 기반) 실행을 전제로 한 수치이므로, 스레드 기반인 현재 구현에서는 worker=1 이 실측상 더 빠르다.

> **캐시 디렉터리 안내** (FAILURE_LOG.md 2026-09 entry — 과거 dry-run 셀이 진짜 결과를
> 삭제한 사고 이후 정리):
>
> | 디렉터리 | 용도 | 자동 삭제 여부 |
> |---|---|---|
> | `cache/round1/` | Stage 1/2 **진짜** 결과(JSON + 모델 체크포인트) | **자동 삭제 절대 안 됨** — 부록 A 에서만, 확인 문구 입력 시에만 |
> | `cache/round1_smoke/` | 셀 4 dry-run(5K) 전용 | 셀 4 실행마다 자동 정리됨 (버릴 데이터) |
> | `cache/round1_cli_smoke/` | `scripts/dryrun_round1.py` — 로컬/CI 전용, **Colab 노트북과 무관** | 그 스크립트 실행마다 자동 정리됨 |

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
print(f"Stage 1 예상 (12 variants × 250K, worker 1 순차): {elapsed*25*12/3600:.1f}시간")
print(f"Stage 2 예상 (6 candidates × 2M, worker 1 순차): {elapsed*200*6/3600:.1f}시간")
```

---

## 3. 셀 3 — wandb 로그인

```python
import wandb
wandb.login()   # API key 입력 (처음 1회)
```

---

## 4. 셀 4 — dry-run (12 variants × 5K, ~10~15분)

```python
"""
dry-run: 12개 variant 모두 정상 init + 단기 학습 확인.
Stage 1/2 본 실행 전 필수.
screening 평가가 이제 75개 시나리오(Easy23/Medium37/Hard15)를 돌기 때문에,
학습(5K) 자체보다 평가가 시간을 더 잡아먹을 수 있다 (variant당 최대 ~1분 내외,
정책이 대부분 timeout 나면 더 걸릴 수 있음 — Stage 1/2 본 실행(250K/2M)에서는
학습 시간이 훨씬 커서 무시할 수준).
"""
import os
os.chdir(PROJECT)

# GPU 확인 (하드 스톱) — FAILURE_LOG.md 참조: 이 확인이 없어 CPU로 7시간 학습되고도
# 아무 경고 없이 넘어간 사고가 있었다. 정말 CPU로 진행하려면 REQUIRE_GPU = False.
import torch
REQUIRE_GPU = True
if torch.cuda.is_available():
    print(f"✅ GPU 사용 중: {torch.cuda.get_device_name(0)}")
else:
    print(
        "⚠️ GPU 를 사용할 수 없습니다 — 이대로 진행하면 CPU 로 학습되어 250K/2M "
        "학습이 수 배 이상 느려집니다 (실제 사고 사례: 7시간 동안 12개 중 8~9개만 완료).\n"
        "   Colab 상단 메뉴 → 런타임 → 런타임 유형 변경 → 하드웨어 가속기 GPU 선택 후 "
        "세션을 다시 시작하세요."
    )
    if REQUIRE_GPU:
        raise RuntimeError(
            "GPU 미사용 — REQUIRE_GPU=True(기본값) 라 중단합니다. "
            "GPU 없이 계속하려면 이 셀 상단의 REQUIRE_GPU = False 로 바꾼 뒤 재실행하세요."
        )

from training.train_step1 import run_training, parse_args, make_env_fn, build_model
from autoresearch.round_orchestrator import RoundOrchestrator

CACHE_DIR = f"{PROJECT}/cache/round1"          # 진짜 Stage 1/2 결과 — 이 셀에서 절대 건드리지 않는다
SMOKE_CACHE_DIR = f"{PROJECT}/cache/round1_smoke"  # dry-run 전용, 매번 새로 씀

# ⚠️ dry-run(smoke test) 전용 캐시/DB 만 정리한다. cache/round1/(진짜 결과)은 이 셀에서
# 절대 삭제하지 않는다 — 예전에 여기서 실수로 CACHE_DIR 까지 지워 진짜 250K 결과
# 8~9개가 통째로 사라진 사고가 있었다 (FAILURE_LOG.md 참조). 진짜 캐시를 지워야 하는
# 경우는 이 문서 맨 아래 "부록 A. 위험 작업" 의 별도 확인 셀만 사용할 것.
import shutil
shutil.rmtree(SMOKE_CACHE_DIR, ignore_errors=True)
_dry_db = f"{PROJECT}/autoresearch_dry.db"
if os.path.exists(_dry_db):
    os.remove(_dry_db)
print(f"dry-run 전용 캐시/DB 정리 완료 ({SMOKE_CACHE_DIR}) — cache/round1/ 은 건드리지 않음")

def make_train_fn_real(n_envs=1, handoff_dir=None, cache_dir=None):
    """실제 Step1Env 기반 train_fn.

    cache_dir 는 반드시 호출부에서 명시적으로 전달한다 (dry-run 은 SMOKE_CACHE_DIR,
    본 실행은 CACHE_DIR). 체크포인트를 여기 저장하므로, 전역 CACHE_DIR 을 암묵적으로
    참조하면 dry-run 이 진짜 캐시 폴더에 체크포인트를 잘못 쓰거나 셀 5/6 실행 시점에
    따라 저장 위치가 바뀌는 등 dry-run/본 실행이 서로 뒤섞일 위험이 있다.
    """
    assert cache_dir is not None, "cache_dir 를 명시적으로 전달하세요 (SMOKE_CACHE_DIR 또는 CACHE_DIR)"
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

        # 체크포인트 저장 — eval 이 실패해도 가중치는 보존된다. 소실 시 복구도
        # 재평가도 불가능했던 문제(FAILURE_LOG.md 참조)의 대응. 크기/시간 비용은
        # 무시 가능한 수준으로 실측됨: net_arch=[256,256] 고정(SKILL §3.1)이라
        # variant 당 항상 ~0.87MB, ~22ms — Round 1 전체(18개) 합쳐도 ~16MB, ~0.4초.
        model.save(f"{cache_dir}/stage{stage_n}_var{variant_id:03d}_model")

        # screening 평가: §11.0 spec 대로 75개 고정 시나리오(Easy 23/Medium 37/Hard 15).
        # 매 episode 마다 reset(seed=) 로 시나리오를 명시 고정한다 — 이전 버그
        # (생성자 seed= 만 주고 reset() 을 seed 없이 호출 → 20개가 전부 동일 시나리오)
        # 는 _evaluate_screening() 내부에서 이미 고쳐져 있다. FAILURE_LOG.md 참조.
        # §9 컴퓨팅 비용 추정을 위해 평가 시간만 따로 잰다 — 학습 시간은
        # StageRunner._run_one() 이 train_fn 호출 전체 소요시간에서 이 값을 빼서
        # 자동 계산한다(autoresearch/stage_runner.py 참조).
        import time
        _eval_t0 = time.time()
        from training.train_step1 import _evaluate_screening
        result = _evaluate_screening(model, alpha, beta)
        eval_time_sec = time.time() - _eval_t0
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

        wandb.log({
            "eval/success_rate": result["success_rate"],
            "eval/mean_length_ratio": result["mean_length_ratio"] or 0.0,
            "eval/score": result["score"],
            "eval/success_rate_easy": result["by_difficulty"]["easy"]["success"] / result["by_difficulty"]["easy"]["n"],
            "eval/success_rate_medium": result["by_difficulty"]["medium"]["success"] / result["by_difficulty"]["medium"]["n"],
            "eval/success_rate_hard": result["by_difficulty"]["hard"]["success"] / result["by_difficulty"]["hard"]["n"],
            "perf/eval_time_sec": eval_time_sec,
        })
        wandb.finish()
        vec_env.close()
        # StageRunner/Optuna 는 튜플의 첫 원소(score)로 variant 순위를 매긴다.
        # score = success_rate - EPS*length_ratio (lexicographic, §11.2) — 상세는
        # training/train_step1.py 의 SCREENING_LENGTH_RATIO_EPS/CAP 주석 참조.
        # (score, eval_time_sec, gpu_name) tuple 반환은 §9 컴퓨팅 비용 추정 opt-in
        # 이며, cache/round1/stage{S}_var{V:03d}.json 에 train_time_sec/eval_time_sec/
        # gpu_name 으로 함께 저장된다 (autoresearch/stage_runner.py 참조).
        return result["score"], eval_time_sec, gpu_name

    return train_fn


# dry-run: 5K timestep
orch_dry = RoundOrchestrator(
    train_fn=make_train_fn_real(n_envs=1, cache_dir=SMOKE_CACHE_DIR),
    step_n=1,
    max_workers=1,           # 무료 Colab: ThreadPoolExecutor 는 GIL 경합으로 worker>1 이 더 느림
    stage1_timesteps=5_000,  # dry-run 전용
    stage2_timesteps=5_000,
    optuna_storage=f"sqlite:///{_dry_db}",
    cache_dir=SMOKE_CACHE_DIR,
)

print("=== dry-run 시작 (12 variants × 5K) ===")
r1_dry = orch_dry.run_round1()
print(f"\n=== dry-run 완료 ===")
print(f"best: {r1_dry.best.params}  metric={r1_dry.best.metric:.4f}")
print(f"importances: {r1_dry.importances}")

# screening 평가 버그 수정 검증: 12개 variant 의 score 가 서로 달라야 한다.
# 전부 동일하면(구 버그처럼) 평가가 아직도 variant 를 구분 못 하고 있다는 뜻이므로
# Stage 1 본 실행으로 넘어가지 말고 먼저 원인을 다시 확인할 것.
import glob, json
_dry_scores = []
for _f in sorted(glob.glob(f"{SMOKE_CACHE_DIR}/stage1_var*.json")):
    _dry_scores.append(json.loads(open(_f).read())["metric"])
_n_unique = len(set(round(s, 6) for s in _dry_scores))
print(f"\n12개 variant score: {[round(s, 4) for s in _dry_scores]}")
assert _n_unique > 1, (
    "12개 variant 의 score 가 전부 동일합니다 — screening 평가가 여전히 variant 를 "
    "구분하지 못하고 있다는 신호입니다 (FAILURE_LOG.md 2026-08-30 entry 의 버그가 "
    "재발했을 가능성). Stage 1 본 실행으로 넘어가지 마세요."
)
print(f"✅ variant 간 score 변별력 확인됨 (고유값 {_n_unique}개) — Stage 1 본 실행 진행 가능")
```

---

## 5. 셀 5 — Stage 1 본 실행 (12 × 250K)

```python
"""
Stage 1: 12 variants × 250K timestep.
무료 Colab T4 + worker 1(순차) 기준 예상 ~3~4시간.
세션 끊기면 같은 셀 재실행 → 완료된 variant 캐시 재사용.
학습 "도중" 끊긴 경우는 §8 의 복구 절차를 먼저 실행할 것.
"""
import torch
from pathlib import Path
import optuna
from autoresearch.optuna_study import (
    create_study, enqueue_round1_grid, ask_waiting_trials, register_grid_params,
)
from autoresearch.stage_runner import StageRunner

# GPU 확인 (하드 스톱) — FAILURE_LOG.md 참조. 정말 CPU로 진행하려면 REQUIRE_GPU = False.
REQUIRE_GPU = True
if torch.cuda.is_available():
    print(f"✅ GPU 사용 중: {torch.cuda.get_device_name(0)}")
else:
    print(
        "⚠️ GPU 를 사용할 수 없습니다 — 이대로 진행하면 CPU 로 학습되어 250K 학습이 "
        "수 배 이상 느려집니다 (실제 사고 사례: 7시간 동안 12개 중 8~9개만 완료).\n"
        "   Colab 상단 메뉴 → 런타임 → 런타임 유형 변경 → 하드웨어 가속기 GPU 선택 후 "
        "세션을 다시 시작하세요."
    )
    if REQUIRE_GPU:
        raise RuntimeError(
            "GPU 미사용 — REQUIRE_GPU=True(기본값) 라 중단합니다. "
            "GPU 없이 계속하려면 이 셀 상단의 REQUIRE_GPU = False 로 바꾼 뒤 재실행하세요."
        )

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

# 재개인지 신규 실행인지 즉시 알 수 있도록 캐시 히트 개수를 먼저 출력한다.
_cache_hit_n = sum(
    1 for i in range(len(variants)) if (Path(CACHE_DIR) / f"stage1_var{i:03d}.json").exists()
)
print(f"{len(variants)}개 중 {_cache_hit_n}개 캐시 재사용, {len(variants) - _cache_hit_n}개 신규 학습")

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
    train_fn=make_train_fn_real(n_envs=1, cache_dir=CACHE_DIR),
    max_workers=1,               # ThreadPoolExecutor GIL 경합 회피 (상단 안내 참조)
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
무료 Colab T4 + worker 1(순차) 기준 예상 ~12~14시간 — 세션 1개로 안 끝날 수 있음.
세션 끊기면 같은 셀 재실행 → 완료된 variant 캐시 재사용.
학습 "도중" 끊긴 경우는 §8 의 복구 절차를 먼저 실행할 것.
"""
from pathlib import Path
import json
import optuna
import torch
from autoresearch.optuna_study import create_study
from autoresearch.stage_runner import StageRunner, VariantResult

# GPU 확인 (하드 스톱) — FAILURE_LOG.md 참조. 정말 CPU로 진행하려면 REQUIRE_GPU = False.
REQUIRE_GPU = True
if torch.cuda.is_available():
    print(f"✅ GPU 사용 중: {torch.cuda.get_device_name(0)}")
else:
    print(
        "⚠️ GPU 를 사용할 수 없습니다 — 이대로 진행하면 CPU 로 학습되어 2M 학습이 "
        "수 배 이상 느려집니다.\n"
        "   Colab 상단 메뉴 → 런타임 → 런타임 유형 변경 → 하드웨어 가속기 GPU 선택 후 "
        "세션을 다시 시작하세요."
    )
    if REQUIRE_GPU:
        raise RuntimeError(
            "GPU 미사용 — REQUIRE_GPU=True(기본값) 라 중단합니다. "
            "GPU 없이 계속하려면 이 셀 상단의 REQUIRE_GPU = False 로 바꾼 뒤 재실행하세요."
        )

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
    train_fn=make_train_fn_real(n_envs=1, cache_dir=CACHE_DIR),
    max_workers=1,                # ThreadPoolExecutor GIL 경합 회피 (상단 안내 참조)
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
3. 끊긴 셀부터 재실행 → 캐시 히트로 완료된 variant 자동 스킵. **셀 4(dry-run)는 매번
   다시 실행할 필요 없다** — 이미 한 번 확인했다면 건너뛰고 바로 이어서 진행할 것.
4. `cache/round1/` 에 `stage1_var000.json` ~ `stage1_var011.json` (Stage 1),
   `stage2_var{V:03d}.json` (Stage 2, 생존자만) 로 진행 상황 확인. 같은 이름의
   `stage{S}_var{V:03d}_model.zip` 은 해당 variant 의 학습된 모델 체크포인트다
   (재평가/시각화용, 자동 삭제되지 않음 — FAILURE_LOG.md 참조).

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

## 9. 컴퓨팅 비용 추정 리포트

> Colab Pro 구독 검토용. `cache/round1/` 에 실측 GPU 시간이 얼마나 쌓였든(3개든 12개든)
> 그 시점까지 측정된 값만으로 실행 가능합니다. **독립 실행 가능** — 셀 1(Drive 마운트)만
> 먼저 실행되어 있으면 되고, 같은 세션에서 셀 4/5/6 을 다시 돌릴 필요는 없습니다.
> 컴퓨팅 단위 잔량은 API로 읽을 수 없어
> `UNITS_BEFORE`/`UNITS_AFTER`/`USD_PER_UNIT` 을 직접 입력해야 합니다(Colab 화면에서 확인).
>
> ⚠️ **Round 2/Step 1 예상치는 Round 2가 Round 1과 같은 2-stage 패턴을 따른다는 가정**
> (§16.5.B: Stage 1 36 variants → 50% 생존 18개 → Stage 2 2M)에 기반합니다. grid 크기나
> keep_ratio 가 바뀌면 이 추정도 다시 계산해야 합니다 — 출력에도 매번 이 가정을 함께 표시합니다.

```python
"""
Round 1~Step 1 컴퓨팅 비용 추정 리포트.
cache/round1/stage1_var*.json 에 기록된 실측 GPU 시간(train_time_sec + eval_time_sec)을
기반으로 Stage 2 / Round 2 / Step 1 전체 예상 시간을 추정한다.
셀 5(Stage 1)가 일부만 끝난 상태에서 실행해도 동작한다 (측정된 만큼만 집계).
독립 실행 가능 — 셀 1(Drive 마운트, PROJECT 정의)만 먼저 실행되어 있으면 되고,
같은 세션에서 셀 4/5/6 을 다시 돌릴 필요는 없다.
"""
import glob
import json
from statistics import mean

if "PROJECT" not in globals():
    raise RuntimeError("PROJECT 가 정의되지 않았습니다 — 먼저 셀 1(환경 설정)을 실행하세요.")
CACHE_DIR = f"{PROJECT}/cache/round1"

# ── (선택) 컴퓨팅 단위 환산 — Colab 화면에서 직접 확인한 값을 입력 ──────────
# UNITS_BEFORE: 이 리포트가 집계하는 GPU 시간이 "시작되기 전" 시점의 잔량
# UNITS_AFTER : 지금(이 셀 실행 시점) 잔량
# 정확하려면 타이밍 기록이 있는 variant들의 학습이 시작되기 *전*에 UNITS_BEFORE 를 읽어야 함.
UNITS_BEFORE = None   # 예: 152.3
UNITS_AFTER  = None   # 예: 148.7
USD_PER_UNIT = None   # 예: 0.0999 — Colab 결제 화면에 표시된 실제 단가(모르면 None 유지)

records = [json.loads(open(_f).read()) for _f in sorted(glob.glob(f"{CACHE_DIR}/stage1_var*.json"))]
timed = [r for r in records if r.get("train_time_sec") is not None and r.get("eval_time_sec") is not None]
gpu_times = [r["train_time_sec"] + r["eval_time_sec"] for r in timed]

print(f"=== Stage 1 캐시 {len(records)}개 중 시간 측정값 있는 variant: {len(timed)}개 ===")

if not timed:
    print("⚠️ 시간 측정값이 있는 variant가 아직 없습니다 — §9 적용 이전에 학습된 variant뿐이거나,")
    print("   아직 새 코드로 학습된 variant가 없습니다. 최소 1개 이상 새로 학습된 뒤 다시 실행하세요.")
else:
    gpu_names = sorted({str(r.get("gpu_name")) for r in timed if r.get("gpu_name")})
    print(f"GPU: {', '.join(gpu_names) if gpu_names else '(기록 없음)'}")

    avg_t, max_t, min_t = mean(gpu_times), max(gpu_times), min(gpu_times)
    print(f"\nvariant당 GPU 시간 — 평균 {avg_t/60:.1f}분 / 최대 {max_t/60:.1f}분 / 최소 {min_t/60:.1f}분")

    def fmt_hours(sec):
        return f"{sec/3600:.2f}시간"

    stage1_measured_total = sum(gpu_times)
    stage1_estimated_12 = avg_t * 12
    stage2_per_variant = avg_t * 8                 # 2M / 250K = 8배
    stage2_estimated = stage2_per_variant * 6       # Stage 1 생존자 6개
    round2_stage1_estimated = avg_t * 36
    round2_stage2_estimated = avg_t * 8 * 18         # ⚠️ 36 → 50% 생존 18개 가정
    round2_estimated = round2_stage1_estimated + round2_stage2_estimated
    round1_estimated = stage1_estimated_12 + stage2_estimated
    step1_estimated = round1_estimated + round2_estimated

    print(f"\nStage 1 누적 GPU 시간 — 실측 합계({len(timed)}개): {fmt_hours(stage1_measured_total)} "
          f"/ 12개 전체 추정: {fmt_hours(stage1_estimated_12)}")
    print(f"Stage 2 예상 (variant당 ×8: {fmt_hours(stage2_per_variant)}, 생존자 6개 합계): {fmt_hours(stage2_estimated)}")
    print(f"Round 1 합계 예상: {fmt_hours(round1_estimated)}")
    print(f"Round 2 예상 (36 variants): {fmt_hours(round2_estimated)}")
    print(f"  └ Stage 1(36개): {fmt_hours(round2_stage1_estimated)} / Stage 2(18개×8배): {fmt_hours(round2_stage2_estimated)}")
    print(f"Step 1 전체(Round 1~2 기준) 예상: {fmt_hours(step1_estimated)}")
    print("\n⚠️ 전제: Round 2가 Round 1과 같은 2-stage 패턴(Stage1 36개 → 50% 생존 18개 → Stage2)을")
    print("   따른다고 가정한 추정치입니다. grid 크기/keep_ratio 가 바뀌면 다시 계산해야 합니다.")

    # Stage 2 가 이미 일부 실행됐다면, ×8 추정과 실측을 비교해 가정의 정확도를 검증할 수 있다.
    stage2_records = [json.loads(open(_f).read()) for _f in sorted(glob.glob(f"{CACHE_DIR}/stage2_var*.json"))]
    stage2_timed = [r for r in stage2_records if r.get("train_time_sec") is not None and r.get("eval_time_sec") is not None]
    if stage2_timed:
        stage2_actual_avg = mean(r["train_time_sec"] + r["eval_time_sec"] for r in stage2_timed)
        print(f"\n(검증) Stage 2 실측 평균({len(stage2_timed)}개): {stage2_actual_avg/60:.1f}분 "
              f"vs ×8 추정: {stage2_per_variant/60:.1f}분")

    # ── 단위/달러 환산 ────────────────────────────────────────────────────
    if UNITS_BEFORE is not None and UNITS_AFTER is not None:
        consumed_units = UNITS_BEFORE - UNITS_AFTER
        if stage1_measured_total > 0:
            rate_per_hour = consumed_units / (stage1_measured_total / 3600)
            print(f"\n=== 단위 환산 (소모 {consumed_units:.2f} units / "
                  f"실측 {fmt_hours(stage1_measured_total)} 기준 → {rate_per_hour:.3f} units/시간) ===")

            def to_units(sec):
                return sec / 3600 * rate_per_hour

            for label, sec in [
                ("Stage 1 (12개 추정)", stage1_estimated_12),
                ("Stage 2 (6개)", stage2_estimated),
                ("Round 1 합계", round1_estimated),
                ("Round 2 (36개, 2-stage 가정)", round2_estimated),
                ("Step 1 전체(Round 1~2)", step1_estimated),
            ]:
                units = to_units(sec)
                if USD_PER_UNIT is not None:
                    print(f"  {label}: {units:.1f} units (${units * USD_PER_UNIT:.2f})")
                else:
                    print(f"  {label}: {units:.1f} units")
            if USD_PER_UNIT is None:
                print("  (USD_PER_UNIT 미입력 — 달러 환산 생략)")
        else:
            print("\n⚠️ 실측 GPU 시간 합계가 0이라 단위 환산이 불가능합니다.")
    else:
        print("\n(UNITS_BEFORE/UNITS_AFTER 미입력 — 시간만 출력)")

    # ── PROGRESS.md 기록용 블록 ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("# 아래를 PROGRESS.md 에 그대로 붙여넣을 수 있습니다")
    print("=" * 60)
    print(f"""
| 항목 | 값 |
|---|---|
| GPU | {', '.join(gpu_names) if gpu_names else '(기록 없음)'} |
| 실측 variant 수 | {len(timed)}개 (Stage 1) |
| variant당 평균 GPU 시간 | {avg_t/60:.1f}분 |
| Stage 1 실측 합계 | {fmt_hours(stage1_measured_total)} ({len(timed)}개) |
| Stage 1 전체 추정(12개) | {fmt_hours(stage1_estimated_12)} |
| Stage 2 예상(6개, ×8) | {fmt_hours(stage2_estimated)} |
| Round 1 합계 예상 | {fmt_hours(round1_estimated)} |
| Round 2 예상(36개, ⚠️ 36→18 2-stage 가정) | {fmt_hours(round2_estimated)} |
| Step 1 전체(Round 1~2) 예상 | {fmt_hours(step1_estimated)} |
""")
```

---

## 타임라인 예상 (무료 Colab T4, worker 1 — 순차 실행)

| 단계 | 예상 시간 | 세션 내 완료 여부 |
|------|---------|----------------|
| dry-run (12 × 5K) | ~3~5분 | ✅ |
| Stage 1 (12 × 250K, worker 1) | ~3~4시간 | ✅ (12시간 내) |
| Stage 2 (6 × 2M, worker 1) | ~12~14시간 | ⚠️ 세션 1개로 안 끝남 — 캐시로 이어서 진행 |
| **Round 1 합계** | **~15~18시간** | ⚠️ 최소 2세션 필요. 무료 tier 사용량 한도는 고정 12시간이 아니라 최근 사용량에 따라 변동하므로 3세션 이상 걸릴 수 있음 |

> **팁**: Stage 1 완료 후 결과 저장 확인 (셀 5 마지막 줄) → Stage 2는 새 세션에서 시작해도 캐시로 복구됩니다. Stage 2 개별 variant(약 2.2시간)는 세션 한도보다 훨씬 짧으므로, 세션이 끊겨도 대부분 "variant 경계"에서 끊기고 §8 의 "학습 도중" 복구 절차까지 필요한 경우는 드뭅니다.

---

## 부록 A. ⚠️ 위험 작업 — cache/round1/ 전체 삭제

**이 셀은 진짜 Stage 1/2 결과(JSON 결과 + 모델 체크포인트)를 되돌릴 수 없이 삭제합니다.**
정상 실행 흐름(셀 0~8)에는 포함돼 있지 않고, "Run all" 로 노트북 전체를 실행해도
안전하도록 아래 `CONFIRM` 문자열을 정확히 고쳐 쓰지 않으면 아무 것도 삭제되지
않습니다. **정말로 Round 1 을 처음부터 다시 시작해야 할 때만** 사용하세요 — 세션
끊김 복구는 §8 절차로 충분하며 이 셀이 필요하지 않습니다.

```python
import os
import shutil

CONFIRM = ""  # 정말 삭제하려면 이 줄만 CONFIRM = "DELETE ROUND1 CACHE" 로 고친 뒤 실행

if CONFIRM == "DELETE ROUND1 CACHE":
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    _db_path = f"{PROJECT}/autoresearch_round1.db"
    if os.path.exists(_db_path):
        os.remove(_db_path)
    print(f"삭제 완료: {CACHE_DIR}, {_db_path}")
else:
    print("삭제되지 않았습니다 — CONFIRM 문자열이 일치하지 않습니다 (의도된 안전장치입니다).")
```
