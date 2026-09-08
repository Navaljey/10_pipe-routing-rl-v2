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
> 삭제한 사고 이후 정리). 아래 표는 **본 문서/코드가 실제로 참조하는 디렉터리 전체**다 —
> Drive 에 이 표에 없는 `cache/*` 폴더가 보이면 옛 이름의 잔여물일 가능성이 높으니
> "정리 대상" 항목을 확인할 것.
>
> | 디렉터리 | 용도 | 결과 성격 | 자동 삭제 여부 |
> |---|---|---|---|
> | `cache/round1/` | Stage 1/2 **진짜** 결과(JSON + 모델 체크포인트) | 영구 보존 | **자동 삭제 절대 안 됨** — 부록 A 에서만, 확인 문구 입력 시에만 |
> | `cache/round1_smoke/` | 셀 4 dry-run(5K) 전용 | 버릴 데이터 | 셀 4 실행마다 자동 정리됨 |
> | `cache/round1_cli_smoke/` | `scripts/dryrun_round1.py` — 로컬/CI 전용, **Colab 노트북과 무관** | 버릴 데이터 | 그 스크립트 실행마다 자동 정리됨 |
> | `cache/round1_seedcheck/` | §10 var008 3-seed 분산 체크 전용 (`cache/round1/` 과 절대 안 겹침) | 영구 보존(진단 근거) | 자동 삭제 없음 — 결과 재사용(캐시 히트)을 위해 유지 |
> | `cache/round1_wiring_check/` | §11 w1/w2/w3 배선 수정 검증 전용 (이번 실행 경로에서는 §11 대신 아래 `round2_phase1_prescreen/` 로 검증) | 영구 보존(진단 근거) | 자동 삭제 없음 — 결과 재사용(캐시 히트)을 위해 유지 |
> | `cache/round2_phase1_prescreen/` | §12 Round 2 Phase 1(w1/w2/w3 효과 크기 사전 스크리닝, 6 variants) 전용 | 영구 보존(진단 근거) | 자동 삭제 없음 — 결과 재사용(캐시 히트)을 위해 유지 |
> | `cache/round2_phase2/` | §13 Round 2 Phase 2(w2×w3 9-cell × 2-seed 본 grid, 18 variants) 전용 | 영구 보존(Round 2 본 결과) | 자동 삭제 없음 |
>
> **정리 대상 (사용자 확인 후 수동 삭제 — 본 문서에 자동 삭제 코드를 넣지 않음)**:
> `cache/round1_dry/`, `cache/dryrun_round1/` 이 Drive 에 남아 있다면 각각 위 표의
> `round1_smoke/`, `round1_cli_smoke/` 로 **이름만 바뀐** 옛 폴더다(FAILURE_LOG.md
> 2026-09-01 entry 참조). 현재 코드/문서 어디에서도 이 옛 이름을 더 이상 참조하지
> 않으며, 애초에 담긴 내용도 dry-run/smoke 용 버릴 데이터였다(진짜 250K/2M 결과는
> 처음부터 `cache/round1/` 한 곳에만 저장됐다) — 안전하게 삭제해도 된다.

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

def make_train_fn_real(n_envs=1, handoff_dir=None, cache_dir=None, round_n=1):
    """실제 Step1Env 기반 train_fn.

    cache_dir 는 반드시 호출부에서 명시적으로 전달한다 (dry-run 은 SMOKE_CACHE_DIR,
    본 실행은 CACHE_DIR). 체크포인트를 여기 저장하므로, 전역 CACHE_DIR 을 암묵적으로
    참조하면 dry-run 이 진짜 캐시 폴더에 체크포인트를 잘못 쓰거나 셀 5/6 실행 시점에
    따라 저장 위치가 바뀌는 등 dry-run/본 실행이 서로 뒤섞일 위험이 있다.

    round_n 은 wandb run 이름/tag(§16.6.6, `build_run_name` 등)에만 쓰인다 — 기본값 1
    은 셀 4/5/6(Round 1) 호출부를 그대로 유지하기 위함이다. Round 2 이상을 학습하는
    새 셀에서는 반드시 `round_n=2` 처럼 명시적으로 넘겨야 wandb 에서 Round 1 결과와
    섞이지 않는다 (2026-09 발견: §12 Phase 1 이 이 인자 없이 호출되어 실제로는 Round 2
    학습인데 wandb run 이름이 `step1_round1_stage1_var*` 로 찍히는 문제가 있었다 —
    이미 실행 중인 Phase 1 은 재현성을 위해 그대로 두고, 이 함수만 향후 호출부가
    올바르게 쓸 수 있도록 고쳤다).
    """
    assert cache_dir is not None, "cache_dir 를 명시적으로 전달하세요 (SMOKE_CACHE_DIR 또는 CACHE_DIR)"
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

    def train_fn(params: dict, timesteps: int, variant_id: int) -> float:
        alpha = params.get("alpha", 1.0)
        beta  = params.get("beta", 0.1)
        # w1/w2/w3 는 make_env_fn() 을 거쳐 Step1Env(w1=..., w2=..., w3=...) 로
        # 실제 전달된다 (2026-09 FAILURE_LOG: 이전에는 여기서 읽기만 하고
        # make_env_fn() 호출에 안 넘겨서 Round 2 sweep 이 전부 no-op 이었다).
        w1    = params.get("w1", 0.1)
        w2    = params.get("w2", 2.0)
        w3    = params.get("w3", 50.0)

        vec_env = DummyVecEnv([
            make_env_fn(150000 + variant_id * 100 + i, alpha, beta, "medium",
                        w1=w1, w2=w2, w3=w3)
            for i in range(n_envs)
        ])
        vec_env = VecMonitor(vec_env)

        model = build_model(vec_env, lr=3e-4, n_steps=2048, batch_size=64, ent_coef=0.01)

        # wandb 연동 (autoresearch 명명 체계 §16.6.6)
        import wandb
        from autoresearch.wandb_callback import build_run_name, build_run_config, build_run_tags
        from wandb.integration.sb3 import WandbCallback
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

## 10. var008 3-seed 분산 체크

> Stage 1 상위 생존자들의 score 차이가 seed 편차보다 큰 신호인지 판별하기 위한 진단 셀
> (PROGRESS.md 2026-09-01 entry 참조). **독립 실행 가능** — 셀 1(Drive 마운트)과 Stage 1
> (셀 5) 완료만 전제로 하며, 같은 세션에서 셀 4/5/6 을 다시 돌릴 필요는 없습니다.
>
> `cache/round1/`(진짜 Stage 1/2 결과)은 이 셀에서 **읽기만** 하고 절대 쓰지 않습니다 —
> 결과는 전부 별도 디렉터리 `cache/round1_seedcheck/` 에 저장됩니다.

```python
"""
var008(α=2.0, β=0.3) 3-seed 분산 체크 — 250K 신규 2개 + 기존 var008 결과 재사용.
목적: Stage 1 상위 생존자 간 score 차이가 seed 편차보다 큰 신호인지 판별
(PROGRESS.md 2026-09-01 entry 참조).

결과는 cache/round1_seedcheck/ 에 저장 — cache/round1/(진짜 Stage 1/2 결과)은
이 셀에서 오직 읽기만 하고 절대 쓰지 않는다.

seed 가 실제로 바꾸는 것: make_env_fn() 의 seed 는 Step1Env 생성자로 들어가는데,
"train" 모드에서는 _train_seed_counter 가 항상 150000 부터 시작하는 기존 이슈
(PROGRESS.md 참조) 때문에 3개 실행의 시나리오(미로) "순서" 자체는 동일하다.
대신 매 episode 의 start_dir_idx/goal_dir_idx(강제 출발/목표 방향)는 seed 로
초기화되는 _episode_rng 에서 뽑히므로 seed 마다 확실히 달라진다 — 아래 사전 점검이
학습 시작 전에 이를 직접 확인/assert 한다. 여기에 더해 build_model() 이 MaskablePPO 를
seed= 없이 생성하므로(training/train_step1.py) PPO 자체의 신경망 초기화/rollout
샘플링도 매 실행마다 시드되지 않은 무작위성을 갖는다 — 두 가지 독립적인 변동 요인이
있어 "seed 를 바꿔도 결과가 우연히 완전히 같다"는 시나리오는 사실상 배제된다.
"""
import json
import time
from pathlib import Path
import torch

if "PROJECT" not in globals():
    raise RuntimeError("PROJECT 가 정의되지 않았습니다 — 먼저 셀 1(환경 설정)을 실행하세요.")
CACHE_DIR = f"{PROJECT}/cache/round1"

from training.train_step1 import make_env_fn, build_model
from envs.step1_env import Step1Env

# GPU 확인 (하드 스톱) — 셀 5/6 과 동일
REQUIRE_GPU = True
if torch.cuda.is_available():
    print(f"✅ GPU 사용 중: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU 를 사용할 수 없습니다 — CPU 로 진행하면 수 배 느려집니다.")
    if REQUIRE_GPU:
        raise RuntimeError(
            "GPU 미사용 — REQUIRE_GPU=True(기본값) 라 중단합니다. "
            "GPU 없이 계속하려면 이 셀 상단의 REQUIRE_GPU = False 로 바꾸세요."
        )

N_SEEDS = 3   # 애매하면(range 가 판별 기준 근처) 5 로 늘려 재실행 가능
SEEDCHECK_CACHE_DIR = f"{PROJECT}/cache/round1_seedcheck"   # cache/round1/ 과 절대 안 겹침
Path(SEEDCHECK_CACHE_DIR).mkdir(parents=True, exist_ok=True)

VAR008_ALPHA, VAR008_BETA = 2.0, 0.3   # Stage 1 var008 파라미터

# ── seed 사전 점검: 신규 슬롯들이 실제로 다른 방향 시퀀스를 타는지 학습 전에 확인 ──
def _direction_sequence(seed, n=5):
    env = Step1Env(mode="train", difficulty="medium", seed=seed)
    seq = []
    for _ in range(n):
        env.reset()
        seq.append((env.start_dir_idx, env.goal_dir_idx))
    env.close()
    return seq

_new_slot_ids = list(range(1, N_SEEDS))   # slot 0 = var008 재사용, 1..N-1 = 신규
_seeds = {slot: 150000 + slot * 100 for slot in _new_slot_ids}
print(f"=== 신규 슬롯 seed: {_seeds} (참고: var008 원래 seed = 150800, 재사용만 함) ===")
assert len(set(_seeds.values())) == len(_seeds), "신규 슬롯 seed 가 서로 겹칩니다 — 버그"

_dir_seqs = {slot: _direction_sequence(seed) for slot, seed in _seeds.items()}
for slot, seq in _dir_seqs.items():
    print(f"  slot {slot} (seed={_seeds[slot]}) 방향 시퀀스 샘플: {seq}")
_unique_seqs = {tuple(s) for s in _dir_seqs.values()}
assert len(_unique_seqs) == len(_dir_seqs), (
    "신규 슬롯들의 방향 시퀀스가 동일합니다 — seed 가 실제로 안 바뀌고 있다는 뜻이므로 "
    "학습을 시작하지 말고 원인을 먼저 확인하세요."
)
print("✅ 신규 슬롯들의 학습 조건(강제 방향 시퀀스)이 서로 다름을 확인 — 학습 진행\n")

# ── 1) 기존 var008 결과를 slot 0 으로 재사용 (재학습하지 않음) ──────────────
var008_path = Path(CACHE_DIR) / "stage1_var008.json"
assert var008_path.exists(), (
    f"{var008_path} 를 찾을 수 없습니다 — Stage 1(셀 5)이 완료된 상태에서 실행하세요."
)
slot0_path = Path(SEEDCHECK_CACHE_DIR) / "stage1_var000.json"
if not slot0_path.exists():
    var008_data = json.loads(var008_path.read_text())
    seed1_data = dict(var008_data)
    seed1_data["variant_id"] = 0   # 이 리스트 안에서의 위치로 재매핑 (α/β 값 자체는 그대로)
    slot0_path.write_text(json.dumps(seed1_data), encoding="utf-8")
print(f"slot 0(기존 var008 재사용): score={json.loads(slot0_path.read_text())['metric']:.4f}")

# ── 2) train_fn — make_train_fn_real() 과 거의 동일하지만 wandb run 이름을 seedcheck
#    전용으로 분리한다(실제 Stage1/2 run 이름과 겹치면 wandb 대시보드에서 혼동되므로
#    의도적으로 별도 구현 — make_train_fn_real() 자체는 건드리지 않는다). ─────────
def make_seedcheck_train_fn(cache_dir):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

    def train_fn(params: dict, timesteps: int, variant_id: int) -> tuple[float, float, str]:
        alpha, beta = params["alpha"], params["beta"]
        vec_env = DummyVecEnv([make_env_fn(150000 + variant_id * 100, alpha, beta, "medium")])
        vec_env = VecMonitor(vec_env)
        model = build_model(vec_env, lr=3e-4, n_steps=2048, batch_size=64, ent_coef=0.01)

        import wandb
        from wandb.integration.sb3 import WandbCallback
        wandb.init(
            project="pipe-routing-rl",
            name=f"step1_round1_seedcheck_var008_slot{variant_id}",
            config={"alpha": alpha, "beta": beta, "purpose": "seed_variance_check",
                    "base_variant": "var008", "seed": 150000 + variant_id * 100},
            tags=["step1", "round1", "seedcheck"],
            reinit=True,
        )
        model.learn(total_timesteps=timesteps,
                    callback=WandbCallback(gradient_save_freq=0, verbose=0))
        model.save(f"{cache_dir}/stage1_var{variant_id:03d}_model")

        t0 = time.time()
        from training.train_step1 import _evaluate_screening
        result = _evaluate_screening(model, alpha, beta)
        eval_time_sec = time.time() - t0
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

        wandb.log({"eval/success_rate": result["success_rate"], "eval/score": result["score"]})
        wandb.finish()
        vec_env.close()
        return result["score"], eval_time_sec, gpu_name

    return train_fn

# ── 3) N_SEEDS 개 모두 같은 (α, β) — stage1_keep_ratio=1.0 으로 전부 유지 ──────
from autoresearch.stage_runner import StageRunner
runner = StageRunner(
    train_fn=make_seedcheck_train_fn(SEEDCHECK_CACHE_DIR),
    max_workers=1,
    stage1_timesteps=250_000,
    stage1_keep_ratio=1.0,   # screening 이 아니라 분산 측정이므로 탈락 없음
    cache_dir=SEEDCHECK_CACHE_DIR,
)

variants = [{"alpha": VAR008_ALPHA, "beta": VAR008_BETA}] * N_SEEDS
_hit = sum(
    1 for i in range(len(variants))
    if (Path(SEEDCHECK_CACHE_DIR) / f"stage1_var{i:03d}.json").exists()
)
print(f"\n{len(variants)}개 중 {_hit}개 캐시 재사용, {len(variants) - _hit}개 신규 학습 "
      f"(예상 추가 소요: 최대 {(len(variants) - _hit) * 50}분)")

results = runner.run_stage1(variants)   # keep_ratio=1.0 이라 전부 반환됨
scores = sorted(r.metric for r in results)

# ── 4) 통계 + 판정 ──────────────────────────────────────────────────────────
import statistics

mean_score = statistics.mean(scores)
std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0
range_score = max(scores) - min(scores)
THRESHOLD = 1 / 75   # Stage 1 상위4-하위2 실측 차이(=1 success/75) 와 동일 기준

print(f"\n=== var008 {len(scores)}-seed 결과: {[f'{s:.4f}' for s in scores]} ===")
print(f"평균: {mean_score:.4f} / 표본표준편차: {std_score:.4f} / range(max-min): {range_score:.4f}")
print(f"판별 기준(1/75): {THRESHOLD:.4f}  ※ range 기준으로 비교 (원래 신호도 max-min 형태였음)")

if range_score > THRESHOLD:
    print(f"\n결론: range({range_score:.4f}) > {THRESHOLD:.4f} → 현재 Stage 1 순위는 노이즈 "
          "지배적일 가능성이 큽니다. Stage 2 를 이 순위 그대로 투입하는 근거가 약합니다 — "
          "임의 조합(예: var008) 고정 후 Round 2(w1·w2·w3)로 이동을 권장합니다.")
else:
    print(f"\n결론: range({range_score:.4f}) < {THRESHOLD:.4f} → 상위권 동률이 noise 안에서도 "
          "유지될 가능성이 있습니다. 축소 Stage 2(상위 2~3개)로 진행을 고려할 수 있습니다.")
    print("(range 가 기준에 가깝다면 N_SEEDS = 5 로 늘려 재실행해 더 확실히 판별하세요.)")
```

---

## 11. w1/w2/w3 배선 수정 검증 (var008 1회 재현 체크)

> 2026-09 FAILURE_LOG: w1/w2/w3 가 CLI 인자·params dict 에는 있었지만 실제 보상 계산
> (`Step1Env._calc_reward()`)에는 한 번도 전달되지 않던 버그를 수정했다 (envs/step1_env.py,
> training/train_step1.py::make_env_fn(), 본 문서 §5/§6 의 `make_train_fn_real`). 수정은
> 하위호환(기본값 = 기존 모듈 상수) 방식이라 코드 레벨 테스트(`tests/test_reward_baseline.py`,
> `tests/test_train_step1_smoke.py`)로는 "기본값을 쓰면 이전과 동일한 함수" 임을 이미
> 확인했지만, GPU 가 없는 환경(Claude Code CLI)에서는 실제 250K 학습 재현까지는 못 했다.
> 이 셀은 그 마지막 경험적 확인 단계 — **수정된 코드로 var008 을 1회(250K) 다시 학습해
> 기존 cache/round1/ 결과 및 §10 3-seed 노이즈 대역과 비교**한다.
>
> `cache/round1/`, `cache/round1_seedcheck/` 은 이 셀에서 **읽기만** 하고 절대 쓰지
> 않는다 — 결과는 `cache/round1_wiring_check/` 전용 디렉터리에 저장한다.

```python
"""
w1/w2/w3 배선 수정 검증 — var008(α=2.0, β=0.3, w1=0.1/w2=2.0/w3=50.0 기본값)을
수정된 코드로 1회(250K) 재학습해 기존 결과와 비교.

판정: 새 score 가 기존 var008 대비 §10 에서 측정한 노이즈 range(0.0135) 안에 있으면
"배선 수정이 기존 결과를 재현함(회귀 없음)"으로 판정한다. 밖에 있으면 배선 수정
자체가 아닌 다른 변화(예: 코드의 다른 부분이 의도치 않게 함께 바뀜)가 있었는지
먼저 의심해야 한다.
"""
import json
import time
from pathlib import Path
import torch

if "PROJECT" not in globals():
    raise RuntimeError("PROJECT 가 정의되지 않았습니다 — 먼저 셀 1(환경 설정)을 실행하세요.")
CACHE_DIR = f"{PROJECT}/cache/round1"
SEEDCHECK_CACHE_DIR = f"{PROJECT}/cache/round1_seedcheck"
WIRING_CHECK_CACHE_DIR = f"{PROJECT}/cache/round1_wiring_check"   # 전용 디렉터리 — round1/, round1_seedcheck/ 절대 안 건드림
Path(WIRING_CHECK_CACHE_DIR).mkdir(parents=True, exist_ok=True)

from training.train_step1 import make_env_fn, build_model, DEFAULT_W1, DEFAULT_W2, DEFAULT_W3

# GPU 확인 (하드 스톱) — 셀 5/6/10 과 동일
REQUIRE_GPU = True
if torch.cuda.is_available():
    print(f"✅ GPU 사용 중: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU 를 사용할 수 없습니다 — CPU 로 진행하면 수 배 느려집니다.")
    if REQUIRE_GPU:
        raise RuntimeError(
            "GPU 미사용 — REQUIRE_GPU=True(기본값) 라 중단합니다. "
            "GPU 없이 계속하려면 이 셀 상단의 REQUIRE_GPU = False 로 바꾸세요."
        )

VAR008_ALPHA, VAR008_BETA = 2.0, 0.3
THRESHOLD = 1 / 75   # §10 에서 확정한 측정 한계(0.0135)와 비교할 기준값

# ── 기존 결과 로드 (비교 기준선) ────────────────────────────────────────────
var008_path = Path(CACHE_DIR) / "stage1_var008.json"
assert var008_path.exists(), f"{var008_path} 를 찾을 수 없습니다 — Stage 1(셀 5)이 완료된 상태에서 실행하세요."
var008_score = json.loads(var008_path.read_text())["metric"]
print(f"기존 var008(cache/round1/) score = {var008_score:.4f} (수정 전 코드, 참조용)")

seedcheck_scores = sorted(
    json.loads(p.read_text())["metric"]
    for p in Path(SEEDCHECK_CACHE_DIR).glob("stage1_var*.json")
) if Path(SEEDCHECK_CACHE_DIR).exists() else []
if seedcheck_scores:
    print(f"§10 3-seed 결과(cache/round1_seedcheck/): {[f'{s:.4f}' for s in seedcheck_scores]} "
          f"(range={max(seedcheck_scores)-min(seedcheck_scores):.4f})")

# ── 신규 1회 실행 — 수정된 make_env_fn() 이 w1/w2/w3 를 실제로 넘기는 경로 사용 ──
result_path = Path(WIRING_CHECK_CACHE_DIR) / "stage1_var000.json"
if result_path.exists():
    new_score = json.loads(result_path.read_text())["metric"]
    print(f"\n캐시 재사용: 이미 실행된 결과가 있습니다 (재실행하려면 {result_path} 를 삭제하세요).")
else:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

    # w1/w2/w3 를 "명시적으로" 넘긴다 — 기본값에 의존하지 않고 실제 배선 경로
    # (make_env_fn(..., w1=, w2=, w3=) → Step1Env(w1=, w2=, w3=)) 를 정확히 왕복시킨다.
    vec_env = DummyVecEnv([make_env_fn(
        150000, VAR008_ALPHA, VAR008_BETA, "medium",
        w1=DEFAULT_W1, w2=DEFAULT_W2, w3=DEFAULT_W3,
    )])
    vec_env = VecMonitor(vec_env)
    model = build_model(vec_env, lr=3e-4, n_steps=2048, batch_size=64, ent_coef=0.01)

    import wandb
    from wandb.integration.sb3 import WandbCallback
    wandb.init(
        project="pipe-routing-rl",
        name="step1_round1_wiring_check_var008",
        config={"alpha": VAR008_ALPHA, "beta": VAR008_BETA,
                "w1": DEFAULT_W1, "w2": DEFAULT_W2, "w3": DEFAULT_W3,
                "purpose": "w1_w2_w3_wiring_regression_check"},
        tags=["step1", "round1", "wiring_check"],
        reinit=True,
    )
    model.learn(total_timesteps=250_000, callback=WandbCallback(gradient_save_freq=0, verbose=0))
    model.save(f"{WIRING_CHECK_CACHE_DIR}/stage1_var000_model")

    from training.train_step1 import _evaluate_screening
    eval_result = _evaluate_screening(model, VAR008_ALPHA, VAR008_BETA)
    new_score = eval_result["score"]
    wandb.log({"eval/success_rate": eval_result["success_rate"], "eval/score": new_score})
    wandb.finish()
    vec_env.close()

    result_path.write_text(json.dumps({
        "variant_id": 0, "params": {"alpha": VAR008_ALPHA, "beta": VAR008_BETA,
                                     "w1": DEFAULT_W1, "w2": DEFAULT_W2, "w3": DEFAULT_W3},
        "metric": new_score, "timesteps": 250_000, "stage": 1,
    }), encoding="utf-8")

print(f"\n=== 배선 수정 후 신규 var008 재현 score: {new_score:.4f} ===")
diff_vs_cached = abs(new_score - var008_score)
print(f"기존 cache/round1/ var008 대비 차이: {diff_vs_cached:.4f} (판정 기준 {THRESHOLD:.4f})")

if diff_vs_cached <= THRESHOLD:
    print("\n결론: 차이가 측정 한계(0.0135) 안입니다 → 배선 수정이 기존 결과를 통계적으로 "
          "재현합니다(회귀 없음). Phase 1(효과 크기 사전 추정) 진행 가능.")
else:
    print("\n⚠️ 결론: 차이가 측정 한계(0.0135) 를 초과합니다 → 배선 수정 외에 다른 변화가 "
          "섞였을 가능성이 있습니다. Phase 1 로 넘어가기 전에 diff 를 다시 검토하세요 "
          "(git diff, envs/step1_env.py 의 _calc_reward() 등).")
```

---

## 12. Round 2 Phase 1 — w1/w2/w3 효과 크기 사전 스크리닝 (6 × 250K)

> 2026-09 PROGRESS.md 의사결정 31: 이번 실행 경로에서는 §11(var008 1회 재현 체크)을
> 건너뛰고 배선 검증을 본 Phase 1 결과로 대체한다. §11 은 all-or-nothing 구조라
> 세션이 끊기면 진행분 회수가 안 되고(실측: 31% 지점에서 중단, ~1 unit 유실), 1-seed
> 비교라 판정 기준(0.0135)과 노이즈(3-seed range 0.0135)가 같은 크기라 배선 영향과
> seed 편차를 구분하지 못한다. 반면 본 셀은 (a) §5 와 동일한 `make_train_fn_real()` 을
> 그대로 재사용해 실전 학습 경로 자체로 검증하고, (b) w1/w2/w3 극값 간 score 가 실제로
> 갈리면 그 자체가 배선이 살아있다는 더 직접적인 증거이며, (c) variant 단위 캐시라
> 중간에 끊겨도 진행분이 보존된다.
>
> `cache/round1/` (진짜 Stage 1/2 결과)은 이 셀에서 var008 기준선을 **읽기만** 하고
> 절대 쓰지 않는다 — 결과는 `cache/round2_phase1_prescreen/` 전용 디렉터리에 저장한다.

```python
"""
Round 2 Phase 1 — w1/w2/w3 효과 크기 사전 스크리닝.
3축(w1/w2/w3) 각각 grid 양극단 2개, 나머지 두 인자는 baseline(w1=0.1/w2=2.0/w3=50.0)
고정, alpha/beta 는 var008 고정값(2.0/0.3). 총 6 variants × 250K.

목적: Round 2(36 variants) 본 실행 전, 각 축이 250K 스케일에서 노이즈(0.0135)를 넘는
신호를 만드는지 확인해 Phase 2 grid 축소 폭을 데이터 기반으로 정한다
(PROGRESS.md 의사결정 30/31 참조).

train_fn 은 셀 4 에서 정의한 make_train_fn_real() 을 그대로 재사용한다 — §5(Stage 1)와
동일한 코드 경로이므로 w1/w2/w3 배선(PR #9) 검증도 겸한다.
"""
import json
from pathlib import Path
import torch

if "PROJECT" not in globals():
    raise RuntimeError("PROJECT 가 정의되지 않았습니다 — 먼저 셀 1(환경 설정)을 실행하세요.")
CACHE_DIR = f"{PROJECT}/cache/round1"
PHASE1_CACHE_DIR = f"{PROJECT}/cache/round2_phase1_prescreen"   # round1/ 과 절대 안 겹침
Path(PHASE1_CACHE_DIR).mkdir(parents=True, exist_ok=True)

# GPU 확인 (하드 스톱) — 셀 5/10/11 과 동일
REQUIRE_GPU = True
if torch.cuda.is_available():
    print(f"✅ GPU 사용 중: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU 를 사용할 수 없습니다 — CPU 로 진행하면 수 배 느려집니다.")
    if REQUIRE_GPU:
        raise RuntimeError(
            "GPU 미사용 — REQUIRE_GPU=True(기본값) 라 중단합니다. "
            "GPU 없이 계속하려면 이 셀 상단의 REQUIRE_GPU = False 로 바꾸세요."
        )

VAR008_ALPHA, VAR008_BETA = 2.0, 0.3
BASELINE_W1, BASELINE_W2, BASELINE_W3 = 0.1, 2.0, 50.0
THRESHOLD = 0.0135   # PROGRESS.md 의사결정 28 — 3-seed 실측 range, 앞으로도 쓰는 판정 기준

# (축 이름, 극값 라벨, variant params) — index 순서가 곧 variant_id(0~5)가 된다.
_AXIS_PLAN = [
    ("w1", "min", {"alpha": VAR008_ALPHA, "beta": VAR008_BETA, "w1": 0.05, "w2": BASELINE_W2, "w3": BASELINE_W3}),
    ("w1", "max", {"alpha": VAR008_ALPHA, "beta": VAR008_BETA, "w1": 0.5,  "w2": BASELINE_W2, "w3": BASELINE_W3}),
    ("w2", "min", {"alpha": VAR008_ALPHA, "beta": VAR008_BETA, "w1": BASELINE_W1, "w2": 1.0, "w3": BASELINE_W3}),
    ("w2", "max", {"alpha": VAR008_ALPHA, "beta": VAR008_BETA, "w1": BASELINE_W1, "w2": 5.0, "w3": BASELINE_W3}),
    ("w3", "min", {"alpha": VAR008_ALPHA, "beta": VAR008_BETA, "w1": BASELINE_W1, "w2": BASELINE_W2, "w3": 20.0}),
    ("w3", "max", {"alpha": VAR008_ALPHA, "beta": VAR008_BETA, "w1": BASELINE_W1, "w2": BASELINE_W2, "w3": 100.0}),
]
variants = [params for _, _, params in _AXIS_PLAN]

# ── 기준선: var008(w1/w2/w3 모두 baseline) 기존 결과를 재사용, 재학습하지 않음 ──
var008_path = Path(CACHE_DIR) / "stage1_var008.json"
assert var008_path.exists(), (
    f"{var008_path} 를 찾을 수 없습니다 — Stage 1(셀 5)이 완료된 상태에서 실행하세요."
)
var008_score = json.loads(var008_path.read_text())["metric"]
print(f"기준선(var008, cache/round1/) score = {var008_score:.4f}")

_cache_hit_n = sum(
    1 for i in range(len(variants)) if (Path(PHASE1_CACHE_DIR) / f"stage1_var{i:03d}.json").exists()
)
print(f"{len(variants)}개 중 {_cache_hit_n}개 캐시 재사용, {len(variants) - _cache_hit_n}개 신규 학습 "
      f"(예상 추가 소요: 최대 {(len(variants) - _cache_hit_n) * 50}분)")

from autoresearch.stage_runner import StageRunner

# ⚠️ 알려진 제한사항(2026-09): 아래 호출이 round_n 을 안 넘겨 wandb run 이름이
# `step1_round1_stage1_var*` 로 찍힌다 — 실제로는 Round 2 Phase 1 인데 Round 1 로
# 표시되어 wandb 대시보드에서 기존 Round 1 결과와 구분이 안 된다. 이미 실행 중인
# Phase 1 은 재현성(세션 재개 시 동일 코드 유지)을 위해 의도적으로 그대로 둔다 —
# 이 실행의 결과는 `cache/round2_phase1_prescreen/` 파일명으로 이미 명확히
# 구분되므로 분석에는 지장 없다. Phase 1 을 처음부터 다시 돌리거나 Phase 2 셀을
# 새로 작성할 때는 반드시 `round_n=2` 를 명시할 것 (make_train_fn_real 정의부 참조).
runner = StageRunner(
    train_fn=make_train_fn_real(n_envs=1, cache_dir=PHASE1_CACHE_DIR),
    max_workers=1,               # ThreadPoolExecutor GIL 경합 회피 (§5 상단 안내 참조)
    stage1_timesteps=250_000,
    stage1_keep_ratio=1.0,       # 스크리닝이 아니라 효과 크기 측정이므로 탈락 없음
    cache_dir=PHASE1_CACHE_DIR,
)

print(f"\n=== Phase 1 시작: {len(variants)} variants × 250K ===")
results = runner.run_stage1(variants)   # keep_ratio=1.0 이라 전부 반환됨
scores = {r.variant_id: r.metric for r in results}

# ── 축별 판정 ─────────────────────────────────────────────────────────────
# ⚠️ 2026-09 FAILURE_LOG: 최초 버전은 var008 기준선과의 Δ(max(diff_lo, diff_hi))로
# 판정했으나 이는 잘못된 비교였다 — var008 은 §10 3-seed 실험에서 0.9187~0.9322 로
# 흔들린 값의 상단이라, "극값이 그 흔들리는 점 하나와 얼마나 다른가"는 축의 효과와
# 무관하게 baseline 자체의 위치에 따라 왜곡된다. 축의 효과는 그 축의 두 극값끼리
# 비교(axis_range)해야 한다 — 실제로 w1 은 값을 10배(0.05→0.5) 바꿔도 axis_range 가
# 노이즈(0.0135)의 2%(0.0003)에 불과해 명백히 효과 없음인데, 첫 버전은 "신호 있음"으로
# 오판정했다(자세한 원인은 FAILURE_LOG.md 참조). 판정 기준을 axis_range 로 수정한다.
print(f"\n{'='*70}\nRound 2 Phase 1 — 축별 효과 크기 (기준선 var008={var008_score:.4f}, "
      f"판정 기준={THRESHOLD:.4f})\n{'='*70}")

_axis_verdicts = {}
for axis in ("w1", "w2", "w3"):
    lo_id = next(i for i, (a, lbl, _) in enumerate(_AXIS_PLAN) if a == axis and lbl == "min")
    hi_id = next(i for i, (a, lbl, _) in enumerate(_AXIS_PLAN) if a == axis and lbl == "max")
    score_lo, score_hi = scores[lo_id], scores[hi_id]
    axis_range = abs(score_hi - score_lo)     # 판정 기준 — 이 축의 실제 효과 크기
    diff_lo = score_lo - var008_score          # 참고용 — 판정에는 사용하지 않음
    diff_hi = score_hi - var008_score          # 참고용 — 판정에는 사용하지 않음
    signal = axis_range > THRESHOLD
    _axis_verdicts[axis] = signal

    print(f"\n[{axis}] min={_AXIS_PLAN[lo_id][2][axis]} → score={score_lo:.4f} "
          f"(참고: baseline 대비 {diff_lo:+.4f})")
    print(f"      max={_AXIS_PLAN[hi_id][2][axis]} → score={score_hi:.4f} "
          f"(참고: baseline 대비 {diff_hi:+.4f})")
    print(f"      극값 간 range={axis_range:.4f}  ← 판정 기준")
    if signal:
        favored = "min" if score_lo > score_hi else "max"
        print(f"      → 판정: 신호 있음 (range {axis_range:.4f} > 임계값 {THRESHOLD:.4f}) "
              f"— sweep 유지 권장, {favored} 방향({_AXIS_PLAN[lo_id if favored=='min' else hi_id][2][axis]}) 우세")
    else:
        print(f"      → 판정: 신호 없음 (range {axis_range:.4f} ≤ 임계값 {THRESHOLD:.4f}) "
              f"— 해당 인자는 sweep 대상에서 제외 권장")

n_signal = sum(_axis_verdicts.values())
print(f"\n{'='*70}\n요약: 3축 중 {n_signal}개 축에서 신호 확인 "
      f"({', '.join(a for a, s in _axis_verdicts.items() if s) or '없음'})")
if n_signal == 0:
    print("→ w1/w2/w3 모두 250K 스케일에서 유의한 차이를 만들지 않습니다. "
          "Round 2 를 조기 종료하고 baseline 그대로 다음 단계로 이동을 권장합니다.")
elif n_signal < 3:
    print(f"→ 신호 있는 축만 원 해상도 유지, 나머지는 baseline 고정한 축소 grid 로 "
          f"Phase 2 를 구성하는 것을 권장합니다.")
else:
    print("→ 3축 모두 신호가 있어 원 36-grid 를 유지하되, 각 점 최소 2-seed 로 "
          "Phase 2 를 구성하는 것을 권장합니다.")
print(f"{'='*70}")
```

---

## 13. Round 2 Phase 2 — w2×w3 본 grid (9 조합 × 2-seed = 18회)

> PROGRESS.md 의사결정 32: Phase 1 결과(w1 range=0.0003 → 제외, w2/w3 모두 grid 경계값이
> 우세)에 따라 §16.3.2 원 grid 를 승자 방향으로 이동했다 — w1=0.1 고정,
> w2 ∈ {0.5, 1.0, 2.0}(원 {1.0,2.0,5.0}에서 패자 5.0 대신 경계 밖 0.5 추가),
> w3 ∈ {50, 100, 200}(원 {20,50,100}에서 패자 20 대신 경계 밖 200 추가). alpha/beta 는
> var008 고정값(2.0/0.3). 근거는 docs/autoresearch-ops.md §16.5.B "Stage 2: 살아남은
> 후보 ± 50% local grid" 원칙 — 자세한 근거는 PROGRESS.md 의사결정 32 참조.
>
> `cache/round1/`, `cache/round2_phase1_prescreen/` 은 이 셀에서 캐시 재사용 후보를
> **읽기만** 하고 절대 쓰지 않는다 — 결과는 `cache/round2_phase2/` 전용 디렉터리에
> 저장한다. wandb run 이름은 `round_n=2` 를 명시해 `step1_round2_stage1_var*` 로
> 찍히도록 한다(§12 는 이미 실행된 뒤라 그대로 두고, 이 셀부터 바로잡음).

```python
"""
Round 2 Phase 2 — w2×w3 9 조합 × 2-seed = 18 variants × 250K.

캐시 재사용 규칙 (사용자 확인 사항 1): alpha/beta/w1/w2/w3 다섯 값이 모두 완전히
일치할 때만 재사용한다. 하나라도 다르면 해당 슬롯은 무조건 새로 학습한다 — "비슷하니
재사용"은 하지 않는다.
  - (w2=2.0, w3=50) 의 seed 슬롯 0 ← cache/round1/stage1_var008.json (Round 1 var008)
  - (w2=1.0, w3=50) 의 seed 슬롯 0 ← cache/round2_phase1_prescreen/stage1_var002.json
둘 다 seed 슬롯 1 은 새로 학습한다 — "재사용 1개 + 신규 1개로 2-seed 채우기"가
"2개 다 신규"와 통계적으로 다르지 않다는 판단 근거는 PROGRESS.md 의사결정 32 참조
(PR #9 로 기본값 경로가 수정 전후 완전히 동일함이 코드 레벨로 이미 검증됐으므로,
재사용 슬롯이 신규 슬롯과 다른 분포에서 나왔다고 볼 근거가 없다).

판정(사용자 확인 사항 3): 각 9-cell 의 2-seed 평균으로 순위를 매기되, 인접한
두 cell 의 평균 차이가 "이 실험에서 실측된 seed 편차"보다 작으면 같은 그룹(구분 불가)
으로 묶는다. 노이즈 임계값은 max(이번 9-cell 의 |seed0-seed1| 평균, 0.0135) 로
정한다 — 기존 0.0135(§10) 보다 이번 실측 편차가 크게 나오면(w3=200 처럼 미검증 영역
확장 시 불안정할 수 있음) 더 보수적인 쪽을 쓴다.
"""
import json
from pathlib import Path
import torch

if "PROJECT" not in globals():
    raise RuntimeError("PROJECT 가 정의되지 않았습니다 — 먼저 셀 1(환경 설정)을 실행하세요.")
ROUND1_CACHE_DIR = f"{PROJECT}/cache/round1"
PHASE1_CACHE_DIR = f"{PROJECT}/cache/round2_phase1_prescreen"
PHASE2_CACHE_DIR = f"{PROJECT}/cache/round2_phase2"   # round1/, round2_phase1_prescreen/ 과 절대 안 겹침
Path(PHASE2_CACHE_DIR).mkdir(parents=True, exist_ok=True)

# GPU 확인 (하드 스톱) — 셀 5/10/11/12 와 동일
REQUIRE_GPU = True
if torch.cuda.is_available():
    print(f"✅ GPU 사용 중: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU 를 사용할 수 없습니다 — CPU 로 진행하면 수 배 느려집니다.")
    if REQUIRE_GPU:
        raise RuntimeError(
            "GPU 미사용 — REQUIRE_GPU=True(기본값) 라 중단합니다. "
            "GPU 없이 계속하려면 이 셀 상단의 REQUIRE_GPU = False 로 바꾸세요."
        )

VAR008_ALPHA, VAR008_BETA = 2.0, 0.3
BASELINE_W1 = 0.1
THRESHOLD_STANDING = 0.0135   # PROGRESS.md 의사결정 28 — 계속 쓰는 표준 임계값

_W2_VALUES = (0.5, 1.0, 2.0)
_W3_VALUES = (50.0, 100.0, 200.0)
_GRID = [(w2, w3) for w2 in _W2_VALUES for w3 in _W3_VALUES]   # 9 cells, index = cell_idx
assert len(_GRID) == 9

def _cell_params(w2, w3):
    return {"alpha": VAR008_ALPHA, "beta": VAR008_BETA, "w1": BASELINE_W1, "w2": w2, "w3": w3}

# variant_id = cell_idx*2 + seed_slot (0 또는 1)
variants = []
for w2, w3 in _GRID:
    variants.append(_cell_params(w2, w3))   # seed slot 0
    variants.append(_cell_params(w2, w3))   # seed slot 1
assert len(variants) == 18

# ── 캐시 재사용 후보 검증 — 5개 값이 모두 일치할 때만 재사용 ────────────────
def _params_match(existing: dict, target: dict, tol: float = 1e-9) -> bool:
    defaults = {"alpha": 1.0, "beta": 0.1, "w1": 0.1, "w2": 2.0, "w3": 50.0}
    for key in ("alpha", "beta", "w1", "w2", "w3"):
        if abs(existing.get(key, defaults[key]) - target[key]) > tol:
            return False
    return True

_REUSE_CANDIDATES = [
    # (target cell (w2, w3), source 경로, source 라벨)
    ((2.0, 50.0), Path(ROUND1_CACHE_DIR) / "stage1_var008.json", "Round 1 var008"),
    ((1.0, 50.0), Path(PHASE1_CACHE_DIR) / "stage1_var002.json", "Phase 1 var002"),
]
print("=== 캐시 재사용 후보 검증 (5개 값 alpha/beta/w1/w2/w3 완전 일치 시에만 재사용) ===")
for (w2, w3), src_path, label in _REUSE_CANDIDATES:
    cell_idx = _GRID.index((w2, w3))
    slot0_id = cell_idx * 2
    dst_path = Path(PHASE2_CACHE_DIR) / f"stage1_var{slot0_id:03d}.json"
    target = _cell_params(w2, w3)
    if dst_path.exists():
        print(f"  (w2={w2}, w3={w3}) slot0(var{slot0_id:03d}): 이미 재사용/학습된 캐시 있음 — 건너뜀")
        continue
    if not src_path.exists():
        print(f"  ⚠️ (w2={w2}, w3={w3}): {src_path} 없음 — 재사용 불가, 슬롯 0 도 새로 학습합니다")
        continue
    src_data = json.loads(src_path.read_text())
    if _params_match(src_data.get("params", {}), target):
        copied = dict(src_data)
        copied["variant_id"] = slot0_id   # 이 리스트 안에서의 위치로 재매핑
        dst_path.write_text(json.dumps(copied), encoding="utf-8")
        print(f"  ✅ (w2={w2}, w3={w3}) slot0(var{slot0_id:03d}) ← {label} 재사용 "
              f"(score={src_data['metric']:.4f}, params 일치 확인됨)")
    else:
        print(f"  ❌ (w2={w2}, w3={w3}): {label} 의 params({src_data.get('params')}) 가 "
              f"목표({target}) 와 다릅니다 — 재사용하지 않고 새로 학습합니다")

_cache_hit_n = sum(
    1 for i in range(len(variants)) if (Path(PHASE2_CACHE_DIR) / f"stage1_var{i:03d}.json").exists()
)
print(f"\n{len(variants)}개 중 {_cache_hit_n}개 캐시 재사용, {len(variants) - _cache_hit_n}개 신규 학습 "
      f"(예상 추가 소요: 최대 {(len(variants) - _cache_hit_n) * 50}분)")

from autoresearch.stage_runner import StageRunner

runner = StageRunner(
    train_fn=make_train_fn_real(n_envs=1, cache_dir=PHASE2_CACHE_DIR, round_n=2),   # wandb: step1_round2_*
    max_workers=1,
    stage1_timesteps=250_000,
    stage1_keep_ratio=1.0,       # 스크리닝이 아니라 grid 자체를 측정하므로 탈락 없음
    cache_dir=PHASE2_CACHE_DIR,
)

print(f"\n=== Phase 2 시작: {len(variants)} variants(9 cells × 2-seed) × 250K ===")
results = runner.run_stage1(variants)
scores = {r.variant_id: r.metric for r in results}

# ── cell 별 2-seed 평균/range 집계 ───────────────────────────────────────────
import statistics

cell_stats = []
for cell_idx, (w2, w3) in enumerate(_GRID):
    s0, s1 = scores[cell_idx * 2], scores[cell_idx * 2 + 1]
    cell_stats.append({
        "w2": w2, "w3": w3, "scores": (s0, s1),
        "mean": statistics.mean((s0, s1)), "range": abs(s1 - s0),
    })

mean_pairwise_range = statistics.mean(c["range"] for c in cell_stats)
NOISE_THRESHOLD = max(mean_pairwise_range, THRESHOLD_STANDING)   # 더 보수적인 쪽 채택

print(f"\n{'='*70}\nRound 2 Phase 2 — cell 별 2-seed 결과 (평균 내림차순)\n{'='*70}")
print(f"이번 실험 실측 seed 편차(9-cell |Δ| 평균) = {mean_pairwise_range:.4f}  "
      f"vs 표준 임계값(§10) = {THRESHOLD_STANDING:.4f}  → 채택: {NOISE_THRESHOLD:.4f}\n")

cell_stats_sorted = sorted(cell_stats, key=lambda c: c["mean"], reverse=True)
for rank, c in enumerate(cell_stats_sorted, start=1):
    s0, s1 = c["scores"]
    print(f"  {rank}. w2={c['w2']}, w3={c['w3']}  mean={c['mean']:.4f}  "
          f"(seeds: {s0:.4f}, {s1:.4f}, range={c['range']:.4f})")

# ── 순위 신뢰도 — 인접 cell 간 차이가 노이즈 이내면 같은 그룹으로 묶는다 ──────
print(f"\n{'='*70}\n순위 신뢰도 판정 (임계값 {NOISE_THRESHOLD:.4f})\n{'='*70}")
groups = [[cell_stats_sorted[0]]]
for c in cell_stats_sorted[1:]:
    if abs(c["mean"] - groups[-1][-1]["mean"]) < NOISE_THRESHOLD:
        groups[-1].append(c)
    else:
        groups.append([c])

for gi, group in enumerate(groups, start=1):
    if len(group) == 1:
        c = group[0]
        print(f"  단독 {gi}그룹: w2={c['w2']}, w3={c['w3']} (mean={c['mean']:.4f})")
    else:
        members = ", ".join(f"(w2={c['w2']}, w3={c['w3']})" for c in group)
        print(f"  구분 불가 {gi}그룹({len(group)}개, 서로 Δ<{NOISE_THRESHOLD:.4f}): {members}")

top1, top2 = cell_stats_sorted[0], cell_stats_sorted[1]
diff_top = abs(top1["mean"] - top2["mean"])
print(f"\n1위 vs 2위 Δ={diff_top:.4f} (임계값 {NOISE_THRESHOLD:.4f})")
if diff_top < NOISE_THRESHOLD:
    print("→ 1위와 2위 구분 불가. 위 '구분 불가 그룹' 안에서 최종 선택은 다른 기준"
          "(예: 재현성, 계산 비용, 안정성)으로 결정하거나 seed 를 늘려 재확인하세요.")
else:
    print(f"→ 1위 확정: w2={top1['w2']}, w3={top1['w3']} (mean={top1['mean']:.4f})")
print(f"{'='*70}")
```

---

## 타임라인 예상 (무료 Colab T4, worker 1 — 순차 실행)

| 단계 | 예상 시간 | 세션 내 완료 여부 |
|------|---------|----------------|
| dry-run (12 × 5K) | ~3~5분 | ✅ |
| Stage 1 (12 × 250K, worker 1) | ~3~4시간 | ✅ (12시간 내) |
| Stage 2 (6 × 2M, worker 1) | ~12~14시간 | ⚠️ 세션 1개로 안 끝남 — 캐시로 이어서 진행 |
| **Round 1 합계** | **~15~18시간** | ⚠️ 최소 2세션 필요. 무료 tier 사용량 한도는 고정 12시간이 아니라 최근 사용량에 따라 변동하므로 3세션 이상 걸릴 수 있음 |
| Round 2 Phase 1 (6 × 250K, worker 1) | ~4.8~5.2시간 (중앙값 ~5.0시간, ~7.1~7.7 units) | ✅ (12시간 내), variant 단위 캐시로 재개 가능 |
| Round 2 Phase 2 (9-cell × 2-seed = 18 슬롯, 캐시 재사용 2 → 신규 16 × 250K) | ~12.8~13.9시간 (중앙값 ~13.4시간, ~18.9~20.6 units) | ⚠️ 12시간 근접/초과 가능 — 세션 2개 대비, variant 단위 캐시로 재개 가능 |

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
