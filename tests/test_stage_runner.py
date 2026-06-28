"""StageRunner 단위 테스트 + sanity check — autoresearch/stage_runner.py.

검증 범위:
  - Stage 1: N variants → 상위 keep_ratio 생존 (정확히 50%)
  - Stage 1: 최소 1개 보장
  - Stage 1: metric 내림차순 정렬
  - Stage 2: best 1 반환
  - 전체 파이프라인 (Stage 1 → Stage 2)
  - max_workers 상한 경고 (§16.5.B)
  - 실패 variant: metric=-inf 로 마킹, Stage 1 에서 자동 제거

sanity check (pytest -m sanity):
  4 variants × 5K timestep — 실제 train_step1.run_training 호출.
  § 5.2.a 검증 전용. 정상 학습과 다른 파라미터.
"""

from __future__ import annotations

import logging
import math

import pytest

from autoresearch.stage_runner import (
    MAX_PARALLEL_WORKERS_SPEED,
    STAGE1_KEEP_RATIO,
    STAGE1_TIMESTEPS,
    STAGE2_TIMESTEPS,
    StageRunner,
    VariantResult,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def make_deterministic_train_fn(scores: list[float]):
    """variant_id 순서대로 고정 metric 반환하는 mock train_fn."""
    def train_fn(params: dict, timesteps: int, variant_id: int) -> float:
        return scores[variant_id]
    return train_fn


def make_runner(scores: list[float], **kwargs) -> StageRunner:
    return StageRunner(
        train_fn=make_deterministic_train_fn(scores),
        stage1_timesteps=100,    # 테스트 전용 (빠른 실행)
        stage2_timesteps=200,
        **kwargs,
    )


# ─── 1. Stage 1: 제거 로직 ───────────────────────────────────────────────────

def test_stage1_eliminates_bottom_50_percent():
    """4 variants → 2 생존 (정확히 50% 제거)."""
    runner = make_runner(scores=[0.1, 0.9, 0.5, 0.3])
    variants = [{"v": i} for i in range(4)]
    survivors = runner.run_stage1(variants)
    assert len(survivors) == 2


def test_stage1_survivors_are_top_by_metric():
    """생존자가 metric 기준 상위 k개."""
    runner = make_runner(scores=[0.1, 0.9, 0.5, 0.3])
    variants = [{"v": i} for i in range(4)]
    survivors = runner.run_stage1(variants)
    survivor_metrics = [r.metric for r in survivors]
    assert max(survivor_metrics) == pytest.approx(0.9)
    assert min(survivor_metrics) == pytest.approx(0.5)


def test_stage1_sorted_descending():
    """생존 결과가 metric 내림차순 정렬."""
    runner = make_runner(scores=[0.2, 0.8, 0.6, 0.4])
    variants = [{"v": i} for i in range(4)]
    survivors = runner.run_stage1(variants)
    metrics = [r.metric for r in survivors]
    assert metrics == sorted(metrics, reverse=True)


def test_stage1_keeps_at_least_one():
    """1개 입력 → 1개 생존 (최소 보장)."""
    runner = make_runner(scores=[0.42])
    survivors = runner.run_stage1([{"v": 0}])
    assert len(survivors) == 1
    assert survivors[0].metric == pytest.approx(0.42)


def test_stage1_3_variants_keeps_2():
    """3개 입력 → ceil(3×0.5)=2 생존."""
    runner = make_runner(scores=[0.3, 0.7, 0.5])
    survivors = runner.run_stage1([{"v": i} for i in range(3)])
    assert len(survivors) == 2


def test_stage1_variant_result_has_correct_stage():
    """VariantResult.stage == 1."""
    runner = make_runner(scores=[0.5])
    survivors = runner.run_stage1([{"v": 0}])
    assert survivors[0].stage == 1


def test_stage1_empty_variants_raises():
    """빈 variants 는 ValueError."""
    runner = make_runner(scores=[])
    with pytest.raises(ValueError, match="비어"):
        runner.run_stage1([])


# ─── 2. Stage 2: best 선택 ───────────────────────────────────────────────────

def test_stage2_returns_best_variant():
    """Stage 2: 2 survivors → best metric 반환."""
    runner = make_runner(scores=[0.4, 0.8])
    survivors = [
        VariantResult(variant_id=0, params={"v": 0}, metric=0.4, timesteps=100, stage=1),
        VariantResult(variant_id=1, params={"v": 1}, metric=0.8, timesteps=100, stage=1),
    ]
    best = runner.run_stage2(survivors)
    assert best.metric == pytest.approx(0.8)
    assert best.variant_id == 1


def test_stage2_variant_result_stage_is_2():
    """Stage 2 결과의 stage == 2."""
    runner = make_runner(scores=[0.5])
    survivors = [VariantResult(variant_id=0, params={}, metric=0.5, timesteps=100, stage=1)]
    best = runner.run_stage2(survivors)
    assert best.stage == 2


def test_stage2_empty_survivors_raises():
    """빈 survivors 는 ValueError."""
    runner = make_runner(scores=[])
    with pytest.raises(ValueError, match="비어"):
        runner.run_stage2([])


# ─── 3. 전체 파이프라인 ──────────────────────────────────────────────────────

def test_run_full_returns_best_and_survivors():
    """run_full: (best, survivors) 반환. best 가 가장 높은 metric."""
    scores = [0.1, 0.9, 0.5, 0.3, 0.7, 0.2]
    runner = make_runner(scores=scores)
    variants = [{"v": i} for i in range(len(scores))]
    best, survivors = runner.run_full(variants)
    assert isinstance(best, VariantResult)
    assert best.metric == max(s.metric for s in survivors)


def test_run_full_pipeline_6_variants():
    """6 variants: Stage 1 → 3 생존 → Stage 2 → best 1."""
    scores = [0.1, 0.9, 0.5, 0.3, 0.7, 0.2]
    runner = make_runner(scores=scores)
    variants = [{"v": i} for i in range(6)]
    best, survivors = runner.run_full(variants)
    assert len(survivors) == math.ceil(6 * STAGE1_KEEP_RATIO)  # 3
    assert best.metric >= max(s.metric for s in survivors) - 1e-9


# ─── 4. 내결함성 캐시 (Colab 세션 중단 대비) ────────────────────────────────

def test_cache_hit_skips_train_fn(tmp_path):
    """캐시 파일이 있으면 train_fn 호출 없이 캐시 반환."""
    call_count = [0]

    def train_fn(params, timesteps, variant_id):
        call_count[0] += 1
        return 0.7

    runner = StageRunner(
        train_fn=train_fn,
        stage1_timesteps=10,
        stage2_timesteps=20,
        cache_dir=tmp_path,
    )
    variants = [{"v": 0}]

    # 첫 실행 → train_fn 호출 + 캐시 저장
    runner.run_stage1(variants)
    assert call_count[0] == 1

    # 두 번째 실행 → 캐시 히트, train_fn 호출 없음
    runner.run_stage1(variants)
    assert call_count[0] == 1   # 여전히 1


def test_cache_file_written_on_completion(tmp_path):
    """variant 완료 시 캐시 JSON 파일 생성됨."""
    runner = StageRunner(
        train_fn=lambda p, t, v: 0.5,
        stage1_timesteps=10,
        stage2_timesteps=20,
        cache_dir=tmp_path,
    )
    runner.run_stage1([{"v": 0}, {"v": 1}])
    assert (tmp_path / "stage1_var000.json").exists()
    assert (tmp_path / "stage1_var001.json").exists()


def test_cache_miss_on_fresh_dir(tmp_path):
    """캐시 파일 없으면 train_fn 정상 호출."""
    call_count = [0]

    def train_fn(params, timesteps, variant_id):
        call_count[0] += 1
        return 0.6

    runner = StageRunner(
        train_fn=train_fn,
        stage1_timesteps=10,
        stage2_timesteps=20,
        cache_dir=tmp_path / "new_dir",  # 새 디렉터리
    )
    runner.run_stage1([{"v": 0}])
    assert call_count[0] == 1


def test_cache_none_means_no_caching():
    """cache_dir=None 이면 캐시 없이 정상 동작."""
    call_count = [0]

    def train_fn(params, timesteps, variant_id):
        call_count[0] += 1
        return 0.5

    runner = StageRunner(
        train_fn=train_fn,
        stage1_timesteps=10,
        stage2_timesteps=20,
        cache_dir=None,
    )
    runner.run_stage1([{"v": 0}])
    runner.run_stage1([{"v": 0}])   # 캐시 없으므로 2번 호출
    assert call_count[0] == 2


def test_cache_stage2_separate_from_stage1(tmp_path):
    """Stage 1 과 Stage 2 캐시 파일이 별도로 저장됨."""
    runner = StageRunner(
        train_fn=lambda p, t, v: 0.8,
        stage1_timesteps=10,
        stage2_timesteps=20,
        cache_dir=tmp_path,
    )
    variants = [{"v": 0}, {"v": 1}]
    survivors = runner.run_stage1(variants)
    runner.run_stage2(survivors)

    # stage1_var000.json 과 stage2_var000.json 모두 존재
    assert (tmp_path / "stage1_var000.json").exists()
    assert (tmp_path / "stage2_var000.json").exists()


# ─── 5. max_workers 경고 ─────────────────────────────────────────────────────

def test_max_workers_over_limit_logs_warning(caplog):
    """max_workers > 3 시 WARNING 로그 발생. §16.5.B."""
    with caplog.at_level(logging.WARNING, logger="autoresearch.stage_runner"):
        runner = StageRunner(
            train_fn=lambda p, t, v: 0.5,
            max_workers=MAX_PARALLEL_WORKERS_SPEED + 1,
            stage1_timesteps=10,
            stage2_timesteps=20,
        )
    assert "OOM" in caplog.text or "spec 상한" in caplog.text


def test_max_workers_within_limit_no_warning(caplog):
    """max_workers ≤ 3 시 경고 없음."""
    with caplog.at_level(logging.WARNING, logger="autoresearch.stage_runner"):
        StageRunner(
            train_fn=lambda p, t, v: 0.5,
            max_workers=MAX_PARALLEL_WORKERS_SPEED,
            stage1_timesteps=10,
            stage2_timesteps=20,
        )
    assert "OOM" not in caplog.text


# ─── 6. 실패 variant 처리 ────────────────────────────────────────────────────

def test_failed_variant_gets_neg_inf_metric():
    """train_fn 예외 → metric=-inf, Stage 1 에서 제거된다."""
    call_count = [0]

    def train_fn_with_failure(params, timesteps, variant_id):
        call_count[0] += 1
        if params.get("fail"):
            raise RuntimeError("intentional failure")
        return 0.8

    runner = StageRunner(
        train_fn=train_fn_with_failure,
        max_workers=2,  # parallel 로 실행해야 exception 처리 경로 통과
        stage1_timesteps=10,
        stage2_timesteps=20,
    )
    variants = [{"fail": True}, {"fail": False}, {"fail": True}, {"fail": False}]
    survivors = runner.run_stage1(variants)
    # 실패 variant (metric=-inf) 는 제거됨
    assert all(s.metric > float("-inf") for s in survivors)


# ─── 7. 상수 값 spec 정합 ────────────────────────────────────────────────────

def test_stage1_timesteps_constant():
    """STAGE1_TIMESTEPS == 250,000. §16.5.B."""
    assert STAGE1_TIMESTEPS == 250_000


def test_stage2_timesteps_constant():
    """STAGE2_TIMESTEPS == 2,000,000. §16.5.B."""
    assert STAGE2_TIMESTEPS == 2_000_000


def test_stage1_keep_ratio_constant():
    """STAGE1_KEEP_RATIO == 0.5. §16.5.B."""
    assert STAGE1_KEEP_RATIO == pytest.approx(0.5)


# ─── 7. sanity check (실제 train_fn, 5K timestep) ────────────────────────────

@pytest.mark.sanity
def test_stage_runner_sanity_4_variants_5k(tmp_path):
    """4 variants × 5K timestep — stage_runner 실제 작동 확인.

    Sub-단계 5.2.a 통합 검증 전용. 느린 테스트이므로 -m sanity 로만 실행.
    """
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from training.train_step1 import parse_args, run_training

    # Round 1 α×β 의 일부 (4 variants 만)
    test_variants = [
        {"alpha": 0.5, "beta": 0.0},
        {"alpha": 1.0, "beta": 0.1},
        {"alpha": 2.0, "beta": 0.1},
        {"alpha": 5.0, "beta": 0.3},
    ]

    def train_fn(params: dict, timesteps: int, variant_id: int) -> float:
        sys.argv = [
            "train_step1.py",
            "--timesteps", str(timesteps),
            "--n-envs", "1",
            "--no-wandb",
            "--offline",
            "--handoff-dir", str(tmp_path),
            "--difficulty", "easy",
            "--seed", str(150000 + variant_id),
            "--alpha", str(params["alpha"]),
            "--beta", str(params["beta"]),
        ]
        result = run_training(parse_args())
        return result["success_rate"]

    runner = StageRunner(
        train_fn=train_fn,
        max_workers=1,
        stage1_timesteps=5_000,   # sanity: 매우 짧음
        stage2_timesteps=5_000,
        stage1_keep_ratio=0.5,
    )

    best, survivors = runner.run_full(test_variants)

    # 기본 정합 검증
    assert isinstance(best, VariantResult)
    assert 0.0 <= best.metric <= 1.0
    assert len(survivors) == 2  # 4 → 2
    assert best.metric >= min(s.metric for s in survivors) - 1e-9
