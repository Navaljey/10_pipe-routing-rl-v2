"""2-Stage screening — CLAUDE.md §16.5.B.

Stage 1 (1차 스크리닝):
  250K timestep × N variants → 하위 50% 제거 (successive halving)
  동시 학습 가능 모델 수: T4 GPU 1개 기준 2~3 (안전 2, 속도 우선 3)

Stage 2 (정상 학습):
  2M timestep × 살아남은 후보 (N/2) → best 1 선택

Stage 1/2 임계값 (§11.0.7):
  success_rate 절대 손실 — Stage 1: −10%p 이내 / Stage 2: −5%p 이내
  length_ratio 증가 — Stage 1: +20% 이내 / Stage 2: +10% 이내

평가 metric (direction: maximize):
  기본 = success_rate (Stage 1/2 공통)

train_fn 인터페이스:
  def train_fn(params: dict, timesteps: int, variant_id: int) -> float:
      # params: 모든 hyperparameter dict
      # timesteps: Stage 1=250K / Stage 2=2M
      # variant_id: logging 용 식별자
      # return: success_rate (0.0~1.0)
"""

from __future__ import annotations

import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, NamedTuple

logger = logging.getLogger(__name__)

# ─── CLAUDE.md §16.5.B 고정값 ───────────────────────────────────────────────
STAGE1_TIMESTEPS: int = 250_000
STAGE2_TIMESTEPS: int = 2_000_000
STAGE1_KEEP_RATIO: float = 0.5        # 하위 50% 제거 (successive halving 표준)
MAX_PARALLEL_WORKERS_SAFE: int = 2     # T4 GPU 1개 기준 안전 (무료 Colab 기준)
MAX_PARALLEL_WORKERS_SPEED: int = 3   # T4 GPU 1개 기준 속도 우선


class VariantResult(NamedTuple):
    """단일 variant 학습 결과."""
    variant_id: int
    params: dict
    metric: float       # success_rate (높을수록 좋음)
    timesteps: int
    stage: int          # 1 or 2


class StageRunner:
    """CLAUDE.md §16.5.B 2-Stage screening 오케스트레이터.

    Parameters
    ----------
    train_fn :
        (params, timesteps, variant_id) → float (metric, 높을수록 좋음)
    max_workers :
        동시 학습 worker 수. 1=sequential.
        무료 Colab T4 기준 2(안전). §16.5.B.
    stage1_timesteps, stage2_timesteps :
        스펙 고정값. 테스트 시 오버라이드 가능 (sanity check 용).
    stage1_keep_ratio :
        Stage 1 생존 비율. 0.5 = 하위 50% 제거. §16.5.B.
    cache_dir :
        내결함성 캐시 디렉터리. None=캐시 없음.
        세션 중단 시 완료된 variant 결과를 JSON으로 보존 → 재시작 시 재사용.
        파일명: {cache_dir}/stage{S}_var{V:03d}.json
    on_stage1_complete :
        Stage 1 완료 콜백. signature: (all_results, survivors) → None.
        None=noop. RoundOrchestrator 가 wandb summary logging 에 사용.
    on_stage2_complete :
        Stage 2 완료 콜백. signature: (all_results, best) → None.
        None=noop.
    """

    def __init__(
        self,
        train_fn: Callable[[dict, int, int], float],
        max_workers: int = 1,
        stage1_timesteps: int = STAGE1_TIMESTEPS,
        stage2_timesteps: int = STAGE2_TIMESTEPS,
        stage1_keep_ratio: float = STAGE1_KEEP_RATIO,
        cache_dir: str | Path | None = None,
        on_stage1_complete: Callable[[list[Any], list[Any]], None] | None = None,
        on_stage2_complete: Callable[[list[Any], Any], None] | None = None,
    ) -> None:
        if max_workers > MAX_PARALLEL_WORKERS_SPEED:
            logger.warning(
                "[stage_runner] max_workers=%d > spec 상한 %d. "
                "GPU OOM 위험. §16.5.B 참조.",
                max_workers, MAX_PARALLEL_WORKERS_SPEED,
            )
        self.train_fn = train_fn
        self.max_workers = max_workers
        self.stage1_timesteps = stage1_timesteps
        self.stage2_timesteps = stage2_timesteps
        self.stage1_keep_ratio = stage1_keep_ratio
        self.cache_dir: Path | None = Path(cache_dir) if cache_dir is not None else None
        self.on_stage1_complete = on_stage1_complete
        self.on_stage2_complete = on_stage2_complete
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("[stage_runner] 내결함성 캐시 활성화: %s", self.cache_dir)

    # ─── public API ─────────────────────────────────────────────────────────

    def run_stage1(self, variants: list[dict]) -> list[VariantResult]:
        """Stage 1: N variants × stage1_timesteps → 상위 keep_ratio 반환 (정렬됨).

        Returns
        -------
        survivors : VariantResult list, metric 내림차순 정렬.
                    최소 1개 보장.
        """
        if not variants:
            raise ValueError("variants 가 비어 있습니다.")

        logger.info(
            "[stage_runner] Stage 1 시작: %d variants × %d timestep, workers=%d",
            len(variants), self.stage1_timesteps, self.max_workers,
        )

        results = self._run_parallel(variants, self.stage1_timesteps, stage=1)
        results_sorted = sorted(results, key=lambda r: r.metric, reverse=True)

        keep_n = max(1, math.ceil(len(results_sorted) * self.stage1_keep_ratio))
        survivors = results_sorted[:keep_n]

        eliminated = results_sorted[keep_n:]
        logger.info(
            "[stage_runner] Stage 1 완료: %d → %d 생존 (제거 %d)",
            len(results_sorted), len(survivors), len(eliminated),
        )
        for r in eliminated:
            logger.debug("  [제거] var%03d params=%s metric=%.4f", r.variant_id, r.params, r.metric)
        for r in survivors:
            logger.debug("  [생존] var%03d params=%s metric=%.4f", r.variant_id, r.params, r.metric)

        if self.on_stage1_complete is not None:
            try:
                self.on_stage1_complete(results_sorted, survivors)
            except Exception as e:
                logger.warning("[stage_runner] on_stage1_complete 콜백 실패: %s", e)

        # Stage 1 survivors 자동 저장 — Stage 2 재개용 (취약점 보강)
        self._save_survivors(survivors)

        return survivors

    def run_stage2(self, survivors: list[VariantResult]) -> VariantResult:
        """Stage 2: 생존한 후보 × stage2_timesteps → best 1 반환.

        Parameters
        ----------
        survivors : Stage 1 결과 (VariantResult list).

        Returns
        -------
        best : 가장 높은 metric 의 VariantResult.
        """
        if not survivors:
            raise ValueError("survivors 가 비어 있습니다.")

        logger.info(
            "[stage_runner] Stage 2 시작: %d 후보 × %d timestep, workers=%d",
            len(survivors), self.stage2_timesteps, self.max_workers,
        )

        variants = [r.params for r in survivors]
        variant_ids = [r.variant_id for r in survivors]
        results = self._run_parallel(variants, self.stage2_timesteps, stage=2,
                                     variant_ids=variant_ids)
        results_sorted = sorted(results, key=lambda r: r.metric, reverse=True)
        best = results_sorted[0]

        logger.info(
            "[stage_runner] Stage 2 완료: best var%03d metric=%.4f params=%s",
            best.variant_id, best.metric, best.params,
        )

        if self.on_stage2_complete is not None:
            try:
                self.on_stage2_complete(results_sorted, best)
            except Exception as e:
                logger.warning("[stage_runner] on_stage2_complete 콜백 실패: %s", e)

        return best

    def run_full(self, variants: list[dict]) -> tuple[VariantResult, list[VariantResult]]:
        """Stage 1 + Stage 2 전체 실행. Stage 1 완료 캐시가 있으면 Stage 1 생략.

        재개 흐름 (세션 중단 후):
          - cache_dir/stage1_survivors.json 존재 → Stage 1 스킵, Stage 2 바로 실행
          - stage1_survivors.json 없음 → 정상 순서 (Stage 1 → Stage 2)
          - Stage 2 개별 variant 캐시 (stage2_var{V}.json) 도 자동 재사용

        Returns
        -------
        (best, survivors)
        """
        survivors = self._load_survivors()
        if survivors is not None:
            logger.info(
                "[stage_runner] Stage 1 survivors 캐시 히트 (%d개) — Stage 1 생략, Stage 2 바로 실행",
                len(survivors),
            )
        else:
            survivors = self.run_stage1(variants)

        best = self.run_stage2(survivors)
        return best, survivors

    # ─── internal ───────────────────────────────────────────────────────────

    def _run_parallel(
        self,
        variants: list[dict],
        timesteps: int,
        stage: int,
        variant_ids: list[int] | None = None,
    ) -> list[VariantResult]:
        """variants 를 max_workers 동시 실행. 1이면 sequential."""
        if variant_ids is None:
            variant_ids = list(range(len(variants)))

        tasks = list(zip(variant_ids, variants))

        if self.max_workers <= 1:
            return [self._run_one_safe(vid, params, timesteps, stage) for vid, params in tasks]

        results: list[VariantResult] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._run_one_safe, vid, params, timesteps, stage): vid
                for vid, params in tasks
            }
            for future in as_completed(futures):
                vid = futures[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    # 방어적 fallback: _run_one_safe 자체가 실패하는 극단적 경우
                    # (예: pickling 오류 등 ThreadPoolExecutor 자체 문제)
                    logger.error(
                        "[stage_runner] var%03d 예상치 못한 실패 (safe wrapper 밖): %s",
                        vid, e, exc_info=True,
                    )
                    results.append(VariantResult(
                        variant_id=vid,
                        params=tasks[variant_ids.index(vid)][1],
                        metric=float("-inf"),
                        timesteps=timesteps,
                        stage=stage,
                    ))
        return results

    def _run_one_safe(
        self, variant_id: int, params: dict, timesteps: int, stage: int
    ) -> VariantResult:
        """_run_one() 을 예외로부터 보호하는 공통 wrapper — sequential/parallel 동일 경로.

        train_fn 예외 시 전체 traceback 을 로그에 남기고 metric=-inf 로 마킹한다
        (Stage 1 에서 자동 제거됨). 이전에는 parallel(max_workers>1) 경로에서만
        이렇게 흡수하고 sequential 경로는 예외를 그대로 전파해 동작이 갈렸다.

        의도적으로 실패 결과는 캐시에 저장하지 않는다 (_run_one() 내부의
        _save_cached() 는 정상 반환 시에만 거친다). 원인을 고치고 재실행하면
        다시 시도되어야 하므로 — 캐싱하면 버그를 고쳐도 오래된 -inf 가
        재사용되어 수정이 반영 안 된 것처럼 보일 수 있다.
        """
        try:
            return self._run_one(variant_id, params, timesteps, stage)
        except Exception:
            logger.error(
                "[stage_runner] var%03d 학습 실패 (stage=%d) — metric=-inf 로 마킹",
                variant_id, stage, exc_info=True,
            )
            return VariantResult(
                variant_id=variant_id,
                params=params,
                metric=float("-inf"),
                timesteps=timesteps,
                stage=stage,
            )

    def _cache_path(self, stage: int, variant_id: int) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"stage{stage}_var{variant_id:03d}.json"

    def _survivors_cache_path(self) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / "stage1_survivors.json"

    def _save_survivors(self, survivors: list[VariantResult]) -> None:
        """Stage 1 survivors 를 캐시에 자동 저장 (Stage 2 재개용)."""
        path = self._survivors_cache_path()
        if path is None:
            return
        try:
            path.write_text(json.dumps([r._asdict() for r in survivors]), encoding="utf-8")
            logger.info("[stage_runner] Stage 1 survivors 자동 저장: %s (%d개)", path, len(survivors))
        except Exception as e:
            logger.warning("[stage_runner] survivors 저장 실패: %s", e)

    def _load_survivors(self) -> list[VariantResult] | None:
        """Stage 1 survivors 캐시 로드. 없거나 손상되면 None."""
        path = self._survivors_cache_path()
        if path is None or not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            survivors = [VariantResult(**d) for d in data]
            return survivors
        except Exception as e:
            logger.warning("[stage_runner] survivors 캐시 로드 실패 (%s) — Stage 1 재실행", e)
            return None

    def _load_cached(self, stage: int, variant_id: int) -> VariantResult | None:
        path = self._cache_path(stage, variant_id)
        if path is None or not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            result = VariantResult(**data)
            logger.info("[stage_runner] var%03d stage%d 캐시 히트 (metric=%.4f) — 재학습 생략",
                        variant_id, stage, result.metric)
            return result
        except Exception as e:
            logger.warning("[stage_runner] var%03d 캐시 로드 실패 (%s) — 재학습", variant_id, e)
            return None

    def _save_cached(self, result: VariantResult) -> None:
        path = self._cache_path(result.stage, result.variant_id)
        if path is None:
            return
        try:
            path.write_text(json.dumps(result._asdict()), encoding="utf-8")
        except Exception as e:
            logger.warning("[stage_runner] var%03d 캐시 저장 실패: %s", result.variant_id, e)

    def _run_one(
        self, variant_id: int, params: dict, timesteps: int, stage: int
    ) -> VariantResult:
        # 캐시 히트 시 재학습 생략 (Colab 세션 중단 내결함성)
        cached = self._load_cached(stage, variant_id)
        if cached is not None:
            return cached

        logger.debug("[stage_runner] var%03d 시작 stage=%d timesteps=%d", variant_id, timesteps, stage)
        metric = self.train_fn(params, timesteps, variant_id)
        logger.debug("[stage_runner] var%03d 완료 metric=%.4f", variant_id, metric)
        result = VariantResult(
            variant_id=variant_id,
            params=params,
            metric=metric,
            timesteps=timesteps,
            stage=stage,
        )
        self._save_cached(result)
        return result
