"""Phase 1 autoresearch Round 운영 오케스트레이터 — CLAUDE.md §16.6 / §16.6.7.

Round 운영 순서 (§16.6.7):
  Round 1: L-D2 α × β sweep (12 variants)
  Round 2: L-16.3-w 가중치 sweep (36 variants)  ← Round 1 best (α, β) 고정
  Round 3 (optional): hyperparameter sweep
  Round 4 (optional): interaction grid
  Round 5 (조건부): 추가 reward 항목 (4-Gate 통과 시)

Round 종료 조건 (§16.6.B — OR):
  1. round_n > MAX_ROUNDS (절대 상한 5)
  2. 최근 2 Round best 변형 동일 (개선 정체)
  3. plateau: 마지막 200K timestep 의 success_rate 증가 < 2%p
  4. 사용자 강제 종료 (stop() 호출)

한 Round 흐름:
  a) Stage 1 (N variants × 250K, worker 2~3)
  b) 하위 50% 제거 → survivors
  c) Stage 2 (survivors × 2M)
  d) best 1 선택 → 다음 Round 의 고정값으로 사용
  e) 종료 조건 평가 → 계속 or 종료

best variant 결정: Stage 2 metric (success_rate) 최고값.
다음 Round input: Round N best params 를 고정값으로 삽입 (Round 1 best → Round 2 에서 α, β 고정).

plateau 감지 (§16.6.B 조건 3):
  train_fn 이 metric_history 를 제공하는 경우 사용.
  제공 안 할 경우 metric_history=None → 조건 3 skip.
  구현 단순성: 마지막 2 Round 간 best metric 증가 < PLATEAU_THRESHOLD.
  wandb trajectory 는 별도 외부 panel (§16.7.8) 에서 확인.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from autoresearch.optuna_study import (
    create_study,
    enqueue_round1_grid,
    enqueue_round2_grid,
    get_param_importances,
)
from autoresearch.stage_runner import StageRunner, VariantResult, STAGE1_TIMESTEPS, STAGE2_TIMESTEPS

logger = logging.getLogger(__name__)

# ─── §16.6 고정 상수 ────────────────────────────────────────────────────────
MAX_ROUNDS: int = 5                    # Round 절대 상한 (§16.6.B 조건 1)
PLATEAU_THRESHOLD: float = 0.02       # 2%p — success_rate 개선 최소 임계값 (§16.6.B 조건 3)
STAGNATION_ROUNDS: int = 2            # N Round 연속 동일 best → 정체 (§16.6.B 조건 2)


@dataclass
class RoundResult:
    """단일 Round 완료 결과."""
    round_n: int
    best: VariantResult
    importances: dict[str, float]       # Optuna importance 분석 결과
    fixed_params: dict                  # 다음 Round 에 고정할 params


@dataclass
class OrchestratorState:
    """Round 간 공유 상태. 종료 조건 평가에 사용."""
    round_results: list[RoundResult] = field(default_factory=list)
    _stop_requested: bool = False

    def request_stop(self) -> None:
        """사용자 강제 종료 (§16.6.B 조건 4)."""
        self._stop_requested = True
        logger.info("[orchestrator] 사용자 강제 종료 요청됨.")

    @property
    def best_metrics(self) -> list[float]:
        return [r.best.metric for r in self.round_results]

    def is_plateau(self) -> bool:
        """§16.6.B 조건 3: 최근 2 Round best metric 증가 < 2%p."""
        metrics = self.best_metrics
        if len(metrics) < STAGNATION_ROUNDS:
            return False
        improvement = metrics[-1] - metrics[-2]
        if improvement < PLATEAU_THRESHOLD:
            logger.info(
                "[orchestrator] plateau 감지: Round %d→%d improvement=%.4f < %.4f",
                len(metrics) - 1, len(metrics), improvement, PLATEAU_THRESHOLD,
            )
            return True
        return False

    def is_stagnated(self) -> bool:
        """§16.6.B 조건 2: 최근 2 Round best 동일 (params 기준)."""
        if len(self.round_results) < STAGNATION_ROUNDS:
            return False
        last_two = self.round_results[-STAGNATION_ROUNDS:]
        if last_two[0].best.params == last_two[1].best.params:
            logger.info(
                "[orchestrator] 정체 감지: Round %d 와 %d 의 best params 동일 %s",
                last_two[0].round_n, last_two[1].round_n, last_two[1].best.params,
            )
            return True
        return False

    def should_stop(self, round_n: int) -> tuple[bool, str]:
        """종료 조건 OR 평가. (종료 여부, 이유) 반환."""
        if self._stop_requested:
            return True, "사용자 강제 종료"
        if round_n > MAX_ROUNDS:
            return True, f"Round {MAX_ROUNDS} 절대 상한 도달"
        if self.is_stagnated():
            return True, "2 Round 연속 best 동일 (개선 정체)"
        if self.is_plateau():
            return True, f"plateau 감지 (improvement < {PLATEAU_THRESHOLD*100:.0f}%p)"
        return False, ""


class RoundOrchestrator:
    """CLAUDE.md §16.6 / §16.6.7 Round 운영 오케스트레이터.

    Parameters
    ----------
    train_fn :
        (params, timesteps, variant_id) → float.
        StageRunner 과 동일 인터페이스.
    step_n : 현재 Step 번호 (1~10).
    max_workers : 동시 worker 수. 1=sequential.
    stage1_timesteps, stage2_timesteps : 테스트 시 오버라이드 가능.
    optuna_storage : sqlite URL. None 이면 인메모리 (테스트용).
    """

    def __init__(
        self,
        train_fn: Callable[[dict, int, int], float],
        step_n: int = 1,
        max_workers: int = 1,
        stage1_timesteps: int = STAGE1_TIMESTEPS,
        stage2_timesteps: int = STAGE2_TIMESTEPS,
        optuna_storage: str | None = "sqlite:///autoresearch.db",
        cache_dir: str | None = None,
    ) -> None:
        self.train_fn = train_fn
        self.step_n = step_n
        self.max_workers = max_workers
        self.stage1_timesteps = stage1_timesteps
        self.stage2_timesteps = stage2_timesteps
        self.optuna_storage = optuna_storage
        self.cache_dir = cache_dir   # Colab 내결함성 캐시 (None=비활성)
        self.state = OrchestratorState()

    # ─── public API ─────────────────────────────────────────────────────────

    def stop(self) -> None:
        """사용자 강제 종료 (§16.6.B 조건 4). 현재 Stage 완료 후 중단."""
        self.state.request_stop()

    def run_round1(self) -> RoundResult:
        """Round 1: L-D2 α × β sweep (12 variants). §16.6.7."""
        logger.info("[orchestrator] === Round 1 시작 (α × β sweep, 12 variants) ===")
        storage = self._storage_for_round(1)
        study = create_study(step_n=self.step_n, round_n=1, storage=storage)
        enqueue_round1_grid(study)

        # WAITING trial 에서 params 추출
        variants = self._waiting_params(study)
        logger.info("[orchestrator] Round 1 variants: %d", len(variants))

        result = self._run_round(round_n=1, variants=variants, fixed_params={})

        # importance 분석 (completed trial 이 충분하면)
        importances = get_param_importances(study)
        result = RoundResult(
            round_n=1,
            best=result.best,
            importances=importances,
            fixed_params=result.fixed_params,
        )
        self.state.round_results.append(result)
        logger.info(
            "[orchestrator] Round 1 완료: best metric=%.4f params=%s importances=%s",
            result.best.metric, result.best.params, importances,
        )
        return result

    def run_round2(self, round1_result: RoundResult) -> RoundResult:
        """Round 2: L-16.3-w sweep (36 variants). Round 1 best α, β 고정. §16.6.7."""
        # Round 1 best 에서 α, β 추출하여 고정
        fixed = {k: v for k, v in round1_result.best.params.items()
                 if k in ("alpha", "beta")}
        logger.info(
            "[orchestrator] === Round 2 시작 (w1 × w2 × w3 sweep, 36 variants, 고정: %s) ===",
            fixed,
        )

        storage = self._storage_for_round(2)
        study = create_study(step_n=self.step_n, round_n=2, storage=storage)
        enqueue_round2_grid(study)

        variants_base = self._waiting_params(study)
        # 각 variant 에 고정 α, β 삽입
        variants = [{**fixed, **v} for v in variants_base]
        logger.info("[orchestrator] Round 2 variants: %d (α=%.2f, β=%.2f 고정)",
                    len(variants), fixed.get("alpha", 0), fixed.get("beta", 0))

        result = self._run_round(round_n=2, variants=variants, fixed_params=fixed)
        result = RoundResult(
            round_n=2,
            best=result.best,
            importances=get_param_importances(study),
            fixed_params=result.fixed_params,
        )
        self.state.round_results.append(result)
        logger.info(
            "[orchestrator] Round 2 완료: best metric=%.4f params=%s",
            result.best.metric, result.best.params,
        )
        return result

    def run_round_generic(
        self,
        round_n: int,
        variants: list[dict],
        fixed_params: dict,
    ) -> RoundResult:
        """Round 3~5 범용 실행. variants 와 fixed_params 는 호출자가 결정. §16.6.7."""
        logger.info(
            "[orchestrator] === Round %d 시작 (%d variants, 고정: %s) ===",
            round_n, len(variants), fixed_params,
        )
        result = self._run_round(round_n=round_n, variants=variants, fixed_params=fixed_params)
        storage = self._storage_for_round(round_n)
        study = create_study(step_n=self.step_n, round_n=round_n, storage=storage)
        result = RoundResult(
            round_n=round_n,
            best=result.best,
            importances=get_param_importances(study),
            fixed_params=result.fixed_params,
        )
        self.state.round_results.append(result)
        logger.info(
            "[orchestrator] Round %d 완료: best metric=%.4f",
            round_n, result.best.metric,
        )
        return result

    def run_phase1(self) -> list[RoundResult]:
        """Phase 1 전체 자동 실행: Round 1 → 2 → (종료 조건 충족 시 중단). §16.6.7.

        Round 3~5 는 사용자가 직접 run_round_generic() 으로 제어.
        본 메서드는 Round 1, 2 자동 실행 + 종료 조건 평가.
        """
        # Round 1
        stop, reason = self.state.should_stop(round_n=1)
        if stop:
            logger.info("[orchestrator] Round 1 진입 전 종료: %s", reason)
            return list(self.state.round_results)

        r1 = self.run_round1()

        # Round 2
        stop, reason = self.state.should_stop(round_n=2)
        if stop:
            logger.info("[orchestrator] Round 2 진입 전 종료: %s", reason)
            return list(self.state.round_results)

        self.run_round2(r1)

        # Round 2 이후 종료 조건 평가 (Round 3~5 는 사용자 제어)
        stop, reason = self.state.should_stop(round_n=3)
        if stop:
            logger.info("[orchestrator] Phase 1 자동 종료 (Round 3 진입 전): %s", reason)

        return list(self.state.round_results)

    # ─── internal ───────────────────────────────────────────────────────────

    def _run_round(
        self,
        round_n: int,
        variants: list[dict],
        fixed_params: dict,
    ) -> RoundResult:
        """Stage 1 → Stage 2 실행. best params + fixed_params 를 다음 Round 고정값으로 반환."""
        runner = StageRunner(
            train_fn=self.train_fn,
            max_workers=self.max_workers,
            stage1_timesteps=self.stage1_timesteps,
            stage2_timesteps=self.stage2_timesteps,
            cache_dir=self.cache_dir,
        )
        best, _ = runner.run_full(variants)

        # 다음 Round 고정값 = 현재 Round 고정값 + best params
        next_fixed = {**fixed_params, **best.params}

        return RoundResult(
            round_n=round_n,
            best=best,
            importances={},          # 호출자가 Optuna study 에서 채움
            fixed_params=next_fixed,
        )

    def _storage_for_round(self, round_n: int) -> str | None:
        """Optuna storage URL. None 이면 인메모리."""
        if self.optuna_storage is None:
            return None
        return self.optuna_storage

    @staticmethod
    def _waiting_params(study) -> list[dict]:
        """WAITING trial 에서 enqueue 된 params 추출.

        WAITING trial 은 t.params 가 비어있고
        실제 값은 t.system_attrs['fixed_params'] 에 있음 (Optuna 내부 동작).
        """
        import optuna
        result = []
        for t in study.trials:
            if t.state == optuna.trial.TrialState.WAITING:
                fp = t.system_attrs.get("fixed_params", {})
                if fp:
                    result.append(dict(fp))
        return result
