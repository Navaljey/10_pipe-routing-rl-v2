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
from typing import Any, Callable

from autoresearch.optuna_study import (
    ask_waiting_trials,
    create_study,
    enqueue_round1_grid,
    enqueue_round2_grid,
    get_param_importances,
    register_grid_params,
)
from autoresearch.stage_runner import StageRunner, VariantResult, STAGE1_TIMESTEPS, STAGE2_TIMESTEPS
from autoresearch.wandb_callback import (
    init_orchestrator_run,
    log_round_summary,
    log_stage1_summary,
    log_stage2_summary,
)

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
        """§16.6.B 조건 3: 최근 2 Round best metric 증가 < 2.0%p (부동소수점 epsilon 1e-6 허용)."""
        metrics = self.best_metrics
        if len(metrics) < STAGNATION_ROUNDS:
            return False
        improvement = metrics[-1] - metrics[-2]
        # 부동소수점 안전성: 0.02 정확히에서 오차로 잘못 감지하는 경우 방지
        if improvement < (PLATEAU_THRESHOLD - 1e-9):
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
        wandb_mode: str = "disabled",
    ) -> None:
        self.train_fn = train_fn
        self.step_n = step_n
        self.max_workers = max_workers
        self.stage1_timesteps = stage1_timesteps
        self.stage2_timesteps = stage2_timesteps
        self.optuna_storage = optuna_storage
        self.cache_dir = cache_dir   # Colab 내결함성 캐시 (None=비활성)
        self.wandb_mode = wandb_mode  # 'online'|'offline'|'disabled'. 기본 disabled (테스트/로컬)
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

        # WAITING trial 을 ask() 로 pop (RUNNING 전환) 하며 params 추출.
        # study.tell() 은 RUNNING trial 에만 적용 가능하므로 (WAITING 은 불가)
        # 실행 전에 미리 전환해 둔다. §16.6.5.
        trials = ask_waiting_trials(study)
        # suggest_categorical() 으로 params/distributions 를 채워야 아래
        # get_param_importances() 가 실제로 계산 가능해진다 (§16.6.5).
        register_grid_params(trials)
        variants = [dict(t.system_attrs.get("fixed_params", {})) for t in trials]
        logger.info("[orchestrator] Round 1 variants: %d", len(variants))

        result = self._run_round(
            round_n=1, variants=variants, fixed_params={}, study=study, trials=trials,
        )

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

        trials = ask_waiting_trials(study)
        register_grid_params(trials)
        variants_base = [dict(t.system_attrs.get("fixed_params", {})) for t in trials]
        # 각 variant 에 고정 α, β 삽입 (trials 와 순서 동일하게 유지)
        variants = [{**fixed, **v} for v in variants_base]
        logger.info("[orchestrator] Round 2 variants: %d (α=%.2f, β=%.2f 고정)",
                    len(variants), fixed.get("alpha", 0), fixed.get("beta", 0))

        result = self._run_round(
            round_n=2, variants=variants, fixed_params=fixed, study=study, trials=trials,
        )
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
        study: Any | None = None,
        trials: list[Any] | None = None,
    ) -> RoundResult:
        """Stage 1 → Stage 2 실행. best params + fixed_params 를 다음 Round 고정값으로 반환.

        study/trials 가 주어지면 (run_round1/run_round2 표준 경로) 각 variant 의
        최종 metric 을 study.tell() 로 대응 Optuna trial 에 반영해
        WAITING(ask 이후엔 RUNNING) → COMPLETE 로 전환한다:
          - Stage 1 탈락 variant → Stage 1 metric 이 최종값
          - Stage 2 까지 도달한 variant (survivor) → Stage 2 metric 이 최종값
        study/trials 가 없으면 (run_round_generic 처럼 enqueue 를 거치지 않은
        variants) trial 완료 처리를 건너뛴다.

        한계: run_full() 이 stage1_survivors 캐시를 히트해 Stage 1 을 스킵하면
        (세션 재개 시) on_stage1_complete 콜백이 호출되지 않으므로, 그 세션에서
        Stage 1 탈락했던 variant 는 이번 호출에서 tell() 되지 않는다 (직전
        세션에서 이미 처리되었어야 함).

        오케스트레이터 wandb run (step{N}_round{R}_orch) 을 열고
        Stage 1/2 집계 summary 를 기록한다.
        wandb_mode='disabled' 이면 run 은 noop (테스트/로컬 환경).
        """
        import wandb as _wandb

        orch_run = init_orchestrator_run(
            step_n=self.step_n,
            round_n=round_n,
            mode=self.wandb_mode,
        )

        def _tell(result: VariantResult) -> None:
            if study is None or trials is None or result.variant_id >= len(trials):
                return
            trial = trials[result.variant_id]
            try:
                study.tell(trial.number, result.metric)
            except Exception as e:
                logger.warning(
                    "[orchestrator] var%03d → trial#%d study.tell 실패: %s",
                    result.variant_id, trial.number, e,
                )

        def _on_stage1(all_results, survivors):
            log_stage1_summary(self.step_n, round_n, all_results, survivors)
            survivor_ids = {r.variant_id for r in survivors}
            for r in all_results:
                if r.variant_id not in survivor_ids:
                    _tell(r)   # Stage 1 탈락 → Stage 1 metric 으로 trial 완료

        def _on_stage2(all_results, best):
            log_stage2_summary(self.step_n, round_n, all_results, best)
            for r in all_results:
                _tell(r)       # Stage 2 도달 → Stage 2 metric 으로 trial 완료

        runner = StageRunner(
            train_fn=self.train_fn,
            max_workers=self.max_workers,
            stage1_timesteps=self.stage1_timesteps,
            stage2_timesteps=self.stage2_timesteps,
            cache_dir=self.cache_dir,
            on_stage1_complete=_on_stage1,
            on_stage2_complete=_on_stage2,
        )
        best, _ = runner.run_full(variants)

        # 다음 Round 고정값 = 현재 Round 고정값 + best params
        next_fixed = {**fixed_params, **best.params}

        # Round 종료 summary (importances 는 호출자가 채운 후 별도 log 가능)
        log_round_summary(round_n, next_fixed, importances={})

        orch_run.finish()
        logger.info("[orchestrator] wandb orch run 종료: step%d_round%d_orch", self.step_n, round_n)

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
