"""Training infrastructure — regression callback + handoff. CLAUDE.md §12.3, §13.1."""

from training.handoff import HandoffConfig, load_handoff, make_regression_report, save_handoff
from training.regression_callback import (
    EscalationController,
    EscalationThresholds,
    RegressionCallback,
    StepMetrics,
    evaluate_step_policy,
)

__all__ = [
    "EscalationController",
    "EscalationThresholds",
    "HandoffConfig",
    "RegressionCallback",
    "StepMetrics",
    "evaluate_step_policy",
    "load_handoff",
    "make_regression_report",
    "save_handoff",
]
