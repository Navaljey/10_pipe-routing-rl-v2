"""Potential-Based Shaping (PBS) helper — CLAUDE.md §16.7.5 1:1 변환.

Ng, Harada & Russell (1999) 의 PBS 는 다음 4대 조건 하에서 optimal policy
보존이 이론적으로 증명된다 (CLAUDE.md §16.7.4):

    조건 1.  Φ(s) 는 state 만의 함수 (action 의존 금지)
    조건 2.  Φ(terminal_state) = 0
    조건 3.  r_shape = γ·Φ(s') - Φ(s) 형식 (단순 cost 차이 아님)
    조건 4.  (본 프로젝트 추가) γ_PBS == γ_PPO

본 모듈은 위 4대 조건을 type level + runtime check 로 hard-enforce 한다.
이 class 를 거치지 않은 `r_shape` 직접 작성은 SKILL.md §3.5 / §3.6 규칙 위반이다
(PotentialBasedShaping 은 다른 곳에서 import 만 — 정의 단일점).
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable

import numpy as np


class PBSSafetyWarning(UserWarning):
    """PBS 조건 1 의 잠재적 위반 가능성을 알리는 경고."""

__all__ = ["PotentialBasedShaping", "PBSSafetyWarning"]


class PotentialBasedShaping:
    """PBS 4대 조건 hard-enforced shaping reward wrapper.

    Parameters
    ----------
    phi_fn :
        Potential function Φ. **state 만** 인자로 받아야 한다 (조건 1).
        시그니처가 state-only 가 아니면 생성 시점에 ``ValueError``.
    gamma_ppo :
        PBS 에 사용할 discount factor. **PPO 의 gamma 와 반드시 동일 값**
        (조건 4). 호출 측에서 ``model.gamma`` 를 그대로 넘겨 강제한다.
        이 class 는 단일 gamma 만 보관하므로 γ_PBS == γ_PPO 가 구조적으로 보장된다.
    terminal_phi :
        Terminal state 에서의 Φ 값. PBS 조건 2 에 따라 ``0.0`` 고정이 정상이며,
        기본값을 0.0 으로 둔다.

    Notes
    -----
    조건 3, 4 는 ``shaping_reward`` 가 ``γ·Φ(s') - Φ(s)`` 한 형식으로만 계산하고
    그 γ 가 생성 시 보관한 ``gamma_ppo`` 단일값이므로 자동 충족된다.
    """

    def __init__(
        self,
        phi_fn: Callable[[np.ndarray], float],  # state-only 시그니처
        gamma_ppo: float,                       # MUST equal PPO's gamma
        terminal_phi: float = 0.0,              # PBS 조건 2
    ) -> None:
        # 조건 1 검증: phi_fn 이 state 만 받는지 시그니처 체크.
        sig = inspect.signature(phi_fn)
        params = list(sig.parameters.values())

        # 정확히 1개의 인자만 허용. *args / **kwargs 는 action 을 몰래 받는 통로가
        # 되므로 (조건 1 우회) 함께 거부한다.
        var_kinds = (
            inspect.Parameter.VAR_POSITIONAL,  # *args
            inspect.Parameter.VAR_KEYWORD,     # **kwargs
        )
        if len(params) != 1 or params[0].kind in var_kinds:
            raise ValueError(
                "PBS 조건 1 위반: Φ must be state-only (single argument). "
                f"got signature {sig}"
            )

        # 조건 1 보조 검증: closure 가 action-related 이름을 캡처하면 경고.
        # signature 1개 == state-only 는 필요 조건일 뿐, closure 로 action 을
        # 캡처해도 parameter 는 1개로 보인다 (의사결정 22).
        _ACTION_HINTS = frozenset({"action", "act", "a", "cmd"})
        captured = getattr(phi_fn, "__code__", None)
        if captured is not None:
            suspicious = _ACTION_HINTS.intersection(captured.co_freevars)
            if suspicious:
                warnings.warn(
                    f"PBS 조건 1 주의: phi_fn 이 {suspicious} 를 closure 로 캡처 중. "
                    "action-dependent 가 아닌지 수동 확인 필요.",
                    PBSSafetyWarning,
                    stacklevel=2,
                )

        self.phi_fn = phi_fn
        self.gamma = gamma_ppo
        self.terminal_phi = terminal_phi

    def shaping_reward(
        self,
        s: np.ndarray,
        s_next: np.ndarray,
        is_terminal: bool,
    ) -> float:
        """조건 2, 3, 4 를 만족하는 r_shape 를 계산한다.

        ``r_shape = γ·Φ(s') - Φ(s)`` (단, s' 가 terminal 이면 Φ(s') = terminal_phi).
        """
        # 조건 2: terminal Φ = 0 (명시적 set, 누락 시 정책 보존 보장 깨짐)
        phi_next = self.terminal_phi if is_terminal else float(self.phi_fn(s_next))
        phi_curr = float(self.phi_fn(s))
        # 조건 3, 4: γ_ppo·Φ(s') - Φ(s)
        return self.gamma * phi_next - phi_curr
