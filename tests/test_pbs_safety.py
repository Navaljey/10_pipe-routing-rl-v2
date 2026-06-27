"""PBS 4대 조건 강제 검증 — CLAUDE.md §16.7.4 / §16.7.5, SKILL.md §3.5.

조건 1. Φ(s) 는 state 만의 함수 (action 의존 금지)
조건 2. Φ(terminal) = 0  (goal / collision / timeout / out_of_bounds 전부)
조건 3. r_shape = γ·Φ(s') - Φ(s) 형식
조건 4. γ_PBS == γ_PPO

각 테스트는 위 조건이 코드 레벨에서 hard-enforce 됨을 증명한다.
"""

import warnings

import numpy as np
import pytest

from shaping import PotentialBasedShaping, PBSSafetyWarning
from shaping.potential_based import PotentialBasedShaping as PBSDirect

pytestmark = pytest.mark.pbs


# ─────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────
GAMMA = 0.99


def _linear_phi(s: np.ndarray) -> float:
    """테스트용 state-only Φ. Φ(s) = -||s|| (단조, 검산 용이)."""
    return -float(np.linalg.norm(s))


@pytest.fixture
def pbs() -> PotentialBasedShaping:
    return PotentialBasedShaping(phi_fn=_linear_phi, gamma_ppo=GAMMA)


# ─────────────────────────────────────────────────────────────────
# 정의 단일점 (SKILL.md §3.6)
# ─────────────────────────────────────────────────────────────────
def test_single_source_of_definition():
    """패키지 re-export 와 모듈 직접 import 가 동일 class 여야 한다."""
    assert PotentialBasedShaping is PBSDirect


# ─────────────────────────────────────────────────────────────────
# 조건 1 — Φ 는 state-only
# ─────────────────────────────────────────────────────────────────
def test_condition1_accepts_state_only_phi():
    """단일 인자 Φ 는 생성 가능."""
    PotentialBasedShaping(phi_fn=lambda s: 0.0, gamma_ppo=GAMMA)


def test_condition1_rejects_action_dependent_phi():
    """Φ(s, a) 처럼 인자가 2개면 거부 (action 의존)."""
    with pytest.raises(ValueError, match="조건 1"):
        PotentialBasedShaping(phi_fn=lambda s, a: 0.0, gamma_ppo=GAMMA)


def test_condition1_warns_on_action_capturing_closure():
    """closure 가 'action' 이름을 캡처하면 PBSSafetyWarning 을 발행한다 (의사결정 22).

    lambda s: s[action] 은 parameter 가 1개이므로 signature 검증을 통과하지만,
    외부 action 변수에 의존하므로 state-only 위반 가능성이 있다.
    """
    action = 2
    phi_with_action = lambda s: float(s[action])  # noqa: E731 — 테스트 목적

    with pytest.warns(PBSSafetyWarning, match="action"):
        PotentialBasedShaping(phi_fn=phi_with_action, gamma_ppo=GAMMA)


def test_condition1_no_warning_for_clean_closure():
    """action-unrelated 이름만 캡처하는 closure 는 경고 없이 생성된다."""
    scale = 0.5
    phi_scaled = lambda s: -scale * float(np.linalg.norm(s))  # noqa: E731

    with warnings.catch_warnings():
        warnings.simplefilter("error", PBSSafetyWarning)
        PotentialBasedShaping(phi_fn=phi_scaled, gamma_ppo=GAMMA)  # 경고 없어야 함


def test_condition1_rejects_zero_arg_phi():
    """인자 0개도 거부 (state 를 안 받음)."""
    with pytest.raises(ValueError, match="조건 1"):
        PotentialBasedShaping(phi_fn=lambda: 0.0, gamma_ppo=GAMMA)


def test_condition1_rejects_var_positional():
    """*args 는 action 을 몰래 받는 통로이므로 거부."""
    with pytest.raises(ValueError, match="조건 1"):
        PotentialBasedShaping(phi_fn=lambda *args: 0.0, gamma_ppo=GAMMA)


def test_condition1_rejects_var_keyword():
    """**kwargs 도 거부."""
    with pytest.raises(ValueError, match="조건 1"):
        PotentialBasedShaping(phi_fn=lambda **kw: 0.0, gamma_ppo=GAMMA)


def test_condition1_shaping_reward_never_passes_action(pbs):
    """shaping_reward 는 Φ 에 state 만 전달 (action 을 넘기지 않는다)."""
    seen_args = []

    def spy_phi(s):
        seen_args.append(s)
        return 0.0

    p = PotentialBasedShaping(phi_fn=spy_phi, gamma_ppo=GAMMA)
    s, s_next = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    p.shaping_reward(s, s_next, is_terminal=False)
    # s 와 s_next 두 번만 호출, 각각 state 하나씩
    assert len(seen_args) == 2


# ─────────────────────────────────────────────────────────────────
# 조건 2 — Φ(terminal) = 0
# ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("terminal_kind", ["goal", "collision", "timeout", "out_of_bounds"])
def test_condition2_terminal_phi_is_zero(terminal_kind):
    """모든 terminal 종류에서 Φ(s') 가 terminal_phi(=0) 로 대체된다.

    s_next 의 비-terminal Φ 가 큰 음수여도 r_shape 에는 0 이 들어가야 한다.
    """
    s = np.array([0.0, 0.0])               # Φ(s) = 0
    s_next = np.array([100.0, 0.0])        # Φ(s_next) = -100 (비-terminal 값)
    p = PotentialBasedShaping(phi_fn=_linear_phi, gamma_ppo=GAMMA)

    r_terminal = p.shaping_reward(s, s_next, is_terminal=True)
    # terminal 이면 phi_next=0 → r = γ·0 - Φ(s) = -0 = 0
    assert r_terminal == pytest.approx(0.0)


def test_condition2_nonterminal_uses_real_phi():
    """비-terminal 에서는 실제 Φ(s') 를 사용 (terminal_phi 로 덮지 않음)."""
    s = np.array([0.0, 0.0])               # Φ(s) = 0
    s_next = np.array([100.0, 0.0])        # Φ(s_next) = -100
    p = PotentialBasedShaping(phi_fn=_linear_phi, gamma_ppo=GAMMA)

    r = p.shaping_reward(s, s_next, is_terminal=False)
    assert r == pytest.approx(GAMMA * (-100.0) - 0.0)


def test_condition2_custom_terminal_phi_respected():
    """terminal_phi 를 명시하면 그 값이 phi_next 로 쓰인다."""
    p = PotentialBasedShaping(phi_fn=_linear_phi, gamma_ppo=GAMMA, terminal_phi=7.0)
    s = np.array([0.0, 0.0])
    s_next = np.array([3.0, 4.0])          # Φ = -5 (무시되어야 함)
    r = p.shaping_reward(s, s_next, is_terminal=True)
    assert r == pytest.approx(GAMMA * 7.0 - 0.0)


# ─────────────────────────────────────────────────────────────────
# 조건 3 — r_shape = γ·Φ(s') - Φ(s) 형식
# ─────────────────────────────────────────────────────────────────
def test_condition3_pbs_formula(pbs):
    """r_shape 가 γ·Φ(s') - Φ(s) 와 정확히 일치 (단순 cost 차이가 아님)."""
    s = np.array([3.0, 4.0])               # Φ = -5
    s_next = np.array([6.0, 8.0])          # Φ = -10
    expected = GAMMA * _linear_phi(s_next) - _linear_phi(s)
    assert pbs.shaping_reward(s, s_next, is_terminal=False) == pytest.approx(expected)


def test_condition3_not_plain_cost_difference(pbs):
    """단순 차이 Φ(s')-Φ(s) (γ 미적용) 와는 달라야 한다 (γ≠1 일 때)."""
    s = np.array([3.0, 4.0])               # Φ = -5
    s_next = np.array([6.0, 8.0])          # Φ = -10
    plain_diff = _linear_phi(s_next) - _linear_phi(s)   # γ 미적용
    pbs_value = pbs.shaping_reward(s, s_next, is_terminal=False)
    assert pbs_value != pytest.approx(plain_diff)


def test_condition3_zero_when_potential_constant():
    """γ=1 이고 Φ 가 상수면 r_shape = 0 (이론적 무영향성 확인)."""
    p = PotentialBasedShaping(phi_fn=lambda s: 3.0, gamma_ppo=1.0)
    s, s_next = np.array([1.0]), np.array([2.0])
    assert p.shaping_reward(s, s_next, is_terminal=False) == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────
# 조건 4 — γ_PBS == γ_PPO
# ─────────────────────────────────────────────────────────────────
def test_condition4_gamma_stored_equals_ppo():
    """생성 시 받은 gamma_ppo 가 그대로 PBS γ 로 보관된다."""
    p = PotentialBasedShaping(phi_fn=_linear_phi, gamma_ppo=0.95)
    assert p.gamma == 0.95


def test_condition4_gamma_used_in_formula():
    """shaping_reward 가 실제로 보관한 γ 를 곱한다 (다른 γ 가 끼어들 여지 없음)."""
    for gamma in (0.5, 0.9, 0.99, 1.0):
        p = PotentialBasedShaping(phi_fn=_linear_phi, gamma_ppo=gamma)
        s = np.array([1.0, 0.0])           # Φ = -1
        s_next = np.array([0.0, 2.0])      # Φ = -2
        expected = gamma * (-2.0) - (-1.0)
        assert p.shaping_reward(s, s_next, is_terminal=False) == pytest.approx(expected)


def test_condition4_single_gamma_no_divergence():
    """PBS γ 와 PPO γ 가 분리 저장되지 않음 — 단일 gamma 속성만 존재.

    구조적으로 γ_PBS != γ_PPO 가 발생할 수 없음을 회귀 방지로 못 박는다.
    """
    p = PotentialBasedShaping(phi_fn=_linear_phi, gamma_ppo=GAMMA)
    gamma_attrs = [a for a in vars(p) if "gamma" in a.lower()]
    assert gamma_attrs == ["gamma"]


# ─────────────────────────────────────────────────────────────────
# 통합 — 실제 Phase 1 Φ (CLAUDE.md §16.7.2) 모사
# ─────────────────────────────────────────────────────────────────
def test_phase1_phi_goal_plus_cong_integration():
    """α·Φ_goal + β·Φ_cong 합성 Φ 도 state-only 라 정상 동작."""
    alpha, beta = 1.0, 0.1
    d_initial = 20.0

    def phi_goal(s):
        return -float(np.abs(s).sum()) / d_initial   # -d_manhattan / d_initial

    def phi_cong(s):
        mean_sdf = 3.0                                # 모사
        return -1.0 / (mean_sdf + 1.0)

    def phi(s):
        return alpha * phi_goal(s) + beta * phi_cong(s)

    p = PotentialBasedShaping(phi_fn=phi, gamma_ppo=GAMMA)
    s = np.array([10.0, 0.0, 0.0])
    s_next = np.array([9.0, 0.0, 0.0])               # goal 에 1칸 접근
    r = p.shaping_reward(s, s_next, is_terminal=False)

    expected = GAMMA * phi(s_next) - phi(s)
    assert r == pytest.approx(expected)
    # goal 에 가까워졌으므로 Φ_goal 증가 → shaping reward 양수 경향
    assert r > 0
