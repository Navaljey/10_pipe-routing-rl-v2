"""L-C4 좌표계 unit test — CLAUDE.md §9, §99 L-C4.

cell ↔ mm 변환 정밀도 및 오른손 좌표계 정합성 검증.
모든 테스트 통과 시 L-C4 사실상 해제 (CLAUDE.md §99 L-C4 안내).
"""

import numpy as np
import pytest

from envs.base_env import (
    DEFAULT_SPACE_MM,
    FACE_DIRS,
    GRID_SHAPE,
    N_FACE_DIRS,
    cell_to_mm,
    mm_to_cell,
)

pytestmark = pytest.mark.coord

# 기본 cell_size_mm (DEFAULT_SPACE_MM / GRID_SHAPE)
CELL_SIZE_MM: np.ndarray = DEFAULT_SPACE_MM / np.array(GRID_SHAPE, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Roundtrip: cell → mm → cell
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "cell",
    [
        (0, 0, 0),
        (29, 29, 29),
        (15, 10, 20),
        (0, 29, 15),
        (29, 0, 0),
    ],
)
def test_roundtrip_cell_to_mm_to_cell(cell: tuple[int, int, int]) -> None:
    """cell → mm → cell 왕복이 원래 cell 과 정확히 일치해야 한다. L-C4."""
    c = np.array(cell, dtype=np.int32)
    pos_mm = cell_to_mm(c, CELL_SIZE_MM)
    c_back = mm_to_cell(pos_mm, CELL_SIZE_MM)
    np.testing.assert_array_equal(c_back, c)


# ─────────────────────────────────────────────────────────────────────────────
# 2. 셀 중심점 정확성
# ─────────────────────────────────────────────────────────────────────────────


def test_cell_center_is_midpoint() -> None:
    """cell_to_mm 은 셀 중심을 반환한다 (CELL_SIZE_MM * (idx + 0.5))."""
    c = np.array([0, 0, 0], dtype=np.int32)
    np.testing.assert_allclose(cell_to_mm(c, CELL_SIZE_MM), CELL_SIZE_MM * 0.5)


def test_cell_center_last_cell() -> None:
    """마지막 셀 (29,29,29) 의 중심은 공간 최대 mm - CELL_SIZE_MM/2."""
    c = np.array([29, 29, 29], dtype=np.int32)
    expected = DEFAULT_SPACE_MM - CELL_SIZE_MM * 0.5
    np.testing.assert_allclose(cell_to_mm(c, CELL_SIZE_MM), expected)


# ─────────────────────────────────────────────────────────────────────────────
# 3. 경계 셀 (boundary cells)
# ─────────────────────────────────────────────────────────────────────────────


def test_boundary_cell_zero_is_valid() -> None:
    """경계 셀 (0, 0, 0) 은 유효하게 처리된다."""
    c = np.array([0, 0, 0], dtype=np.int32)
    result = mm_to_cell(cell_to_mm(c, CELL_SIZE_MM), CELL_SIZE_MM)
    np.testing.assert_array_equal(result, c)


def test_boundary_cell_max_is_valid() -> None:
    """경계 셀 (29, 29, 29) 은 유효하게 처리된다."""
    c = np.array([29, 29, 29], dtype=np.int32)
    result = mm_to_cell(cell_to_mm(c, CELL_SIZE_MM), CELL_SIZE_MM)
    np.testing.assert_array_equal(result, c)


# ─────────────────────────────────────────────────────────────────────────────
# 4. 범위 밖 좌표 거부 (L-C4 핵심 요구사항)
# ─────────────────────────────────────────────────────────────────────────────


def test_rejects_negative_x() -> None:
    """X 축 음수 mm 좌표는 ValueError 를 발생시킨다."""
    with pytest.raises(ValueError, match="범위 밖"):
        mm_to_cell(np.array([-1.0, 0.0, 0.0]), CELL_SIZE_MM)


def test_rejects_negative_all_axes() -> None:
    """모든 축이 음수인 좌표도 거부한다."""
    with pytest.raises(ValueError):
        mm_to_cell(np.array([-100.0, -200.0, -50.0]), CELL_SIZE_MM)


def test_rejects_beyond_space_x() -> None:
    """X 축 공간 크기 초과 좌표는 거부한다."""
    pos = np.array([DEFAULT_SPACE_MM[0] + 1.0, CELL_SIZE_MM[1] * 0.5, CELL_SIZE_MM[2] * 0.5])
    with pytest.raises(ValueError, match="범위 밖"):
        mm_to_cell(pos, CELL_SIZE_MM)


def test_rejects_beyond_space_z() -> None:
    """Z 축 공간 크기 초과 좌표는 거부한다."""
    pos = np.array([CELL_SIZE_MM[0] * 0.5, CELL_SIZE_MM[1] * 0.5, DEFAULT_SPACE_MM[2] + 1.0])
    with pytest.raises(ValueError, match="범위 밖"):
        mm_to_cell(pos, CELL_SIZE_MM)


def test_exactly_at_space_boundary_is_rejected() -> None:
    """pos_mm == space_size_mm 은 grid 밖(cell=30) 이므로 거부한다."""
    with pytest.raises(ValueError):
        mm_to_cell(DEFAULT_SPACE_MM.copy(), CELL_SIZE_MM)


# ─────────────────────────────────────────────────────────────────────────────
# 5. 단위 일관성 (오른손 좌표계, mm, CLAUDE.md §9)
# ─────────────────────────────────────────────────────────────────────────────


def test_cell_size_is_positive() -> None:
    """cell_size_mm 은 모든 축에서 양수여야 한다."""
    assert np.all(CELL_SIZE_MM > 0)


def test_face_dirs_right_hand_opposites() -> None:
    """FACE_DIRS 각 방향의 반대가 목록 안에 존재한다 (오른손 6방향). CLAUDE.md §9, §12.4."""
    for i, d in enumerate(FACE_DIRS):
        has_opposite = any(
            np.array_equal(-d, FACE_DIRS[j]) for j in range(N_FACE_DIRS) if j != i
        )
        assert has_opposite, f"FACE_DIRS[{i}]={d} 의 반대 방향이 없음"


def test_face_dirs_are_unit_l1() -> None:
    """모든 FACE_DIRS 는 L1 norm 이 1 인 단위 벡터여야 한다."""
    for d in FACE_DIRS:
        assert np.abs(d).sum() == 1, f"{d} 의 L1 norm != 1"


def test_n_face_dirs_constant() -> None:
    """N_FACE_DIRS 는 6 (SKILL §3.1 절대 변경 금지 항목)."""
    assert N_FACE_DIRS == 6


# ─────────────────────────────────────────────────────────────────────────────
# 6. 다른 cell_size_mm 로도 정합성 확인
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "space_mm,grid_n",
    [
        ([10_000.0, 10_000.0, 10_000.0], 30),  # 정방형 공간
        ([6_000.0, 9_000.0, 3_000.0], 30),  # 비정방형
    ],
)
def test_roundtrip_various_space_sizes(space_mm: list[float], grid_n: int) -> None:
    """다양한 공간 크기에서도 왕복 정합성이 유지된다."""
    space = np.array(space_mm, dtype=np.float64)
    csz = space / grid_n
    gshape = (grid_n, grid_n, grid_n)
    for cell in [(0, 0, 0), (grid_n - 1, grid_n - 1, grid_n - 1), (10, 15, 5)]:
        c = np.array(cell, dtype=np.int32)
        pos = cell_to_mm(c, csz)
        c_back = mm_to_cell(pos, csz, grid_shape=gshape)
        np.testing.assert_array_equal(c_back, c)


# ─────────────────────────────────────────────────────────────────────────────
# 7. BaseEnv instance 메서드 편의 래퍼 확인
# ─────────────────────────────────────────────────────────────────────────────


def test_base_env_instance_wrappers() -> None:
    """BaseEnv.cell_to_mm / mm_to_cell 은 module-level 함수와 동일 결과를 반환한다."""
    from envs.step1_env import Step1Env

    env = Step1Env()
    c = np.array([5, 10, 15], dtype=np.int32)

    mm_via_module = cell_to_mm(c, env.cell_size_mm)
    mm_via_method = env.cell_to_mm(c)
    np.testing.assert_array_equal(mm_via_method, mm_via_module)

    c_back_module = mm_to_cell(mm_via_module, env.cell_size_mm)
    c_back_method = env.mm_to_cell(mm_via_module)
    np.testing.assert_array_equal(c_back_method, c_back_module)
    np.testing.assert_array_equal(c_back_method, c)
