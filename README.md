# pipe-routing-rl-v2

3D 공간 파이프 자동 배치 강화학습 시스템. 단일 에이전트 + 전이학습 + Dense Reward (PBS) + autoresearch.

## 문서 읽는 순서

1. **CLAUDE.md** — 프로젝트 spec 본체 (WHAT)
2. **SKILL.md** — 구현 규칙 (HOW)
3. **PROGRESS.md** — 의사결정 이력 (WHY)
4. **FAILURE_LOG.md** — 실패 / 시행착오 archive

## 현재 상태

- spec v1.0 졸업 (2026-06-24)
- Phase 1 (Step 1) harness engineering 진행 가능
- 활성 Lock 4개: L-A3', L-A4, L-C4, L-D3 (자기 자연 시점 해제)

## 시작

```bash
# 의존성 설치 (poetry / uv / pip 중 선택)
poetry install   # 또는: pip install -r requirements.txt

# Phase 1 학습 시작
python training/train_step1.py
```