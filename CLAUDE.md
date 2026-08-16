# CLAUDE.md — 파이프 자동배치 강화학습 프로젝트 (단일 에이전트 + 전이학습)

> 본 문서는 `reference/common_spec.md`를 참조하여 정의된 본 프로젝트의 spec 본체이다.
>
> **🎓 2026-06-24 졸업**: 본 spec 은 `_ing` 시스템에서 졸업했다 (`CLAUDE_ing.md` → `CLAUDE.md`).
> Phase 1 학습 시작 조건 충족 — harness engineering 진행 가능 상태.
> 남은 활성 Lock 4개 (L-A3', L-A4, L-C4, L-D3) 는 모두 자기 자연 시점에 해제 예정 (각 Lock 의 해제 시점 안내는 §99 참조).
>
> **⚠️ 2026-05-14 구조 변경**: Hierarchical (Macro+Micro 별도 네트워크) 구조 폐기됨.
> Phase 1 실패 양상 재진단 결과 "Local부터 부실"이 확인되어, Hierarchical RL의 표준 가정 ("low-level이 reasonably well 학습된 후 high-level이 조립") 이 깨진 상태임이 드러남. 본 spec은 **단일 에이전트 + 전이학습** 구조로 채택됨. 자세한 의사결정 근거는 `PROGRESS.md` Session 2026-05-14 참조.

---

## 0. 작성 상태 요약

| 항목 | 상태 |
|------|:----:|
| 전체 구조 (단일 에이전트 + 전이학습) | ✅ Resolved |
| Step 1 학습/평가 체계 (Phase 1 대상) | ✅ Resolved (시작 가능) |
| Step 2~6 학습/평가 체계 | ✅ Resolved (세부 가중치는 각 Step 진입 시 결정) |
| Step 7~10 학습/평가 체계 | ⚠️ 골격만 Resolved (L-D1 후속, Step 7 진입 시 보강) |
| Dense Reward (PBS) Phase 1 적용 | ✅ Resolved (L-D2 해제) |
| Macro-action 도입 검토 | ⚠️ Phase 1 후 결정 (L-D3) |
| 평가 체계 Layer 1/2 분리 (Step 4~6) | ✅ Resolved (세부 hard constraint 는 L-A4) |
| Step 1~3 평가 임계값 | ⚠️ Phase 1 결과 후 후행 결정 (L-A3') |
| 시나리오 set + Generator | ✅ Resolved (L-A5, L-A6 해제) |
| baseline reward 가중치 (Step 1) | ✅ Resolved (L-16.3-w 해제) |
| 전이학습 + 회귀 감시 체계 | ✅ Resolved (L-D1 해제) |
| autoresearch 운영 (Optuna + wandb) | ✅ Resolved (L-C2 해제) |
| 좌표계 unit test | ⚠️ implementation 시 위임 (L-C4) |
| 물리/공학 정의 | ✅ Resolved (common_spec 계승) |
| **harness engineering 시작 가능 여부** | **✅ 가능** (2026-06-24 졸업) |

---

## 1. 프로젝트 비전

3D 공간 내 파이프를 자동 배치하는 강화학습 시스템을 구축한다.

**최종 목표:** 숙련 엔지니어를 **뛰어넘는** 품질의 파이프 라우팅을 자동 생성.

**근본 철학:**
- 숙련 엔지니어 시연을 정답으로 삼지 않는다 (시연은 엔지니어마다 다르고, 시연을 정답으로 삼으면 시연을 천장으로 만든다).
- 대신 **물리적으로 측정 가능한 평가 기준**을 세우고, 그 기준 위에서 RL이 자율 탐색한다.
- 평가 기준은 step에 따라 진화한다: A* benchmark → 물리 metric → 다중 파이프 metric.

---

## 2. 전체 아키텍처: 단일 에이전트 + 전이학습

> **⚠️ 2026-05-14 변경**: 이전 v0.1의 Hierarchical (Macro+Micro 별도 네트워크) 구조 폐기. 자세한 근거는 PROGRESS.md Session 2026-05-14, 의사결정 3-r1/3-r2 참조.

### 2.1 단일 에이전트 구조

```
┌─────────────────────────────────────────────────────┐
│  Single Agent (Step 1~10 공통)                       │
│   - 입력: 150-dim observation (Zero-padding)         │
│   - 출력: 7-direction discrete + (잠정) macro-action │
│   - 알고리즘: sb3-contrib MaskablePPO               │
│   - Backbone hidden dim: 256 (전 Step 고정)         │
│   - Reward: baseline (§16.3) + Dense PBS (§16.7)    │
└─────────────────────────────────────────────────────┘
```

### 2.2 Step 간 전이학습

```
Step 1 학습 → Step 1 정책 → Step 2 초기화 → Step 2 학습 → ... → Step 10

전이학습 전략 (구체 선택은 §99-Locked L-D1):
  A. 순차적 fine-tuning
  B. Mixed curriculum (Step 1~N 시나리오 혼합 샘플링)
  C. EWC regularization
  D. LoRA Adapter
```

### 2.3 본 구조의 정당성

- Phase 1 실패 양상이 "Local부터 부실"이므로 단일 에이전트의 Micro 능력 자체를 끌어올리는 것이 우선
- Step 1~10의 본질적 변화가 작음 (관측 150-dim, 행동 7-direction 고정. reward 항목만 누적)
- 단일 에이전트가 디버깅/평가/autoresearch 모두 단순
- Hierarchical의 전이학습 이점은 "Step 1~6 능력 보존" 한 가지뿐이었고, 이는 단일 에이전트 + 위 A/B/C/D 전략으로 대체 가능

### 2.4 Dense Reward 도입 (HRLP 영감)

Tan & Mu (2024) "Hierarchical Reinforcement Learning for Chip-Macro Placement" 의 dense reward 기법을 차용. 다만 별도 네트워크 (Hierarchical) 는 채택하지 않고 **Potential-Based Shaping (PBS) 형식의 dense reward만** 단일 에이전트 안에 통합.

자세한 reward 설계는 §16.7 참조.

### 2.5 Macro-action 도입 검토 (잠정)

HRLP의 options 정신 (temporal abstraction) 을 별도 네트워크 없이 단일 네트워크 안에서 macro-action 형태로 차용 가능. Phase 1 결과 (dense reward 단독 도입 후) 에 따라 도입 결정.

자세한 사항은 §99-Locked L-D3 참조.

---

## 참조 문서 지도 (docs/)

CLAUDE.md는 매 세션 상시 로드해야 하는 핵심(비전 / 아키텍처 / 전 Step 불변 규칙 / 작업 경로 원칙 / 활성 Lock)만 남긴다. 주제별 세부 spec은 아래 `docs/` 참조 문서에 있으며, 해당 주제로 작업할 때 로드한다.

| 문서 | 원 섹션 | 로드 시점 |
|------|---------|-----------|
| `@docs/pipe-engineering-spec.md` | §3~§8, §10 | 파이프 타입 / JIS 사이즈 / ASME 벤딩 / 중력관 / 밸브 / 서포트 / 입출력 JSON 스펙 구현 시 |
| `@docs/evaluation-spec.md` | §11 | 평가 시나리오(4분할/난이도/seed range), Procedural Generator, Step별 평가 metric 작업 시 |
| `@docs/transfer-learning-and-handoff.md` | §12.1~12.3, §12.5, §13 | Step 전환 전이학습, 회귀 감시 체계, Action Masking 누적표, 핸드오프 패키지 작업 시 |
| `@docs/autoresearch-ops.md` | §16 전체 | autoresearch 루프, baseline reward sweep, Dense Reward(PBS), Optuna/wandb 작업 시 |
| `@docs/ops-and-viz.md` | §14~§15 | 시각화 스택, 구동 환경(Colab 등) 세부사항 참조 시 |
| `@docs/deprecated-and-lock-history.md` | §17, 해제 완료 Lock 상세, §101 | 과거 의사결정 근거·이력 조사 시 |

> §12.4 (Face 6방향 제약) 와 §16.2.B (변경 절대 금지 항목) 는 전 Step에서 상시 지켜야 할 불변 규칙이라 본문에 남아 있다.

---

## 9. 좌표계 & 단위

| 항목 | 상세 내용 |
|------|-----------|
| 좌표계 | 오른손 좌표계 (Right-handed System) |
| X축 | 가로 (동-서) |
| Y축 | 세로 (남-북) |
| Z축 | 높이 (상-하) |
| 기본 단위 | mm |
| 각도 단위 | degree |
| 구배 표시 | 비율 (예: 1/100) |
| 비용 단위 | kg (물리적 중량) |

---

### 12.4 [Step 1~10 공통] 출발/목표 연결 방향 제약 (Face 6방향)

> **배관 공학 물리 원칙**: 파이프는 장비 노즐 또는 다른 파이프와 연결 시 반드시 6개 축 정렬 방향(+X/-X/+Y/-Y/+Z/-Z) 중 하나로 직각 접속한다. 이 원칙은 Step 1~10 전체에 예외 없이 적용된다.

| 제약 항목 | 적용 방식 | 유지 여부 |
|-----------|-----------|:---------:|
| **출발 방향** (`start_dir_idx`) | `action_masks()` Hard Constraint | **전 Step 유지** |
| **목표 도달 방향** (`goal_dir_idx`) | `_calc_reward()` Soft Constraint | **전 Step 유지** |
| **Face 방향 인덱스** | `0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z` | **전 Step 동일** |

```python
# 에피소드 reset() 공통 로직 (Step 1~10 동일)
start_dir_idx = rng.integers(0, 6)
goal_dir_idx  = rng.choice([i for i in range(6)
                             if i != start_dir_idx
                             and not np.array_equal(FACE_DIRS[i],
                                                    -FACE_DIRS[start_dir_idx])
                            ])
```

| 보상 항목 | 값 | 발동 조건 |
|---|---:|---|
| `direction_align_bonus` | **+5.0** | 첫 스텝이 `start_dir_idx` 일치 |
| `wrong_start_dir_penalty` | **-15.0** | 첫 스텝 불일치 (Fallback) |
| `wrong_goal_dir_penalty` | **-15.0** | 목표 도달 시 Face 불일치 |
| `wrong_goal_dir_penalty × 1.5` | **-22.5** | Edge/Vertex 진입 |

> ⚠️ `start_dir_idx`, `goal_dir_idx`는 에피소드마다 랜덤 배정.
> obs[91:94] (출발방향 단위벡터), obs[94:97] (목표방향 단위벡터).

> §12.1~12.3, §12.5 (전이학습 메커니즘, 회귀 감시, Action Masking 누적표) 는 `@docs/transfer-learning-and-handoff.md` 참조.

---

## 변경 절대 금지 항목 (§16.2.B 요약)

autoresearch를 포함한 모든 변형 실험에서 다음은 절대 변경 금지다 (전이학습 보장). 변형 가능 / 조건부 허용 항목 전체 목록과 baseline reward, Dense Reward(PBS) 등 autoresearch 운영 spec 전체는 `@docs/autoresearch-ops.md` §16 참조.

| 항목 | 이유 |
|------|------|
| Backbone hidden dim | Step 1~10 전체 256 (전 Step 동일) |
| Observation dim 구조 | 150-dim 고정 슬롯 할당 |
| Action space 정의 | 마스킹은 변경 가능, 기본 정의 불변 (※ macro-action 도입 시 L-D3 재해석) |
| Layer 수의 큰 변경 | 가중치 mapping 호환성 |

---

## 18. 작업 경로 원칙 (A안)

- GitHub이 코드 정본이다. 모든 코드 수정은 Claude Code 웹에서 브랜치 → PR → main 병합으로만 이루어진다.
- Colab은 학습 실행 전용이다. 드라이브 폴더에서 코드를 직접 수정하지 않는다.
- Colab 세션 시작 시 반드시 git pull 을 먼저 실행한다. pull 없이 학습을 시작하지 않는다.
- 예외적으로 Colab에서 수정이 불가피한 경우, 그 세션 안에 커밋·push하고 사유를 FAILURE_LOG에 남긴다.

---

## 99. Locked Items - Pending Resolution

> **🎓 2026-06-24 졸업 후 상태**: harness engineering 시작 차단 의미는 해제됨. 본 섹션은 **남은 4개 활성 Lock 의 자기 자연 시점 해제 가이드** 역할로 전환.
>
> 활성 Lock 4개 (L-A3', L-A4, L-C4, L-D3) 는 모두 Phase 1 학습 시작에 영향 없음. 각 Lock 의 해제 권장 시점은 항목별 안내 참조.
>
> ⚠️ **vibe coding 금지 원칙은 졸업 후에도 유지**. 새 결정이 필요한 경우 항상 Lock 해제 절차 (별도 논의 세션) 를 거친다.
>
> **⚠️ 2026-06-18 정리**: L-A5 해제 (§11.0 시나리오 spec 신설), L-D1 해제 (§12.2~12.3 채움). L-A6 신설 (시나리오 생성 알고리즘 spec). **L-D2 해제 (§16.7 Phase 1 적용 spec 확정)**. **L-A6 / L-C2 / L-16.3-w 해제 (§11.0.8 generator 알고리즘 / §16.5~16.6.7 autoresearch 운영 / §16.3.1~3 가중치 초기값). Phase 1 시작 조건 충족.**

### Lock 명명 규칙 (범례)

Lock ID는 `L-{카테고리}{일련번호}` 또는 `L-{spec 섹션}-{tag}` 형식이다. 카테고리 식별자는 다음과 같다:

| 접두사 | 의미 | 비고 |
|--------|------|------|
| `L-A{n}` | **A**ssessment — 평가 체계 (metric, 임계값, 시나리오) | 활성 |
| `L-B{n}` | **B**lock-level — Hierarchical / Macro 구조 | 2026-05-14 전면 폐기 |
| `L-C{n}` | **C**ontrol / Operations — autoresearch 운영, 좌표계, 인프라 | 활성 |
| `L-D{n}` | **D**ynamics — 단일 에이전트 + 전이학습 구조의 학습 동역학 결정 (2026-05-14 신규) | 활성 |
| `L-{§n.m}-{tag}` | spec 특정 섹션 직참조 (예: `L-16.3-w` = §16.3의 weight) | 알파벳 카테고리화 곤란한 항목 |

폐기된 카테고리(L-B)는 ID 안정성을 위해 재할당하지 않는다. 이력 추적(PROGRESS.md 의사결정 로그)에서 과거 L-B 결정을 참조하기 때문이다.

---

### L-A4: Step 4~6 hack 방지 hard constraint 세부 (Layer 1 정의)

각 step의 metric을 hack하는 방식을 미리 차단할 추가 hard constraint:

- Step 4: 분기 위치 분산도 (한 곳 몰림 방지) 필요?
- Step 5: 밸브 위치 분산도 필요?
- Step 6: 서포트 최소 개수 외 추가 제약?

> 2026-05-14: 의사결정 8의 Layer 1 (Hard Constraint) 의 구체 정의가 본 항목.

### L-A3': Step 1~3 평가 임계값 구체값

- Step 1: success_rate 임계값, length_ratio 임계값
- Step 2: distance_to_pareto 정의 및 임계값
- Step 3: length_ratio 임계값 (slope_violation은 0 hard로 결정)

> **2026-06-18 참고**: L-A5 해제로 시나리오 set 이 정해졌으므로 Phase 1 결과의 empirical CDF 측정 가능. Phase 1 후 후행 결정 권장.

### L-C4: grid 좌표계 통합 (Macro 폐기 후 단순화)

> **2026-05-14 단순화**: 이전의 "grid ↔ graph 좌표계 통합" 에서 graph 좌표계 (Macro의 visibility graph) 가 폐기되어 단순화됨.

- grid cell 좌표 ↔ mm 좌표 변환 정밀도
- 환경 reset / step 시 좌표 단위 일관성 검증

### L-D3: Macro-action 도입 결정 및 구체 설계 (신규, 2026-05-14)

> 도입 시점: Phase 1에서 dense reward (§16.7) 단독 도입 후 결과 평가. 천장 확인 시 본 항목 해제 세션 진행.

- 도입 여부 자체 결정 (Phase 1 결과 기반)
- 도입 결정 시:
  - 추가 macro-action 종류 ("직선 N칸" / "방향 유지 max" / 기타)
  - Termination condition
  - Action mask 처리
  - SKILL §3.1 "Action space 정의 불변" 재해석안
  - 전이학습 시 호환성 (Step 1 학습 정책이 Step 2~10에서 동일 macro-action 사용 가능해야)

---

> **해제 완료 Lock 상세** (L-A5, L-A6, L-C2, L-16.3-w, L-D1, L-D2) 와 **폐기된 Lock 목록** (L-B1~4, L-12.2, L-17.1) 은 `@docs/deprecated-and-lock-history.md` 참조.

---

## 100. Lock 해제 절차

각 Locked Item은 다음 절차로 해제한다:

```
1. 해당 항목에 대한 별도 논의 세션
2. 결론 도달 시 본 문서 업데이트:
   - L-XXX 섹션 삭제
   - 본문 해당 위치에 결정 사항 작성
3. Lock 해제 로그 (§101, `@docs/deprecated-and-lock-history.md`) 에 기록
4. 모든 Lock 해제 시 본 문서 → CLAUDE.md (확정판) 으로 rename
```

---

**문서 버전:** v1.4
**졸업일:** 2026-06-24
**마지막 갱신:** 2026-08-16 (CLAUDE.md 77KB → 핵심 규칙만 남기고 docs/ 6개 참조 문서로 분리, §18 작업 경로 원칙 A안 신설)
**이전 갱신:** 2026-06-28 v1.3 (§101 의사결정 27: §16.3.2 산술 오류 수정 + FAILURE_LOG entry), 2026-06-28 v1.2 (§101 의사결정 25/26), 2026-06-25 v1.1 (§101 의사결정 22/23/24), 2026-06-24 v1.0 (\_ing 시스템 졸업), 2026-06-18 v0.6 (3개 Lock 추가 해제), 2026-06-18 v0.5 (L-A5/L-D1/L-D2 해제), 2026-06-11 (§99 Lock 명명 규칙), 2026-05-14 (Hierarchical 폐기)

**졸업 후 spec 운영 원칙**:
1. spec 결정 변경 시 — Claude AI 세션에서 4개 파일 일괄 수정
2. 코드 구현 — Claude Code CLI 에서 본 spec 1:1 변환 (harness engineering)
3. 학습 결과 분석 → 새 의사결정 — Claude AI 세션
4. FAILURE_LOG entry 추가 시 — 실패 케이스를 수작업 seed pool 로 영구 등록 (§11.0.5, 상세는 `@docs/evaluation-spec.md`)
5. vibe coding 금지 원칙 유지 — 새 결정 필요 시 별도 논의 세션
6. SKILL.md 갱신 시 — 로컬 수정 → git commit → 서버 skill in-place 갱신 → CLI 검증 (의사결정 23, SKILL.md §0.5)
7. **CLAUDE.md 는 상시 로드 핵심만 유지** — 주제별 세부 spec 은 `docs/` 참조 문서에서 관리, 필요 시 `@docs/파일명.md` 로 로드 (2026-08-16 문서 분리)

**남은 활성 Lock 4개 — 자기 자연 시점 해제**:
- L-A3' (Phase 1 결과 후 후행 결정)
- L-A4 (Step 4 진입 시점)
- L-C4 (좌표계, implementation 시 unit test 위임 가능 — 사실상 자동 해제)
- L-D3 (Phase 1 결과 후 macro-action 도입 검토)
