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

## 3. 파이프 타입 분류 체계 (8개 타입)

3개 이진 속성의 조합으로 8개 타입이 정의된다:

| Type ID | 관종 | 분기 | 밸브 | 도입 Step |
|---------|------|------|------|-----------|
| T1 | 압력관 | 비분기 | 밸브무 | Step 1 |
| T2 | 압력관 | 비분기 | 밸브유 | Step 5 |
| T3 | 압력관 | 분기 | 밸브무 | Step 4 |
| T4 | 압력관 | 분기 | 밸브유 | Step 5 |
| T5 | 중력관 | 비분기 | 밸브무 | Step 3 |
| T6 | 중력관 | 비분기 | 밸브유 | Step 5 |
| T7 | 중력관 | 분기 | 밸브무 | Step 4 |
| T8 | 중력관 | 분기 | 밸브유 | Step 5 |

---

## 4. JIS 파이프 사이즈 규격

### 4.1 기본 사이즈 (JIS G 3452 / G 3454 기준)

| 호칭경 | 외경(OD) mm | 보온재 미포함 반경 mm | 보온재(50mm) 포함 유효 반경 mm |
|--------|------------|---------------------|------------------------------|
| 15A | 21.7 | 10.85 | 60.85 |
| 20A | 27.2 | 13.60 | 63.60 |
| 25A | 34.0 | 17.00 | 67.00 |
| 32A | 42.7 | 21.35 | 71.35 |
| 40A | 48.6 | 24.30 | 74.30 |
| 50A | 60.5 | 30.25 | 80.25 |
| 65A | 76.3 | 38.15 | 88.15 |
| 80A | 89.1 | 44.55 | 94.55 |
| 100A | 114.3 | 57.15 | 107.15 |
| 125A | 139.8 | 69.90 | 119.90 |
| 150A | 165.2 | 82.60 | 132.60 |
| 200A | 216.3 | 108.15 | 158.15 |
| 250A | 267.4 | 133.70 | 183.70 |
| 300A | 318.5 | 159.25 | 209.25 |
| 350A | 355.6 | 177.80 | 227.80 |
| 400A | 406.4 | 203.20 | 253.20 |
| 450A | 457.2 | 228.60 | 278.60 |
| 500A | 508.0 | 254.00 | 304.00 |

### 4.2 보온재 적용 규칙
- 보온재 두께: 50mm (전방위 균일)
- 유효 외경 = 파이프 외경 + (보온재 두께 × 2)
- 충돌 판정 시 유효 외경 기준으로 판정
- 보온재 유무는 파이프 속성 입력값으로 결정

### 4.3 사전 계산 및 다중 센서 융합 전략 (Pre-calculated Hybrid Sensing)

강화학습 환경의 연산량 폭발(OOM) 및 보상 희소성 문제를 근본적으로 해결하기 위해, 3D 그리드를 직접 합성곱(CNN)으로 처리하는 방식을 탈피하고 **사전 계산(Pre-calculation)**을 통한 **1D 벡터(MLP)** 구조로 관측 공간을 구성한다.

1. **메모리(RAM) 점유율 최적화 (총 목표: ~24MB 이하)**
   - **Occupancy Grid (장애물/파이프):** `bool` 타입 적용 (~12MB)
   - **SDF Field (간섭거리):** `float32` 대신 `uint8`로 양자화하여 저장 (~12MB)
   - **Gradient:** 별도 저장 없이 런타임에 인접 셀과의 차분으로 실시간 계산

2. **동적 장애물과 부분 EDT 업데이트 (Partial EDT)**
   - 에피소드 시작 시 정적 장애물에 대해 1회 전체 EDT 계산 (~1초)
   - 파이프 추가 등 동적 변화 발생 시, 해당 파이프의 AABB 주변만 로컬 부분 업데이트 (~0.01초)

3. **RL 에이전트 관측 (Observation): 150-dim 고정 (Zero-padding)**
   - 전이학습 호환성을 위해 **관측 벡터를 150차원으로 Step 1~10 고정**한다. 각 Step에서 미사용 슬롯은 `0.0`으로 채운다.
   - 입력값의 스케일 차이는 `VecNormalize`를 통한 자동 정규화로 대응
   - **Step 7 이후 (다중 파이프)**: 다른 파이프 정보는 obs[136:150] 예비 슬롯 활용 (구체 정의는 §99-Locked L-D1 후속)

   **Step별 활성 슬롯 할당:**

   | Step | 활성 dim | 누적 범위 | 신규 피처 |
   |------|:---:|---------|----------|
   | Step 1 | 98 | obs[0:98] | SDF센서, Raycast, 위치, 방향, 파이프속성 등 |
   | Step 2 | +5 = 103 | obs[98:103] | current_direction(3), straight_dist(1), min_bend_radius(1) |
   | Step 3 | +8 = 111 | obs[103:111] | is_gravity, slope, elevation 등 |
   | Step 4 | +10 = 121 | obs[111:121] | branch_info, target, phase 등 |
   | Step 5 | +8 = 129 | obs[121:129] | valve_info, accessibility 등 |
   | Step 6 | +7 = 136 | obs[129:136] | support_cost, wall/ceiling dist 등 |
   | 예비 | 14 | obs[136:150] | 향후 확장용 Zero-padding |

---

## 5. ASME B31.3 벤딩 표준 (Step 2 이후 적용)

### 5.1 최소 벤딩 반경

| 벤딩 타입 | 최소 반경 | 비고 |
|-----------|-----------|------|
| Long Radius Elbow | 1.5D | D = 호칭경 |
| Short Radius Elbow | 1.0D | 공간 제약 시 |
| Hot Bend | 5.0D | 현장 벤딩 |
| Cold Bend | 6.0D | 소구경 |

### 5.2 벤딩 각도
- 90° 엘보우 (표준)
- 45° 엘보우
- 기타 각도 (Step 후반)

### 5.3 벤딩 시 추가 공간
벤딩 구간에서는 직선 구간보다 더 큰 공간이 필요하며, 유효 외경 + 벤딩 반경을 고려한 점유 영역을 계산해야 한다.

---

## 6. 중력관 규격 (Step 3 이후 적용)

### 6.1 최소 구배

| 파이프 용도 | 최소 구배 | 비고 |
|------------|-----------|------|
| 배수관 (일반) | 1/100 (1%) | 가장 일반적 |
| 배수관 (대구경 200A+) | 1/200 (0.5%) | 대구경 허용 |
| 통기관 | 1/200 (0.5%) | 역구배 금지 |
| 우수관 | 1/100 (1%) | |
| 오배수 주관 | 1/50 ~ 1/100 | 상황에 따라 |

### 6.2 중력관 제약 조건
- 흐름 방향으로 지속적 하향 구배 유지
- 역구배(상향) 구간 절대 금지
- 수평 구간은 최소 구배 이상 유지
- 수직 낙차 구간은 허용
- 트랩 방지를 위한 연속 하향 보장

---

## 7. 밸브 & 인간 접근성 (Step 5 이후 적용)

### 7.1 밸브 설치 위치 제약

| 조건 | 값 | 비고 |
|------|-----|------|
| 최소 높이 (바닥면 기준) | 700mm | (※ 평가 metric에서 적용) |
| 최대 높이 (바닥면 기준) | 1500mm | (※ 평가 metric에서 적용) |
| 전면 여유 공간 | 1m × 1m × 1m | (※ 평가 metric에서 적용) |

> 본 spec의 평가 metric (§11.5) 은 700~1500mm 범위와 1m³ 공간을 사용한다.
> common_spec.md의 600~1800mm 범위는 reference로만 보존한다.

### 7.2 밸브 타입별 추가 공간
- Gate Valve: 핸들 회전 반경 고려
- Ball Valve: 레버 회전 반경 고려
- Check Valve: 추가 접근 불필요
- Control Valve: 계기 접근 공간 추가

---

## 8. 파이프 서포트 (Step 6 이후 적용)

### 8.1 서포트 간격 기준

| 호칭경 | 수평 최대 간격 (m) | 수직 최대 간격 (m) |
|--------|-------------------|-------------------|
| 15A~25A | 1.8 | 2.5 |
| 32A~50A | 2.5 | 3.0 |
| 65A~100A | 3.0 | 4.0 |
| 125A~200A | 4.0 | 5.0 |
| 250A~500A | 5.0 | 6.0 |

### 8.2 서포트 평가 (재정의)

> common_spec.md의 추상 비용 모델(벽 1.0, 행거 1.5 등)을 폐기하고
> **물리적 총 중량(kg)** 으로 평가한다 (§11.6 참조).

```
total_weight = pipe_weight + sum(support_weights)
  pipe_weight = JIS_table[size].weight_per_m × pipe_length
  support_weight = 서포트 종류별 실제 중량 (kg)
```

서포트 간격이 §8.1 max를 초과하면 hard constraint 위반 → 실패 처리.

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

## 10. 입출력 JSON 형식

### 10.1 입력 (JSON)

```json
{
  "space": {
    "width": 20000, "depth": 15000, "height": 5000,
    "unit": "mm"
  },
  "obstacles": [
    {"id": "OBS_001", "type": "column", "position": [x,y,z], "size": [w,d,h]}
  ],
  "pipes": [
    {
      "id": "P001",
      "type_id": "T1",
      "gravity_pipe": false,
      "has_branch": false,
      "has_valve": false,
      "nominal_size": "100A",
      "insulation": true,
      "insulation_thickness": 50,
      "start": [x1, y1, z1],
      "end": [x2, y2, z2],
      "branch_points": [],
      "valve_positions": [],
      "min_slope": null
    }
  ]
}
```

### 10.2 출력 (JSON)

```json
{
  "routes": [
    {
      "pipe_id": "P001",
      "waypoints": [[x,y,z], ...],
      "segments": [
        {"type": "straight", "start": [x,y,z], "end": [x,y,z], "length": 1500},
        {"type": "elbow_90", "center": [x,y,z], "radius": 150, "plane": "XZ"}
      ],
      "fittings": [
        {"type": "elbow_90_LR", "position": [x,y,z]},
        {"type": "tee", "position": [x,y,z]},
        {"type": "gate_valve", "position": [x,y,z], "accessible": true}
      ],
      "supports": [
        {"type": "wall_bracket", "position": [x,y,z], "weight_kg": 2.5}
      ],
      "metrics": {
        "total_length": 12500,
        "num_bends": 3,
        "slope_maintained": true,
        "valve_accessible": true,
        "total_weight_kg": 145.2
      }
    }
  ],
  "global_metrics": {
    "total_pipes": 1,
    "success_count": 1,
    "kpi": 100.0,
    "total_system_weight_kg": 145.2,
    "computation_time_sec": 5.2
  }
}
```

---

## 11. 평가 체계 (재정의)

> common_spec.md 9장(KPI)을 폐기하고 본 장으로 대체한다.

### 11.0 평가 시나리오 구성 (2026-06-18 신규, L-A5 해제)

본 §11.0은 §11.1~11.6의 모든 평가가 어떤 시나리오 set 위에서 수행되는지를 정의한다. 시나리오 set이 흔들리면 모든 평가가 무의미해지므로 본 항목이 평가 체계의 토대다.

#### 11.0.1 시나리오 set 4분할

각 Step마다 다음 4가지 용도의 시나리오를 운영한다. 용도 간 seed range는 **절대 겹치지 않는다** (데이터 누수 방지).

| 용도 | 개수 (Step별) | 생성 방식 | 운영 |
|------|--------------|----------|------|
| **학습용** | 무한대 (generator) | Procedural, 매 episode 새 seed | 학습 중 끝없이 새 시나리오 |
| **autoresearch screening용** | **75개 고정** | Procedural + 고정 seed list | Stage 1 빠른 비교 (짧은 학습 후 평가) |
| **회귀 검증용** | **150개 고정** | Procedural (초기) + 수작업 (시간 경과 시 누적) | Step N≥2 학습 중 주기적 실행 (§12.3) |
| **최종 평가용** | **500개 고정** | Procedural (초기) + 수작업 (시간 경과 시 누적) | Step 완료 시 1회. A* benchmark / Layer 1/2 평가 |

#### 11.0.2 난이도 분포 (Easy / Medium / Hard = 3 : 5 : 2)

각 용도의 시나리오 set 안에서 난이도 분포는 다음과 같이 고정한다.

| 용도 | Easy 비율 | Medium 비율 | Hard 비율 |
|------|----------|-------------|----------|
| Screening 75개 | 22~23개 (30%) | 37~38개 (50%) | 15개 (20%) |
| 회귀 150개 | 45개 (30%) | 75개 (50%) | 30개 (20%) |
| 최종 평가 500개 | 150개 (30%) | 250개 (50%) | 100개 (20%) |

학습용 procedural generator는 별도. 학습 초기에 Easy 비율을 높게 (예: 60%) 시작해 후반으로 갈수록 Hard 비율을 높이는 micro-curriculum 가능 (Step 내부 curriculum).

#### 11.0.3 Step별 난이도 정의 (난이도 축 차별화)

각 Step의 학습 목표가 다르므로 난이도 축도 Step별로 다르게 정의한다.

**Step 1 — 장애물 회피**

| 축 | Easy | Medium | Hard |
|----|------|--------|------|
| 장애물 밀도 (cell 점유 %) | 10% | 20% | 35% |
| Start-Goal Manhattan 거리 | 5~10 cell | 10~20 cell | 20~30 cell |
| 직선거리 / Manhattan 거리 ratio | 0.9~1.0 (직선 가능) | 0.6~0.9 | 0.3~0.6 (강한 우회) |

**Step 2 — Bend 최소화 추가**

| 축 | Easy | Medium | Hard |
|----|------|--------|------|
| Step 1 축 상속 | Easy | Medium | Hard |
| Corridor 강제 정도 | 자유 공간 많음 | 일부 corridor | 강한 corridor |
| 최적 경로의 bend 수 (A* 기준) | 0~2 | 3~5 | 6+ |

**Step 3 — 중력관**

| 축 | Easy | Medium | Hard |
|----|------|--------|------|
| Step 1~2 축 상속 | Easy | Medium | Hard |
| 중력관 비율 (시나리오 내) | 0% (=Step 2) | 30~50% | 70~100% |
| Z 차이 / Horizontal 거리 (요구 슬로프 압박) | ≤ 1/200 | ~1/100 | ~1/50 (한계) |
| 장애물의 슬로프 방해 | 슬로프 자유 | 부분 방해 | 강제 우회 |

**Step 4 — 분기**

| 축 | Easy | Medium | Hard |
|----|------|--------|------|
| Step 1~3 축 상속 | Easy | Medium | Hard |
| 분기점 수 | 1개 | 2~3개 | 4+개 |
| 목적지 수 | 2개 | 3~4개 | 5+개 |
| 목적지 공간 분산도 | 가까움 | 중간 | 멀리 분산 |

**Step 5 — 밸브**

| 축 | Easy | Medium | Hard |
|----|------|--------|------|
| Step 1~4 축 상속 | Easy | Medium | Hard |
| 밸브 개수 | 1~2개 | 3~4개 | 5+개 |
| 밸브 높이 제약 (700~1500mm) 충돌 정도 | 자연 경로가 제약 내 | 일부만 가능 | 거의 모든 경로가 위반 (우회 필요) |
| 밸브 간 최소 간격 제약 | 느슨 | 중간 | 빡빡 |

**Step 6 — 서포트**

| 축 | Easy | Medium | Hard |
|----|------|--------|------|
| Step 1~5 축 상속 | Easy | Medium | Hard |
| 파이프 총 길이 | 짧음 (서포트 1~2개) | 중간 (3~5개) | 김 (6+개) |
| 서포트 가능 면 비율 | 천장/벽 충분 | 일부만 | 제한적 |
| 파이프 무게 (지름·재질) | 가벼움 | 중간 | 무거움 (간격 빡빡) |

**Step 7~10 — 다중 파이프** (잠정 골격, L-D1 후속 보강 필요)

| 축 | Easy | Medium | Hard |
|----|------|--------|------|
| Step 1~6 축 상속 | Easy | Medium | Hard |
| 파이프 수 | 2개 | 3~4개 | 5+개 |
| 파이프 간 공간 경합도 | 자유 (deadlock 불가) | 일부 경합 | 강한 경합 (deadlock 위험) |
| 우선순위 분리도 | 명확 | 모호 | 동등 (양보 필요) |

#### 11.0.4 Seed Range 분리 체계

각 Step별로 100,000 단위 range를 할당하고, 그 안에서 용도별로 sub-range를 분리한다.

```
Step N의 seed range:
  Screening:    N*100000 + 10000 ~ N*100000 + 19999  (10000개 슬롯 중 75개 사용)
  Regression:   N*100000 + 20000 ~ N*100000 + 29999  (10000개 슬롯 중 150개 사용)
  Final Eval:   N*100000 + 30000 ~ N*100000 + 39999  (10000개 슬롯 중 500개 사용)
  학습용:        N*100000 + 50000 ~ N*100000 + 99999  (50000개 슬롯, 학습 중 순차 소비)

예) Step 1:
  Screening:    110000~119999 중 75개
  Regression:   120000~129999 중 150개
  Final Eval:   130000~139999 중 500개
  학습용:        150000~199999 중 매 episode 순차 사용

예) Step 6:
  Screening:    610000~619999 중 75개
  Regression:   620000~629999 중 150개
  Final Eval:   630000~639999 중 500개
  학습용:        650000~699999 중 순차 사용
```

#### 11.0.5 생성 방식 — Procedural 100% 시작, 수작업 점진 추가

초기에는 모든 시나리오를 procedural generator로 100% 생성한다. **수작업 시나리오 비율은 0%에서 시작**하고, 다음 메커니즘으로 시간이 지나며 누적된다:

```
[수작업 seed 추가 트리거]
FAILURE_LOG.md에 학습/평가 실패 entry가 추가될 때마다:
  1. 해당 실패 케이스의 grid / start / goal / 추가 메타데이터를 YAML로 저장
  2. 해당 Step의 수작업 seed pool에 영구 등록
  3. 다음 학습/평가부터 회귀 / 최종 평가 set에 자동 포함
  4. (선택) procedural 생성 시 비슷한 패턴이 더 많이 나오도록 generator parameter 조정 검토

[비율 운영]
- 회귀 검증 150개: procedural 100% → 시간 경과 시 수작업 비율 자연 증가 (최대 30% 권장)
- 최종 평가 500개: procedural 100% → 시간 경과 시 수작업 비율 자연 증가 (최대 30% 권장)
- 30% 초과 시 procedural과 수작업 비율을 의식적으로 관리 (특정 패턴 과대표집 방지)
```

이 메커니즘에 따라 **FAILURE_LOG는 단순 archive가 아니라 회귀 set의 동적 source**가 된다. 실패한 케이스가 영구 평가 대상으로 승격되는 구조.

#### 11.0.6 Seed 관리 인프라

- **저장 형식**: YAML, Git으로 추적 (`scenarios/stepN/{screening,regression,final_eval}_seeds.yaml`)
- **RNG**: `numpy.random.default_rng(seed)` (PCG64) 만 사용. legacy API (`numpy.random.seed`, `random.seed`) 금지
- **Generator 버전 관리**: procedural generator 알고리즘에 version 박음. 알고리즘 수정 시 version 증가 + seed 파일에 generator_version 기록 + 기존 seed로 생성된 시나리오 hash 보관 (호환성 검증)
- **결정성 환경**: 평가 시 `set_random_seed(seed)` (SB3) + `torch.use_deterministic_algorithms(True)` (옵션, 학습 속도 손실 있음)
- **YAML 파일 예시**:

```yaml
# scenarios/step1/regression_seeds.yaml
generator_version: "v1.0.0"
created: "2026-06-18"
seed_range: [120000, 129999]
seeds:
  easy:   [120000, 120001, ..., 120044]  # 45개
  medium: [120045, 120046, ..., 120119]  # 75개
  hard:   [120120, 120121, ..., 120149]  # 30개
manual_additions: []  # 향후 FAILURE_LOG entry로부터 추가
```

#### 11.0.7 회귀 검증 통과 임계값 (Stage 1 vs Stage 2 분리)

§12.3의 회귀 감시가 사용할 임계값. **임시값**으로 시작하고 Phase 1 결과 분포 보고 v0.4 spec에서 조정.

| 지표 | Stage 1 (screening) | Stage 2 (정상 학습) |
|------|---------------------|----------------------|
| success_rate (절대 손실) | −10%p 이내 | **−5%p 이내** |
| length_ratio (증가) | +20% 이내 | **+10% 이내** |
| bend_count_ratio (증가, Step 2+) | +30% 이내 | **+15% 이내** |
| Layer 1 hard constraint 통과율 (Step 3+) | 95% 이상 | **100% 절대** |
| BWT_k (Backward Transfer) | ≥ −0.10 | **≥ −0.05** |
| Forgetting measure F_k | ≤ 0.20 | **≤ 0.10** |

지표 정의는 SKILL.md §3.7 참조. 측정 주기와 트리거 escalation은 §12.3에서 정의.

#### 11.0.8 Procedural Scenario Generator 알고리즘 (2026-06-18 신규, L-A6 해제)

본 sub-section 은 §11.0.5 의 procedural generator 의 알고리즘 자체를 정의한다. Generator version v1.0.0 의 spec.

##### 11.0.8.1 장애물 배치 알고리즘 — Poisson Disk Sampling + Cuboid 합성

```python
def generate_obstacles(seed, grid_size=(30, 30, 30), difficulty="medium"):
    """선박 배관 환경 모사. Poisson disk 로 중심점 생성 후 cuboid 합성."""
    rng = np.random.default_rng(seed)
    grid = np.zeros(grid_size, dtype=bool)
    
    # 난이도별 obstacle volume budget
    density = {"easy": 0.10, "medium": 0.20, "hard": 0.35}[difficulty]
    budget = int(grid.size * density)
    
    # 도메인 형상 비율 (선박 배관 환경 모사)
    types = {
        "tank":       {"size_range": (5, 12), "weight": 0.4},  # 대형 탱크/박스
        "structural": {"size_range": (2, 4),  "weight": 0.3},  # 구조재 (긴 막대)
        "small_eq":   {"size_range": (2, 5),  "weight": 0.3},  # 소형 장비
    }
    
    # Poisson disk sampling 으로 중심점 생성 (장애물 간 최소 거리 보장)
    centers = poisson_disk_sample(grid_size, min_dist=3, rng=rng)
    
    # 각 중심점에서 type 선택 후 cuboid 배치
    used = 0
    for center in centers:
        if used >= budget: break
        obs_type = rng.choice(list(types), p=[t["weight"] for t in types.values()])
        size = rng.integers(*types[obs_type]["size_range"], size=3)
        place_cuboid(grid, center, size)
        used += np.prod(size)
    
    return grid
```

**Poisson disk 채택 근거**: 장애물 간 최소 거리 보장 → 항상 RL 이 통과 가능한 좁은 corridor 존재 → "통과 불가능한 시나리오" 자동 방지.

**기각된 대안**:
- Uniform random voxel sampling: 단일 voxel 흩어짐, 비현실적
- Perlin noise: 선박 배관 환경의 angular 형상과 부정합
- Voronoi tessellation: cell 경계 모호, 6-direction action 과 부정합

##### 11.0.8.2 Start-Goal Sampling 알고리즘 — Rejection Sampling + A* Reachable 검증

```python
def sample_start_goal(grid, seed, difficulty="medium"):
    """난이도별 거리/ratio 제약 + reachable 검증."""
    rng = np.random.default_rng(seed + 1)  # generator seed 와 분리
    
    free_cells = np.argwhere(~grid)
    
    dist_range = {
        "easy":   (5, 10),
        "medium": (10, 20),
        "hard":   (20, 30),
    }[difficulty]
    
    ratio_range = {
        "easy":   (0.9, 1.0),
        "medium": (0.6, 0.9),
        "hard":   (0.3, 0.6),
    }[difficulty]
    
    # rejection sampling
    for _ in range(1000):
        start = free_cells[rng.integers(len(free_cells))]
        goal  = free_cells[rng.integers(len(free_cells))]
        
        d_manhattan = np.abs(goal - start).sum()
        d_euclidean = np.linalg.norm(goal - start)
        ratio       = d_euclidean / d_manhattan if d_manhattan > 0 else 1.0
        
        if dist_range[0] <= d_manhattan <= dist_range[1]:
            if ratio_range[0] <= ratio <= ratio_range[1]:
                # A* reachable 검증 (학습 신호 아님, 시나리오 검증 전용)
                if astar_reachable(grid, start, goal):
                    return start, goal
    
    raise ScenarioGenerationFailure(seed=seed, difficulty=difficulty)
```

**중요**: 본 A* 호출은 **시나리오 검증 전용**이며 **학습 신호로 사용되지 않음**. §11.1 평가 철학 ("A\* = 평가 도구") 과 충돌하지 않음. Generator 가 풀이 불가능한 시나리오를 만들지 않도록 하는 안전장치.

##### 11.0.8.3 난이도 파라미터 매핑 함수

§11.0.3 의 추상 정의 (Easy/Medium/Hard) 를 generator 입력값으로 변환:

```python
DIFFICULTY_PARAMS = {
    "step1": {
        "easy":   {"obstacle_density": 0.10, "dist_range": (5, 10),  "ratio_range": (0.9, 1.0)},
        "medium": {"obstacle_density": 0.20, "dist_range": (10, 20), "ratio_range": (0.6, 0.9)},
        "hard":   {"obstacle_density": 0.35, "dist_range": (20, 30), "ratio_range": (0.3, 0.6)},
    },
    "step2": {
        # Step 1 상속 + corridor 강제 정도 + A* bend count
        "easy":   {"_inherits": "step1.easy",   "min_bend_count": 0, "corridor_factor": 0.0},
        "medium": {"_inherits": "step1.medium", "min_bend_count": 3, "corridor_factor": 0.3},
        "hard":   {"_inherits": "step1.hard",   "min_bend_count": 6, "corridor_factor": 0.7},
    },
    # Step 3~6 동일 패턴 (각 Step 진입 직전 구체화)
}
```

`_inherits` 키로 이전 Step 상속 표현. 명시적이고 spec 변경 추적 용이.

##### 11.0.8.4 Step 4~6 sub-algorithm (개요만, 각 Step 진입 직전 구체화)

```
Step 4 (분기):
  - 분기점 후보 선택: free space 중 connectivity 높은 cell
  - 목적지 분산: centroid distance 제약
  - tee/elbow fitting 가능성 사전 검증

Step 5 (밸브):
  - 밸브 위치 후보: 높이 700~1500mm 범위 cell
  - 밸브 간 최소 간격: spec 별도 결정
  - 밸브 clearance volume 확보

Step 6 (서포트):
  - 서포트 가능 면: 천장/벽 인접 cell 식별
  - 파이프 길이 → 필요 서포트 수 함수
  - 무게 분포 시뮬레이션 (재질·지름 기반)
```

각 sub-algorithm 은 **해당 Step 진입 직전 별도 세션에서 구체화**. Phase 1 진입 자체에는 영향 없음.

##### 11.0.8.5 Step 7~10 sub-algorithm (Step 7 진입 시점 별도 세션)

```
Step 7~10: Step 6 시나리오에 N개 파이프 routing 요구 추가
  - N (파이프 수): 난이도별 2 / 3-4 / 5+
  - 파이프 간 공간 경합도: free space 중 동시 routing 가능 cell 비율
  - 우선순위 분리도: 파이프별 priority weight 분포
```

##### 11.0.8.6 Generator Version 관리

```
[Version Bump 규칙 — semver]
v1.0.0 → v1.0.1 : 버그 수정, 알고리즘 동작 동일
v1.0.0 → v1.1.0 : 새 sub-algorithm 추가 (예: Step 4 분기점)
                  기존 Step 1~3 시나리오는 동일하게 재현 가능
v1.0.0 → v2.0.0 : 기존 알고리즘 변경 (예: Poisson disk → Perlin)
                  기존 seed 로 생성된 시나리오 재현 안 됨

[관리 의무]
1. version 증가 시 git tag (예: generator-v1.1.0)
2. seed 파일에 generator_version 기록 (§11.0.6)
3. major version 증가 시:
   - 기존 시나리오 hash 보관 (검증 후 폐기 여부 결정)
   - 회귀 / 최종 평가 set 재생성 또는 호환성 검증 의무
   - PROGRESS.md 의사결정 entry 의무
```

### 11.1 평가 철학

- 숙련공 시연 = 정답 채택 (X)
- **물리적 측정 가능 metric으로 평가 (O)**
- Step 1~3: A* benchmark 활용
- Step 4~6: 물리 metric 활용
- Step 7~10: 다중 파이프 metric 활용 (세부 §99-Locked)

### 11.2 Step 1 평가

```
benchmark = A*_basic(cost = length)
metrics = {
    'success_rate': 도착 성공률,
    'length_ratio': L_rl / L_astar,
    'collision_count': 충돌 횟수
}

채택 조건:
  success_rate ≥ [임계값] AND length_ratio ≤ [임계값]
  ※ 임계값 §99-Locked
```

### 11.3 Step 2 평가 (Pareto front)

```
benchmarks = {
    'length_only': A*(cost = length),
    'bend_minimal': A*(cost = length, hard: bend ≤ N),
    'balanced_set': [A*(cost = length + w × bend) for w in sweep]
}
metrics = {
    'success_rate',
    'length_ratio',
    'bend_count_ratio',
    'distance_to_pareto_front',
    'asme_compliance_rate': 모든 벤딩의 ASME B31.3 준수율
}

채택 조건:
  ASME 100% 준수 (hard)
  AND distance_to_pareto ≤ [임계값]
  ※ 임계값 §99-Locked
```

### 11.4 Step 3 평가

```
benchmark = A*_gravity(hard constraint: Z 상승 금지 + min slope)
metrics = {
    'success_rate',
    'length_ratio',
    'slope_violation_count': 0 (hard)
}

채택 조건:
  slope_violation = 0 (hard)
  AND length_ratio ≤ [임계값]
  ※ 임계값 §99-Locked
```

### 11.5 Step 4~6 평가 (물리 metric, 2-Layer 분리)

> **2026-05-14 보강**: 의사결정 8 - Step 4 이후는 절대 최적의 정의가 수학적으로 불가능. 평가를 두 층으로 분리.

```
[Layer 1. Hard Constraint 충족 여부 - Binary]
  통과/실패의 binary 판정. autoresearch screening의 1차 필터.

[Layer 2. 정량적 Quality - Relative]
  - 물리 metric 절대값 (kg, mm 등)
  - 가능한 경우 Multi-cost A* 대비 비율 (Lower bound estimation)
  - 변형 간 ranking (autoresearch가 자연스럽게 산출)
  
  "절대 최적"이라는 개념을 명시적으로 포기.
  Autoresearch의 변형 간 상대 비교 + 시간에 따른 단조 개선이 종결 기준.
```

#### Step 4 (분기)

```
[Layer 1 Hard Constraints]
  - branch_existence: 분기가 제대로 형성됐는가
  - branch_angle_validity: 분기 각도가 90° 또는 45°
  - branch_on_straight: 분기점이 직선 구간에 있는가

[Layer 2 Quality Metric]
  - main_plus_branch_total_length: 전체 길이 (minimize)
  - (가능 시) Multi-cost A* 대비 비율
```

#### Step 5 (밸브)

```
[Layer 1 Hard Constraints]
  - valve_existence: 밸브 설치 여부
  - valve_height_compliance: 700~1500mm 범위 (100%)
  - valve_clearance_compliance: 전면 1m³ 공간 확보 (100%)

[Layer 2 Quality Metric]
  - total_length: 전체 길이 (minimize)
```

#### Step 6 (서포트)

```
[Layer 1 Hard Constraints]
  - support_interval_compliance: §8.1 max 간격 준수
  - support_minimum_count: 파이프 길이 대비 최소 서포트 수 (L-A4)

[Layer 2 Quality Metric]
  - total_system_weight_kg: pipe_weight + support_weight (minimize)
```

### 11.6 Step 7~10 평가

> ⚠️ **§99-Locked**: 다중 파이프 평가 metric의 구체 정의는 미결.
> 골격: 개별 파이프 metric × N + 파이프 간 상호작용 metric.

---

## 12. 모델 상속 및 전이학습 전략

> **⚠️ 2026-05-14 변경**: §12.2 (Step 6→7 전환), §12.3 (Step 7~10 Macro 신규 정의) 폐기됨. Hierarchical 폐기로 인한 부수 변경. 자세한 근거는 PROGRESS.md Session 2026-05-14 참조.

### 12.1 Step 1~10 단일 에이전트 (전 Step 동일)

```
Backbone Hidden Dim: 256 (전 Step 동일, 변경 절대 금지)
Architecture: MLP Fusion
Projection: 불필요
Observation: 150-dim Zero-padding (전 Step 동일)
Action: MaskablePPO + action_masks()
        (잠정 macro-action 도입 시 L-D3 참조)

전이학습 규칙:
  - 하위 Step 모델 로드 실패 시 학습 시작 금지
  - state_dict key naming 통일 → strict=True 로드 가능
  - normalizer.json 정의값 동일 참조
```

### 12.2 Step 간 전이학습 메커니즘 (2026-06-18 L-D1 해제, A+B 혼합 채택)

#### 12.2.1 채택된 전략 — 후보 A (순차 fine-tuning) + 후보 B (mixed curriculum) 혼합

후보 C (EWC) 와 후보 D (LoRA) 는 다음 근거로 비채택:

- **C (EWC)**: RL의 sample inefficiency 와 EWC의 Fisher 정보 추정이 잘 안 맞음. PPO의 on-policy 특성과도 충돌.
- **D (LoRA)**: 본 프로젝트 backbone hidden_dim=256 이 LoRA 효과 보기엔 작음. LoRA 는 대형 모델에서 빛남.

#### 12.2.2 학습 데이터 mix 비율 (Warm-up + 정상)

```
Step N 학습 시 (N ≥ 2):
  1. Step N-1 정책 가중치 θ_{N-1} 로 초기화 (fine-tuning, 후보 A)
  2. 학습률은 신규 학습의 50% 로 시작 (fine-tuning 표준 관행)
  3. Warm-up phase (학습 초기 100K timestep):
       50%  Step N 신규 시나리오 (procedural generator)
       40%  Step 1 ~ Step N-1 시나리오 (mixed curriculum, 후보 B)
       10%  FAILURE_LOG 누적 실패 케이스 (regression hardening)
  4. Warm-up 후 정상 phase:
       60%  Step N 신규 시나리오
       30%  Step 1 ~ Step N-1 시나리오
       10%  FAILURE_LOG 누적 실패 케이스
```

비율은 continual RL 문헌의 표준 관행을 따른 잠정값. Phase 1 후 BWT 데이터 보고 조정 가능 (autoresearch sweep 대상에 포함).

#### 12.2.3 비율 변경 시 안전 조건

- Step N-1 시나리오 비율을 30% 미만으로 낮추는 변경은 회귀 감시 임계값 (§11.0.7) 위반 위험 → autoresearch sweep 대상에서 제외 (FORBIDDEN)
- FAILURE_LOG 비율은 케이스 수에 따라 자동 조정 (등록된 케이스가 적으면 10% 채우기 위해 동일 케이스 반복 sampling)

### 12.3 Step 간 회귀 감시 체계 (2026-06-18 L-D1 해제, 의사결정 13 구체화)

> **2026-05-14 변경**: 이전의 "Macro 신규 도입 + Micro freeze + 3-stage 전환 프로토콜" 폐기.
> **2026-06-18 보강**: 회귀 감시 임계값 / 측정 주기 / 트리거 체계 구체화.

#### 12.3.1 측정 주기 (wandb logging과 정합)

```
MaskablePPO n_steps=2048 가정.
1 rollout ≒ 2048 timestep.

회귀 감시 콜백:
  - 매 25 rollout (≒ 50K timestep) 마다 실행
  - Step 1 ~ Step N-1 의 회귀 시나리오 150개 모두 평가
  - 결과를 wandb 에 학습 reward 곡선과 같은 panel 에 logging
  - Step N 총 학습량 2M timestep 가정 시 = 40번 측정 (충분한 해상도)

Stage 1 screening (autoresearch 1차):
  - 짧은 학습 (500K timestep 가정) 동안 10번 측정 (매 50K)
  - Stage 1 임계값 (§11.0.7 Stage 1 컬럼) 적용
```

#### 12.3.2 트리거 escalation (3단계 + 즉시 중단 case)

```
[Level 0 — 즉시 중단]
  조건: Layer 1 hard constraint 5% 이상 위반 (예: slope_violation, ASME compliance)
  대응:
    - 학습 즉시 중단
    - 이전 체크포인트 복귀
    - 사용자 알림 (Telegram bot)
    - spec 재검토 세션 권장

[Level 1 — 경고 (1번째 위반)]
  조건: Stage 2 임계값 1개 이상 soft 위반
  대응:
    - wandb 에 alert tag 기록
    - 학습 계속, 추가 모니터링

[Level 2 — 자동 조정 (2번째 연속 위반)]
  조건: 다음 회귀 측정에서도 동일 또는 다른 임계값 위반
  대응:
    - Mixed curriculum 비율 상향: 30% → 45%
    - 학습률 50% 감소
    - 학습 계속

[Level 3 — 학습 중단 (3번째 연속 위반)]
  조건: Level 2 조정 후에도 다음 측정에서 위반
  대응:
    - 학습 즉시 중단
    - 이전 체크포인트 복귀
    - 사용자 알림
    - Lock 해제 세션 또는 spec 재검토 필요
```

#### 12.3.3 회귀 지표 (3종 동시 기록, SKILL §3.7과 정합)

```
지표 1 — 각 과거 step 의 평가 metric 직접 측정 (Step 별 success_rate, length_ratio, ...)
지표 2 — Backward Transfer (BWT):  BWT_k = perf_k(end_of_stepN) - perf_k(end_of_stepK)
지표 3 — Forgetting measure:        F_k = max_{t ≤ now} perf_k(t) - perf_k(now)
```

임계값은 §11.0.7 표 참조.

#### 12.3.4 Step 6 → 7 전환 (다중 파이프 진입)

```
Step 6 정책 → Step 7 정책 초기화 (§12.2 전이학습 전략 적용)
Observation space: 150-dim 유지 (다른 파이프 정보를 obs[136:150] 활용)
Network: 동일 단일 에이전트 (별도 Macro 없음)
회귀 검증: §12.3.1~12.3.3 그대로 적용 (Step 1~6 회귀 시나리오 = 150개 × 6 Step = 900개 전체 평가)
```


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

### 12.5 [전 Step 공통] Action Masking + MaskablePPO

| Step | 마스킹 규칙 (누적) |
|------|-------------------|
| Step 1 | 충돌 + 경계 초과 + 자기경로 역주행 + 출발방향 강제 |
| Step 2 | + 벤딩 반경 위반 차단 |
| Step 3 | + 중력관 Z 상승 소프트 마스킹 |
| Step 4 | + 분기점 전용 행동 제약 |
| Step 5 | + 밸브 접근성 확보 불가 방향 차단 |
| Step 6 | + 서포트 비용 과다 경로 소프트 마스킹 |
| Step 7~10 | + 다른 파이프 영역 침입 마스킹 (구체 정의는 L-D1 후속) |

---

## 13. 핸드오프 패키지 (재정의)

> **⚠️ 2026-05-14 변경**: §13.2 (Step 6→7 전환 핸드오프 확장 4파일), §13.3 (Macro 핸드오프) 폐기. Hierarchical 폐기로 인한 부수 변경.

### 13.1 Step 핸드오프 (Step N → N+1, N ∈ [1,9])

> **★ Step 별 폴더 분리 의무 (의사결정 25, 2026-06-28):**
> 모든 핸드오프 파일은 `base_dir/step{N}/` 하위에 저장된다.
> `save_handoff(step_n=N, save_dir=base_dir)` → `base_dir/step{N}/` 자동 생성.
> Step 1 → 2 전이 시: `load_handoff(step_n=1, save_dir=base_dir)` → `base_dir/step1/` 탐색.
> **Step 간 자산 독립성 보장: Step 2 저장이 Step 1 자산을 덮어쓰지 않는다.**

```
base_dir/
  step1/                  ← Step 1 학습 완료 후 생성
    best_model.zip
    best_model_meta.json
    step1_regression_report.json
    step1_wandb_run_url.txt
    step1_env_config.json
    step1_reward_config.json
    step1_kpi_report.json
    step1_normalizer.json       (VecNormalize 저장 시)
    step1_action_space_map.json
    step1_training_log.csv
    step1_validation_set.json
  step2/                  ← Step 2 학습 완료 후 생성
    ...
```

전 Step 공통 핸드오프 (단일 에이전트 구조이므로 Step 6→7도 동일):

| 파일명 | 역할 |
|-------|------|
| `best_model.zip` | 단일 에이전트 가중치 (MaskablePPO.save) |
| `best_model_meta.json` | 아키텍처 정의 (HandoffConfig) |
| `stepN_normalizer.json` | 정규화 파라미터 (VecNormalize) |
| `stepN_kpi_report.json` | 평가 metric 결과 (Step 1~3: A* 비교, Step 4~6: Layer 1/2, Step 7~10: 다중 파이프) |
| `stepN_training_log.csv` | 학습 이력 |
| `stepN_action_space_map.json` | 행동 공간 |
| `stepN_reward_config.json` | 채택된 reward 설정 (baseline + dense PBS) |
| `stepN_env_config.json` | 환경 설정 |
| `stepN_validation_set.json` | 평가 시나리오 + benchmark 결과 |
| `stepN_regression_report.json` | Step 1~N-1 회귀 검증 결과 (Step 2 이후) |
| `stepN_wandb_run_url.txt` | 본 학습의 wandb run URL (재현/분석용) |

### 13.2 내부 텐서 규격
- **Input Dimension**: 150-dim 고정 (Step 1~10 전체)
- **Normalization**: `normalizer.json` 정의값 동일 참조
- **Weight Key Mapping**: `state_dict` key naming 통일 → `strict=True` 로드

---

## 14. 시각화 및 평가 스택

전체 시스템의 학습 평가 및 디버깅 시각화는 **Matplotlib + IPython HTML** 방식 사용.

- **렌더링**: Matplotlib(mpl_toolkits.mplot3d)으로 72프레임 사전 렌더링 → base64 → `IPython.display.HTML`로 Colab 셀에 출력
- **필수 형태**: 파이프 중심선(Centerline) + 두께 반영 볼륨(Cylinder) 동시 렌더링
- **뷰어 기능**: 에피소드 선택, 3D 회전, Plan/Section/Profile 2D 탭, 재생/정지/속도/줌
- **사용법:**
  ```python
  from visualization.visualization import run_visualization
  run_visualization(model=model, phase="phase_1_3")
  ```

---

## 15. 구동 환경

- **Python + Google Colab Pro+** (T4/A100 GPU)
- 각 Step 노트북(`.ipynb`)으로 배포/실행
- 가상 디스플레이(xvfb 등) 불필요 (matplotlib 사전 렌더링 방식)
- 파일 저장: H: drive (Google Drive 동기화)

---

## 16. autoresearch 운영 방침

### 16.1 기본 원칙

```
autoresearch = Claude API + RL 학습 + 자동 평가의 순환 루프
목적: 각 Step에서 baseline reward를 출발점으로
      변형을 자동 생성·학습·평가하여 최적 reward 채택
```

### 16.2 변형 대상 인자

#### A. 변형 가능

| 분류 | 항목 | 비고 |
|------|------|------|
| Reward 가중치 | 모든 reward 항목의 가중치 | Claude API 자유 판단 |
| Reward 항목 | 새 항목 추가 / 기존 항목 제거 | **§16.4 자동 게이트 통과 필수** |
| Hyperparameter | learning rate, gamma, n_steps, batch_size, ent_coef 등 | 자유 |
| 부가 구조 | Activation, Dropout, LayerNorm 위치, Optimizer 종류 | 자유 |

#### B. 변경 절대 금지 (전이학습 보장)

| 항목 | 이유 |
|------|------|
| Backbone hidden dim | Step 1~10 전체 256 (전 Step 동일) |
| Observation dim 구조 | 150-dim 고정 슬롯 할당 |
| Action space 정의 | 마스킹은 변경 가능, 기본 정의 불변 (※ macro-action 도입 시 L-D3 재해석) |
| Layer 수의 큰 변경 | 가중치 mapping 호환성 |

#### C. 조건부 허용

| 항목 | 조건 |
|------|------|
| Hidden dim 미세 조정 (예: 256 → 224 / 288) | 다음 step 진입 시 256으로 되돌릴 수 있어야 함 |
| Layer 추가 (residual 등) | 가중치 mapping 명확해야 함 |

### 16.3 baseline reward function (Step별)

```
Step 1: r = -w1·length - w2·collision + w3·goal_bonus
        + w4·direction_align - w5·wrong_dir

Step 2: Step 1 + (-w6·bend_radius_violation - w7·excess_bend_count)

Step 3: Step 2 + (-w8·slope_violation - w9·z_uplift_in_gravity)

Step 4: Step 3 + (-w10·branch_total_length_excess
                   - w11·branch_angle_violation
                   - w12·branch_on_bend)

Step 5: Step 4 + (-w13·valve_height_violation
                   - w14·valve_clearance_violation
                   - w15·valve_missing)

Step 6: Step 5 + (-w16·support_interval_violation
                   - w17·total_system_weight_excess)
```

> **baseline의 항목과 부호는 lock**. autoresearch는 가중치 값 변형 + 새 항목 추가 가능.

#### 16.3.1 Phase 1 가중치 초기값 (2026-06-18, L-16.3-w 해제)

```
[Step 1 — Phase 1 시작 초기값]
  w1 (length)         = 0.1     ← step 당 가벼운 길이 penalty
  w2 (collision)      = 2.0     ← 마스킹 외 충돌 억제, length 의 2배 자릿수
  w3 (goal_bonus)     = 50.0    ← 도달 시 +50, episode penalty 대비 약 3배
  w4 (direction_align) = 5.0    ← §12.4 고정값 (이미 spec 박힘)
  w5 (wrong_dir)      = 15.0    ← §12.4 고정값 (이미 spec 박힘)
```

**단위 분석 근거** (sparse reward trap 방지):

```
Episode 평균 (Medium 난이도, 학습 초기):
  length 50 step / collision 5회 / goal 도달 시:
    r = -0.1*50 - 2.0*5 + 50.0*1 + 5*1 - 15*0
      = -5 - 10 + 50 + 5
      = +40  (net positive 확보)

  goal 미도달 (timeout):
    r = -0.1*100 - 2.0*10 + 0
      = -30
  
  도달/미도달 차이 = 70, 충분한 학습 신호
```

FAILURE_LOG §6.5 placeholder 예시 (goal_bonus 100 → 500 의 sparse trap) 가 가리키는 함정 회피. w3 가 episode 누적 penalty 보다 충분히 크다.

#### 16.3.2 Step 1 autoresearch sweep 범위 (2026-06-18, L-16.3-w 해제)

```
Stage 1 sweep (Round 2 — Round 1 의 α/β best 고정 후):
  w1 ∈ {0.05, 0.1, 0.2, 0.5}      # 4 values
  w2 ∈ {1.0, 2.0, 5.0}             # 3 values
  w3 ∈ {20, 50, 100}               # 3 values
  → 4 × 3 × 3 = 36 variants

Stage 2: 살아남은 후보 ± 50% local grid
```

w4, w5 는 §12.4 고정값이므로 sweep 대상 아님.

#### 16.3.3 Step 2~6 가중치 — 상대 스케일 원칙 (Phase 1 후 후행 결정)

w6 ~ w17 의 구체값은 **각 Step 진입 직전 결정**. 다만 다음 원칙을 따른다:

```
1. Hard constraint 위반 (slope_violation, bend_radius_violation, valve_height_violation 등) 의 w:
   해당 Step reward 합산이 항상 net negative 가 되도록 큰 값 (~20)
   
2. Soft constraint (excess_bend_count, branch_total_length_excess 등) 의 w:
   length penalty (w1) 와 자릿수 비슷 (~0.5)
   
3. 새 Step 의 w 도입 시 이전 Step 의 r 합산을 크게 변경하지 않도록
   회귀 검증 통과 위함 (§12.3 회귀 임계값과 정합)
```

### 16.4 새 reward 항목 자동 검증 게이트 (4-Gate)

autoresearch가 새 reward 항목을 제안할 때 다음 4개 게이트를 모두 통과해야 학습 진행:

```python
def validate_new_reward_item(proposal):
    # Gate 1: 측정 가능성 (격자 좌표로 계산 가능한 수식인지)
    if not is_grid_computable(proposal.formula):
        return REJECT("Not grid-computable")

    # Gate 2: 기존 항목과 충돌 (반대 방향 작동 등)
    if conflicts_with_existing(proposal):
        return REJECT("Conflicts with existing items")

    # Gate 3: 평가 metric 정렬 (해당 step의 evaluation metric을 해치지 않음)
    if hurts_evaluation_metric(proposal):
        return REJECT("Misaligned with evaluation")

    # Gate 4: 부호 일관성 (예: "길이 짧을수록 좋다"인데 "길이 길수록 보너스" → reject)
    if sign_inconsistent(proposal):
        return REJECT("Sign inconsistent")

    return ACCEPT
```

> 게이트 통과 != 효과 보장. 학습 후 평가에서 결과 나쁘면 자동 폐기.

### 16.5 변형 실행 방식

#### A. 동시 실행 원칙
**한 번에 한 인자만 변형** (ablation 원칙).
인과관계 추적을 위해 동시 다인자 변형 금지.
단, **같은 인자의 다른 값**을 동시 학습하는 것은 허용 (예: w_length=[0.5, 1.0, 1.5, 2.0] 동시).

#### B. 학습 시간 압축 (2-stage screening) — 2026-06-18 L-C2 해제

```
정상 학습 시간 (Phase 1 Step 1):  2,000,000 timestep

Stage 1 (1차 스크리닝):  250,000 timestep (정상의 1/8)
  - 변형 N개 모두 학습
  - 평가 metric 측정 → 하위 50% 제거

Stage 2 (정상 학습):       2,000,000 timestep
  - 살아남은 후보만 정상 학습
  - 최종 평가 → 1개 채택

동시 학습 가능 모델 수:  T4 GPU 1개 기준 2~3개 (안전 2, 속도 우선 3)
                          Colab Pro+ / A100 환경에서는 4~6개 가능
```

**근거**: 1/8 비율은 (a) 학습 곡선 초기 trajectory 안정화 충분 + (b) 12 variants 합리적 시간 (T4 기준 약 2시간) 균형점. 50% 제거는 successive halving 표준 비율. 70% 제거는 짧은 학습 노이즈로 좋은 variant 잘려나갈 위험.

#### C. 변형 제안 메커니즘 (Optuna TPE + Claude API)

```
1. Stage 1 (12~36 variants):
   Optuna 의 enqueue_trial() 로 grid 명시 등록 (sweep 보장)
   예: α × β = 12 variants 또는 w1 × w2 × w3 = 36 variants

2. Stage 2 (살아남은 후보):
   TPE (Tree-structured Parzen Estimator) sampler 로 local search
   예: best 주변 ± 50% 영역에서 효율적 탐색

3. Round 종료 시:
   Optuna importance analysis 로 "어떤 인자가 결과 분산을 가장 설명하는지" 정량 추출
   다음 Round 의 sweep 대상 결정 근거
```

#### D. 인자 간 interaction 처리
```
1차 sequential sweep (Round 별):
  Round 1: α, β (L-D2 PBS 가중치) — w 는 초기값 고정
  Round 2: w1, w2, w3 — α, β 는 Round 1 best 고정
  Round 3+: 인자 추가 또는 interaction grid

2차 grid search (좋은 영역 주변):
  인자 간 interaction 효과 포착
  예: α ∈ [best ± 50%] × w3 ∈ [best ± 50%]
```

### 16.6 autoresearch 종료 조건 (2026-06-18 L-C2 해제)

#### A. Round 단위 정의

```
1 Round = Stage 1 (N variants) + Stage 2 (N/2 variants) + 평가
        ≒ 1.5 ~ 3 시간 (T4 GPU, 동시 3개 가정)
```

#### B. 종료 조건 (OR)

```
1. Round 5 도달 (절대 상한)
2. 최근 2 Round 의 best 변형이 동일 (개선 정체)
3. wandb best variant 학습 곡선 plateau 명확
   기준: 마지막 200K timestep 의 success_rate 증가 < 2.0%p (부동소수점 epsilon 1e-6 허용)
4. 사용자 강제 종료
```

**근거**: Round 5 상한은 Stage 1 reward tuning 의 표준 깊이. 그 이상은 천장 (ceiling) 가능성. plateau 임계값 2%p 는 통계 노이즈 (1%p) 보다 약간 큰 값.

### 16.6.5 Optuna 통합 spec (2026-06-18 L-C2 해제)

#### A. Optuna 채택 근거

- `optuna.importance.get_param_importances()` 가 "어떤 인자가 결과 분산을 가장 설명하는가" 정량 추출
- Phase 1 천장 vs search 부족 진단의 핵심 도구
- TPE sampler 가 단순 grid 보다 효율적

#### B. 운영 spec

```python
import optuna

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(n_startup_trials=12),  # Stage 1 grid 보장
    pruner=optuna.pruners.MedianPruner(),                      # plateau 자동 차단
    storage="sqlite:///autoresearch.db",                        # 영구 보관
    study_name=f"step{N}_round{R}",
)

# Stage 1: 명시 grid enqueue
for variant in stage1_grid:
    study.enqueue_trial(variant)

# Stage 2 이후: TPE 가 알아서 sampling
study.optimize(objective, n_trials=N_total)

# Round 종료 시 importance 추출
importances = optuna.importance.get_param_importances(study)
```

### 16.6.6 wandb naming 체계 (2026-06-18 L-C2 해제)

```python
wandb.init(
    project="pipe-routing-rl",                       # 전 Step / 전 Phase 공통
    name=f"step{N}_round{R}_stage{S}_var{V:03d}",    # 예: step1_round1_stage1_var007
    config={
        "step": N,
        "round": R,
        "stage": S,
        "variant_id": V,
        "alpha": ...,          # PBS Φ_goal 가중치
        "beta": ...,           # PBS Φ_cong 가중치
        "w1": ..., "w2": ..., "w3": ...,  # baseline 가중치 (L-16.3-w)
        # 기타 sweep 대상 hyperparameter
        "git_commit": ...,     # 재현성
        "generator_version": ...,  # 시나리오 호환성
    },
    tags=[
        f"step{N}",
        f"round{R}",
        f"stage{S}",
        "autoresearch",
        "phase1" if N == 1 else "phase2plus",
    ],
)
```

parallel coordinates plot 의 의미를 위해 sweep 대상은 모두 config 에 logging. tag 는 wandb UI 필터링용.

### 16.6.7 Phase 1 autoresearch 운영 순서 (2026-06-18 L-D2 + L-16.3-w + L-C2 정합)

```
Round 1: L-D2 α × β sweep (12 variants)
  Stage 1: 250K timestep × 12 → 하위 6 제거
  Stage 2: 2M timestep × 6 → best 1 선택
  결과: best (α, β) 확정

Round 2: L-16.3-w 가중치 sweep (36 variants)
  α, β 는 Round 1 best 고정
  Stage 1: 250K timestep × 36 → 하위 18 제거 (50%)
  Stage 2: 2M timestep × 18 → best 1 선택
  결과: best (w1, w2, w3) 확정

Round 3 (optional): hyperparameter sweep
  candidate: learning_rate, batch_size, ent_coef, gamma
  Stage 1 / Stage 2 동일 패턴

Round 4 (optional): interaction grid
  예: α × w3 grid 또는 β × w1 grid

Round 5 (조건부): 추가 reward 항목 검토
  4-Gate 통과 시에만

종료: Round 5 도달 or plateau 감지 or 사용자 강제 종료
```

### 16.7 Dense Reward (Potential-Based Shaping) — 의사결정 10, L-D2 해제

> **2026-05-14 신규**: HRLP (Tan & Mu 2024) 영감. 자세한 근거는 PROGRESS.md Session 2026-05-14 의사결정 10 참조.
> **2026-06-18 L-D2 해제**: Phase 1 적용 spec 구체화. Φ 구성, 초기값, sweep 범위, 코드 강제 안전장치, wandb 진단 panel 정의.

#### 16.7.1 도입 형식

baseline reward (§16.3) 에 다음 dense term을 추가:

```
r_total(s, a, s') = r_baseline(s, a, s') + r_shape(s, s')

r_shape(s, s') = γ_PPO · Φ(s') - Φ(s)     ← Potential-Based Shaping (Ng et al. 1999)
```

`γ_PPO` 는 PPO 의 discount factor 와 **반드시 동일 값**. autoresearch sweep 시 γ 변경하면 PBS 의 γ 도 함께 변경 (`pbs_gamma_sync` constraint).

#### 16.7.2 Phase 1 Φ 구성 (L-D2 해제 결과)

```
Phase 1 (Step 1 dense reward 도입):

  Φ(s) = α · Φ_goal(s) + β · Φ_cong(s)
  
  Φ_goal(s) = -d_manhattan(s, goal) / d_initial
              범위: [-1, 0]
              d_manhattan: 7-direction discrete action 과 정합
              d_initial: episode 시작 시점의 d_manhattan(s_0, goal)
                         → 정규화로 |Φ_goal| ≤ 1 보장
  
  Φ_cong(s) = -1 / (mean_SDF_5x5x5(s) + 1.0)
              mean_SDF_5x5x5: 현재 위치 주변 5x5x5 cube 의 평균 SDF
              범위: 음수, |Φ_cong| ~ O(1)
              해석: 막힌 영역 진입 회피
  
  Φ_future(s) = Phase 1 미사용. Phase 2+ 검토.
                (정의 모호성 — mini A* / value approximation / heuristic
                 중 어느 것도 비용·신뢰도 trade-off 미해결)
```

#### 16.7.3 Terminal 처리 (PBS 조건 2 명시)

```
Φ(s) where s is terminal:

  goal_reached terminal:    Φ = 0  (자연스러움, d_goal = 0)
  collision terminal:        Φ = 0  (명시적 set, PBS 조건 2)
  timeout terminal:          Φ = 0  (명시적 set, PBS 조건 2)
  out_of_bounds terminal:    Φ = 0  (명시적 set, PBS 조건 2)
```

비-goal terminal 에서 명시적 0 set 을 누락하면 정책 보존 이론적 보장이 깨짐. **구현 시 unit test 의무**.

#### 16.7.4 PBS 안전장치 (이론적 보장 조건)

PBS는 다음 조건 하에서 **optimal policy 보존이 이론적으로 증명**됨 (Ng, Harada & Russell 1999):

```
조건 1. Φ(s)는 state만의 함수 (action 의존 금지)
조건 2. Φ(terminal_state) = 0
조건 3. r_shape = γ·Φ(s') - Φ(s) 형식 (단순 cost 차이 아님)
조건 4. (본 프로젝트 추가): γ_PBS == γ_PPO
```

위 조건을 지키면 reward hacking 위험 없이 학습 속도만 빨라짐. 4번째 조건은 본 프로젝트의 PBS-PPO 호환성을 위한 안전장치.

#### 16.7.5 코드 레벨 강제 — `PotentialBasedShaping` helper class

PBS 조건을 type level + runtime check 로 강제:

```python
from typing import Callable
import inspect
import numpy as np

class PotentialBasedShaping:
    """PBS 4대 조건 hard-enforced shaping reward wrapper."""
    
    def __init__(
        self,
        phi_fn: Callable[[np.ndarray], float],  # state-only 시그니처
        gamma_ppo: float,                       # MUST equal PPO's gamma
        terminal_phi: float = 0.0,              # PBS 조건 2
    ):
        # 조건 1 검증: phi_fn이 state만 받는지 시그니처 체크
        sig = inspect.signature(phi_fn)
        if len(sig.parameters) != 1:
            raise ValueError(
                "PBS 조건 1 위반: Φ must be state-only (single argument)"
            )
        
        self.phi_fn = phi_fn
        self.gamma = gamma_ppo
        self.terminal_phi = terminal_phi
    
    def shaping_reward(self, s, s_next, is_terminal: bool) -> float:
        """조건 2, 3, 4를 만족하는 r_shape 계산."""
        # 조건 2: terminal Φ = 0
        phi_next = self.terminal_phi if is_terminal else self.phi_fn(s_next)
        phi_curr = self.phi_fn(s)
        # 조건 3, 4: γ_ppo·Φ(s') - Φ(s)
        return self.gamma * phi_next - phi_curr
```

본 class 를 거치지 않은 `r_shape` 직접 작성은 SKILL §3.5 규칙 위반.

#### 16.7.6 초기값과 autoresearch sweep 범위

```
초기값 (Phase 1 1차 학습):
  α = 1.0   (Φ_goal 가중치, 핵심)
  β = 0.1   (Φ_cong 가중치, 보수적 시작)

autoresearch Stage 1 sweep:
  α ∈ {0.5, 1.0, 2.0, 5.0}
  β ∈ {0.0, 0.1, 0.3}
  → 4 × 3 = 12 variants

autoresearch Stage 2 (살아남은 후보):
  α, β ± 50% local grid
  (예: Stage 1 best 가 α=2.0, β=0.1 이면
       Stage 2: α ∈ {1.0, 1.5, 2.0, 2.5, 3.0}, β ∈ {0.05, 0.075, 0.1, 0.125, 0.15})
```

**12 variants 의 wall-clock 시간 검토 필요**: L-C2 해제 시 동시 학습 가능 모델 수 (Colab GPU 자원) 가 결정되면 wall-clock 산출 가능. 12 variants 가 과도하면 Stage 1 을 α 단독 sweep (4 variants) → β 별도 sweep (3 variants) 의 sequential 로 분리 가능.

#### 16.7.7 autoresearch 변형 인자 규칙 (SKILL §4.1 연계)

```python
# Phase 1 autoresearch에서 자유 변형 가능 (PBS 4-Gate 통과 조건):
ALLOWED_FREE_PBS = [
    'alpha',          # Φ_goal 가중치
    'beta',           # Φ_cong 가중치
    'phi_goal_norm',  # d_initial 정규화 방식 (manhattan / euclidean / sdf-aware)
    'phi_cong_window_size',  # SDF window (3x3x3 / 5x5x5 / 7x7x7)
]

# 변경 절대 금지:
FORBIDDEN_PBS = [
    'pbs_form',           # γΦ(s') - Φ(s) 형식 불변
    'phi_state_only',     # state-only 시그니처 불변
    'terminal_phi_zero',  # Φ(terminal) = 0 불변
    'gamma_sync',         # γ_PBS == γ_PPO 불변
]

# 조건부 허용:
CONDITIONAL_PBS = [
    'phi_future_addition',  # Phase 2+ Φ_future 도입 (별도 4-Gate + PBS 안전조건 통과)
]
```

#### 16.7.8 wandb 진단 panel (의사결정 9 연계)

dense reward 효과 진단을 위해 wandb 에 다음 metric 을 분리 logging:

```
r_baseline_mean         : baseline reward 의 step 평균
r_shape_mean            : shaping reward 의 step 평균
r_shape_to_baseline_ratio : |r_shape_mean| / |r_baseline_mean|
                          → 정상 범위: 0.1 ~ 1.0
                          → > 2.0: shape 가 baseline 압도 (정책 왜곡 위험)
                          → < 0.05: shape 가 학습 신호로 미작동

phi_goal_at_step0       : episode 시작 시점 Φ_goal 값
phi_goal_at_terminal    : episode 종료 시점 Φ_goal 값 (goal_reached 한정)
phi_goal_trajectory     : Φ_goal 의 episode 진행 곡선 (평균)

phi_cong_distribution   : Φ_cong 의 histogram (학습 진행에 따른 shift 추적)

action_value_phi_corr   : Q-value gradient 와 Φ gradient 의 코사인 유사도
                          → ≈ 0: dense reward 가 학습 신호로 작동 안 함 (조정 필요)
                          → > 0.5: dense reward 가 정책 방향 일치
```

**진단 우선순위**:
1. `r_shape_to_baseline_ratio` — 첫 진단. 자릿수가 안 맞으면 다른 metric 의미 없음
2. `action_value_phi_corr` — dense reward 가 학습에 실제 기여하는지
3. `phi_goal_trajectory` — Φ 가 episode 진행에 따라 정상 변화하는지

세 metric 이 모두 정상인데 학습 plateau 라면 **천장 (ceiling)** 일 가능성. 그 외에는 search 부족 또는 Φ 설계 결함.

### 16.8 wandb 통합 — 의사결정 9

> **2026-05-14 신규**: autoresearch 가시성 확보. 자세한 근거는 PROGRESS.md Session 2026-05-14 의사결정 9 참조.

#### 도입 의무

모든 autoresearch 학습 run은 wandb 로그를 남긴다. sb3-contrib MaskablePPO + wandb native callback 통합.

```python
from wandb.integration.sb3 import WandbCallback

run = wandb.init(
    project="pipe-routing-rl",
    name=f"step{N}_variant_{variant_id}",
    config={...전체 config...},
    tags=[f"step{N}", "autoresearch"]
)

model.learn(
    total_timesteps=TIMESTEPS,
    callback=WandbCallback(verbose=2)
)
```

#### 활용 방식

1. **개별 학습 추적**: learning curve, episode reward, value loss 등 자동 기록
2. **다중 실험 비교**: Parallel Coordinates Plot으로 변형 간 패턴 즉시 인식
3. **천장 진단**: best variant 결과의 시간순 진화 확인 → 천장 vs 추가 search 여지 판단
4. **Colab 세션 끊김 대비**: 클라우드 저장 → 세션 끊겨도 결과 보존

#### Optuna importance analysis 보조 (선택)

autoresearch가 Optuna 기반 운영 시 (L-C2 결정 시점) 다음을 추가 활용:

```python
import optuna
importances = optuna.importance.get_param_importances(study)
# {'learning_rate': 0.42, 'w_length': 0.28, ...}
```

→ "어떤 인자가 결과에 가장 영향" 정량적 추출. wandb parallel coordinates 와 결합 시 "어떤 인자의 어떤 영역이 좋은지" 시각적 + 정량적 진단 가능.

---

## 17. (폐기) Macro 학습 환경

> **⚠️ 2026-05-14 폐기**: 본 장은 v0.1의 Hierarchical 구조 전제로 작성되었으며, 의사결정 3-r1 (Hierarchical 폐기) 으로 전체 폐기됨.
>
> 폐기된 내용:
> - §17.1 Macro 후보 생성 (A* + Yen's K-shortest paths)
> - §17.2 Visibility Graph 구성
> - §17.3 Macro Observation/Action
> - §17.4 Macro Reward
>
> Step 7~10 다중 파이프 환경의 단일 에이전트 구조 적용 방식은 §12.3 + L-D1 (전이학습 전략) 후속에서 정의됨.

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

### ~~L-A5: 평가 시나리오 구성~~ (2026-06-18 해제)

해제 내용: §11.0 시나리오 spec 신설로 결정 완료.
- 시나리오 4분할 (학습 무한대 / screening 75 / regression 150 / final eval 500)
- 난이도 분포 3:5:2 + Step별 난이도 축 차별화
- Seed range 분리 체계 (Step별 100,000 단위)
- Procedural 100% 시작 + FAILURE_LOG 누적 수작업 추가
- 회귀 검증 통과 임계값 (Stage 1 / Stage 2 분리)
- 구체 spec: §11.0.1 ~ §11.0.7 참조

### ~~L-A6: Procedural Scenario Generator 알고리즘 spec~~ (2026-06-18 해제)

해제 내용: §11.0.8 신설 (Generator v1.0.0 spec 확정).

핵심 결정:
- 장애물 배치: Poisson disk sampling + Cuboid 합성 (tank/structural/small_eq 비율 4:3:3)
- Start-Goal sampling: rejection sampling + A* reachable 검증 (학습 신호 아님, 시나리오 검증 전용)
- 난이도 파라미터 매핑: `DIFFICULTY_PARAMS` dict + `_inherits` 키로 Step 상속
- Step 4~6 sub-algorithm: 개요만 spec 화, 각 Step 진입 직전 구체화
- Step 7~10 sub-algorithm: Step 7 진입 시점 별도 세션
- Generator version 관리: semver (major.minor.patch) + git tag + hash 보관
- 구체 spec: §11.0.8.1 ~ §11.0.8.6 참조

### L-A3': Step 1~3 평가 임계값 구체값

- Step 1: success_rate 임계값, length_ratio 임계값
- Step 2: distance_to_pareto 정의 및 임계값
- Step 3: length_ratio 임계값 (slope_violation은 0 hard로 결정)

> **2026-06-18 참고**: L-A5 해제로 시나리오 set 이 정해졌으므로 Phase 1 결과의 empirical CDF 측정 가능. Phase 1 후 후행 결정 권장.

### ~~L-C2: autoresearch 메타 구조~~ (2026-06-18 해제)

해제 내용: §16.5 Stage 1/2 timestep 명시, §16.6 종료 조건 채움, §16.6.5 Optuna 통합 spec, §16.6.6 wandb naming, §16.6.7 Phase 1 운영 순서 신설.

핵심 결정:
- 1차 스크리닝 시간 비율: 1/8 (정상 2M 의 250K)
- 1차 스크리닝 제거 비율: 50% (successive halving 표준)
- 동시 학습 가능 모델 수: T4 GPU 1개 기준 2~3개 (Pro+/A100 환경에서 4~6개)
- autoresearch Round 상한: 5 (또는 plateau 감지 시 조기 종료)
- Round 단위: Stage 1 + Stage 2 + 평가 ≒ 1.5~3 시간
- Optuna 채택: TPE + MedianPruner + sqlite storage
- wandb naming: `step{N}_round{R}_stage{S}_var{V:03d}` + tag 체계
- 구체 spec: §16.5 ~ §16.6.7 참조

### L-C4: grid 좌표계 통합 (Macro 폐기 후 단순화)

> **2026-05-14 단순화**: 이전의 "grid ↔ graph 좌표계 통합" 에서 graph 좌표계 (Macro의 visibility graph) 가 폐기되어 단순화됨.

- grid cell 좌표 ↔ mm 좌표 변환 정밀도
- 환경 reset / step 시 좌표 단위 일관성 검증

### ~~L-16.3-w: baseline reward 가중치 초기값 및 sweep 범위~~ (2026-06-18 해제)

해제 내용: §16.3.1 Phase 1 초기값, §16.3.2 sweep 범위, §16.3.3 Step 2~6 상대 스케일 원칙 추가.

핵심 결정:
- Step 1 초기값: w1=0.1, w2=2.0, w3=50.0, w4=5.0 (§12.4 고정), w5=15.0 (§12.4 고정)
- Step 1 sweep: w1∈{0.05, 0.1, 0.2, 0.5} × w2∈{1, 2, 5} × w3∈{20, 50, 100} = 36 variants (구 spec "27" 오타 — 의사결정 27)
- 운영 방식: Sequential sweep (Round 1 = α/β, Round 2 = w1/w2/w3)
- Step 2~6 w: Phase 1 후 후행 결정. 상대 스케일 원칙 명시 (hard ~20 / soft ~0.5)
- 구체 spec: §16.3.1 ~ §16.3.3 참조

---

### ~~L-D1: 단일 에이전트 전이학습 전략~~ (2026-06-18 해제)

해제 내용: §12.2 (전이학습 메커니즘) + §12.3 (회귀 감시 체계) 채워짐. 의사결정 13 (2026-06-11) 의 범위 확장 (회귀 감시 임계값/주기/트리거 동시 결정) 도 반영 완료.

핵심 결정:
- 전략: 후보 A (순차 fine-tuning) + 후보 B (mixed curriculum) 혼합. C, D 비채택.
- 학습 데이터 mix: Warm-up 50/40/10 → 정상 60/30/10 (Step N 신규 / Step 1~N-1 / FAILURE_LOG)
- 학습률: 신규 학습의 50% 로 시작
- 회귀 감시 주기: 매 50K timestep
- 트리거: 3단계 escalation + Level 0 즉시 중단 (Layer 1 hard 5% 이상 위반 시)
- 회귀 임계값: §11.0.7 표 (Stage 1 / Stage 2 분리)
- 구체 spec: §12.2 ~ §12.3 참조

### ~~L-D2: Dense Reward Potential function Φ 구체 정의~~ (2026-06-18 해제)

해제 내용: §16.7 (Dense Reward) 본문 채워짐. Phase 1 적용 spec 확정.

핵심 결정:
- Φ 구성: Φ(s) = α·Φ_goal(s) + β·Φ_cong(s). Φ_future 는 Phase 1 미사용 (Phase 2+ 검토)
- Φ_goal = -d_manhattan(s, goal) / d_initial (정규화, 범위 [-1, 0])
- Φ_cong = -1 / (mean_SDF_5x5x5(s) + 1.0)
- Terminal Φ = 0 명시적 set (goal / collision / timeout / out_of_bounds)
- 초기값: α = 1.0, β = 0.1
- Stage 1 sweep: α ∈ {0.5, 1.0, 2.0, 5.0} × β ∈ {0.0, 0.1, 0.3} = 12 variants
- Stage 2: best 주변 ± 50% local grid
- `PotentialBasedShaping` helper class 로 PBS 4대 조건 코드 강제 (§16.7.5)
- wandb 진단 panel: r_shape_to_baseline_ratio, action_value_phi_corr, phi_goal_trajectory 등 (§16.7.8)
- γ_PBS == γ_PPO 강제 (4번째 안전조건 신설)

구체 spec: §16.7.1 ~ §16.7.8 참조

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

### (폐기된 Lock 항목 - 2026-05-14)

다음 항목은 Hierarchical 폐기로 인해 항목 자체가 폐기됨 (해제가 아님):

| 폐기 Lock | 폐기 사유 |
|----------|----------|
| L-B1 (Macro Observation/Action Space) | Macro 자체 폐기 |
| L-B2 (Macro Reward 설계) | Macro 자체 폐기 |
| L-B3 (다중 파이프 양보 학습 방식) | Macro 의존 항목, L-D1에서 단일 에이전트 방식으로 재정의 |
| L-B4 (Macro 학습용 시나리오 정의) | Macro 자체 폐기, L-A5에 통합 |
| L-12.2 (Macro 아키텍처 결정) | Macro 자체 폐기 |
| L-17.1 (Macro K값 및 다양성 threshold) | Macro 자체 폐기 |

---

## 100. Lock 해제 절차

각 Locked Item은 다음 절차로 해제한다:

```
1. 해당 항목에 대한 별도 논의 세션
2. 결론 도달 시 본 문서 업데이트:
   - L-XXX 섹션 삭제
   - 본문 해당 위치에 결정 사항 작성
3. Lock 해제 로그 (§101) 에 기록
4. 모든 Lock 해제 시 본 문서 → CLAUDE.md (확정판) 으로 rename
```

---

## 101. Lock 해제 로그

| 일자 | 해제/폐기 항목 | 결론 요약 |
|------|----------|----------|
| 2026-04-30 | - | 초기 spec 작성, Hierarchical 구조 채택 (의사결정 1~7) |
| 2026-05-14 | **§2 폐기/재작성** | Hierarchical 폐기, 단일 에이전트 + 전이학습 (의사결정 3-r1, 3-r2) |
| 2026-05-14 | **§12.2 폐기** | Step 6→7 Macro 도입 폐기 (의사결정 7 자동 폐기) |
| 2026-05-14 | **§12.3 폐기** | Step 7~10 Macro 신규 정의 폐기 |
| 2026-05-14 | **§13.2 폐기** | Step 6→7 핸드오프 확장 4파일 폐기 |
| 2026-05-14 | **§13.3 폐기** | Macro 핸드오프 폐기 |
| 2026-05-14 | **§17 폐기** | Macro 학습 환경 전체 폐기 |
| 2026-05-14 | **§11.5 보강** | Step 4~6 평가 Layer 1/2 분리 (의사결정 8) |
| 2026-05-14 | **§16.7 신규** | Dense Reward (PBS) 도입 (의사결정 10) |
| 2026-05-14 | **§16.8 신규** | wandb 통합 (의사결정 9) |
| 2026-05-14 | **L-B1/B2/B3/B4/12.2/17.1 폐기** | Macro 관련 Lock 항목 전체 폐기 |
| 2026-05-14 | **L-D1/D2/D3 신규** | 전이학습 전략 / Dense reward Φ / Macro-action 도입 검토 |
| 2026-06-11 | **§99 Lock 명명 규칙 신설** | Lock ID 카테고리 범례 표 내장 (의사결정 12) |
| 2026-06-11 | **L-D1 범위 확장** | 회귀 검증 임계값/주기/트리거 동시 결정 의무 (의사결정 13) |
| 2026-06-18 | **§11.0 신설** | 평가 시나리오 spec — 4분할, 난이도, seed range, 임계값 (L-A5 해제, 의사결정 14) |
| 2026-06-18 | **L-A6 신규** | Procedural Scenario Generator 알고리즘 spec (의사결정 15) |
| 2026-06-18 | **§12.2~12.3 채움** | 전이학습 A+B 혼합 + 회귀 감시 체계 구체화 (L-D1 해제, 의사결정 16) |
| 2026-06-18 | **§16.7 채움** | Dense Reward Φ Phase 1 spec 확정 (L-D2 해제, 의사결정 17) |
| 2026-06-18 | **§16.5~16.6.7 채움** | autoresearch 운영 spec — Stage 1/2 timestep, Optuna, wandb naming, Round 운영 (L-C2 해제, 의사결정 18) |
| 2026-06-18 | **§16.3.1~3 채움** | baseline reward 가중치 — Step 1 초기값 + sweep 범위 + 상대 스케일 원칙 (L-16.3-w 해제, 의사결정 19) |
| 2026-06-18 | **§11.0.8 신설** | Procedural Scenario Generator v1.0.0 알고리즘 — Poisson disk + rejection sampling (L-A6 해제, 의사결정 20) |
| 2026-06-24 | **🎓 \_ing 시스템 졸업** | CLAUDE_ing/SKILL_ing/PROGRESS_ing → CLAUDE/SKILL/PROGRESS. harness engineering 시작 가능 상태. 차단 표 의미 전환 (의사결정 21) |
| 2026-06-25 | **PBS 검증 강화** | state-only closure 경고 + unit test 의무화 (의사결정 22) |
| 2026-06-25 | **skill 충돌 해결** | Option A-1 채택. 로컬 SKILL.md = source of truth. SKILL.md §0.5 신설 (의사결정 23) |
| 2026-06-25 | **졸업 체크리스트 v2** | 서버 skill 동기화 + CLI 검증 항목 추가. SKILL.md §0.5 (의사결정 24) |
| 2026-06-28 | **핸드오프 폴더 분리** | `base_dir/step{N}/` 자동 생성. §13.1 보강 (의사결정 25) |
| 2026-06-28 | **Phase 1 첫 실증** | Sub-단계 5.1 50K baseline: success_rate 94%, PBS ratio 0.631. §16.3.1/§16.7 검증 (의사결정 26) |
| 2026-06-28 | **§16.3.2 산술 오류 수정** | `4×3×3=27` → `36` 오타 수정. 관련 파일 전수 갱신 + FAILURE_LOG entry (의사결정 27) |

> Lock **해제** 가 아니라 상위 **의사결정 재검토** 결과임에 유의.

---

**문서 버전:** v1.3
**졸업일:** 2026-06-24
**마지막 갱신:** 2026-06-28 (§101 의사결정 27: §16.3.2 산술 오류 수정 + FAILURE_LOG entry)
**이전 갱신:** 2026-06-28 v1.2 (§101 의사결정 25/26: 핸드오프 폴더 분리 + Phase 1 첫 실증)
**이전 갱신:** 2026-06-25 v1.1 (§101 의사결정 22/23/24), 2026-06-24 v1.0 (\_ing 시스템 졸업), 2026-06-18 v0.6 (3개 Lock 추가 해제), 2026-06-18 v0.5 (L-A5/L-D1/L-D2 해제), 2026-06-11 (§99 Lock 명명 규칙), 2026-05-14 (Hierarchical 폐기)

**졸업 후 spec 운영 원칙**:
1. spec 결정 변경 시 — Claude AI 세션에서 4개 파일 일괄 수정
2. 코드 구현 — Claude Code CLI 에서 본 spec 1:1 변환 (harness engineering)
3. 학습 결과 분석 → 새 의사결정 — Claude AI 세션
4. FAILURE_LOG entry 추가 시 — 실패 케이스를 수작업 seed pool 로 영구 등록 (§11.0.5)
5. vibe coding 금지 원칙 유지 — 새 결정 필요 시 별도 논의 세션
6. SKILL.md 갱신 시 — 로컬 수정 → git commit → 서버 skill in-place 갱신 → CLI 검증 (의사결정 23, SKILL.md §0.5)

**남은 활성 Lock 4개 — 자기 자연 시점 해제**:
- L-A3' (Phase 1 결과 후 후행 결정)
- L-A4 (Step 4 진입 시점)
- L-C4 (좌표계, implementation 시 unit test 위임 가능 — 사실상 자동 해제)
- L-D3 (Phase 1 결과 후 macro-action 도입 검토)
