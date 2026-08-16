# common_spec_r1.md — 파이프 자동배치 강화학습 공통 명세 (r1, 2026-06-24 보강)

> **본 문서는 `common_spec.md` 의 r1 (revision 1) 이다.**
>
> ## 변경 이력
>
> - **r0 (원본)**: pipe-routing-rl v1 시절 (Hierarchical + 숙련공 시연 + IRL 기반) 작성된 공통 명세.
> - **r1 (2026-06-24)**: pipe-routing-rl-v2 spec (`CLAUDE.md` v1.0 졸업판) 의 후속 의사결정에 의해 일부 영역이 폐기되었음을 표시.
>   본문 자체는 역사적 사실로서 보존하고, **§ 헤더 직후에 폐기 안내 박스만 추가**.
>
> ## 폐기 영역 mapping (2026-06-24 기준)
>
> | 영역 | 폐기 의사결정 | 대체 위치 |
> |------|--------------|----------|
> | §8.2 추상 비용 모델 | 물리 단위 (kg/mm) 채택 | `CLAUDE.md` §16.3 |
> | §9 공통 KPI | Layer 1/2 평가 체계 도입 (의사결정 8, 2026-05-14) | `CLAUDE.md` §11 |
> | §12.1 Backbone Hidden Dim Projection | Hierarchical 폐기 (의사결정 3-r1, 2026-05-14) | `CLAUDE.md` SKILL §3.1 (256 고정) |
> | §16 숙련공 피드백 루프 | 평가 철학 §11.1 ("숙련공 시연 = 정답 채택 X") | `CLAUDE.md` §11 |
> | §17 IRL 전략 | 평가 철학 §11.1 | `CLAUDE.md` §11 + §16.7 (Dense Reward 대체) |
> | §19 피드백 루프 종료 조건 | 평가 철학 §11.1 | `CLAUDE.md` §11 |
> | §20 step1 파일 구조 | 평가 철학 §11.1 + 단일 에이전트 구조 | `CLAUDE.md` SKILL §5 (졸업판 폴더 구조) |
>
> ## 살아있는 영역 (현재 v2 spec 과 정합)
>
> §1 비전 / §2 Step 로드맵 / §3 파이프 타입 / §4 JIS 사이즈 / §5 ASME B31.3 / §6 중력관 / §7 밸브 / §8.1, §8.3 서포트 간격 / §10 입출력 / §11 좌표계 / §12.2 Face 6방향 / §12.3 Action Masking / §13 핸드오프 / §14 시각화 / §15 구동 환경. 단, §7 밸브 범위는 `CLAUDE.md` §11.5 에서 700~1500mm 로 협소화.
>
> ## 원본 보존 원칙
>
> 폐기 영역의 본문은 그대로 보존한다. 폐기는 "그 결정이 잘못됐다" 가 아니라 "후속 의사결정으로 대체됐다" 의 의미이며, **역사적 정확성과 의사결정 흐름 추적을 위해 archive 로서 가치 있다**.

---

## 1. 프로젝트 비전

3D 공간 내 파이프를 자동 배치하는 강화학습 모델을 10단계에 걸쳐  
점진적으로 고도화하여, 최종적으로 300개 파이프(8개 타입)를  
동시에 최적 배치할 수 있는 시스템을 구축한다.

---

## 2. Step 진행 로드맵

| Step | 핵심 목표 | 누적 복잡도 |
|------|-----------|------------|
| 1 | 기본 경로 탐색 (직선+회피) | ★☆☆☆☆ |
| 2 | ASME B31.3 벤딩 반경 준수 | ★★☆☆☆ |
| 3 | 중력관 속성 (구배) 학습 | ★★☆☆☆ |
| 4 | 분기(Branch/Tee) 배치 | ★★★☆☆ |
| 5 | 밸브 + 인간 접근성 배치 | ★★★☆☆ |
| 6 | 서포트 최적화 (벽/천장 근접) | ★★★★☆ |
| 7 | 압력관 2개 동시 배치 (양보) | ★★★★☆ |
| 8 | 8개 타입 동시 배치 (양보) | ★★★★★ |
| 9 | 50개 파이프 동시 배치 | ★★★★★ |
| 10 | 300개 파이프 동시 배치 | ★★★★★ |

---

## 3. 파이프 타입 분류 체계 (8개 타입)

3개 이진 속성의 조합으로 8개 타입이 정의된다:

| Type ID | 관종 | 분기 | 밸브 | 도입 Step |
|---------|------|------|------|-----------|
| T1 | 압력관 | 비분기 | 밸브무 | Step 1 |
| T2 | 압력관 | 비분기 | 밸브유 | Step 5 |
| T3 | 압력관 | 분기 | 밸브무 | Step 4 |
| T4 | 압력관 | 분기 | 밸브유 | Step 
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

1.  **메모리(RAM) 점유율 최적화 (총 목표: ~24MB 이하)**
    - 거대한 공간(예: 20m × 15m × 5m @ 50mm 해상도)의 메모리 폭발을 방지하기 위해 데이터 타입을 최적화한다.
    - **Occupancy Grid (장애물/파이프):** `bool` 타임 적용 (~12MB).
    - **SDF Field (간섭거리):** `float32` 대신 `uint8`로 양자화(Quantization)하여 저장 (~12MB).
    - **Gradient:** 별도 저장 없이 런타임에 인접 셀과의 차분(Difference)으로 실시간 계산하여 메모리를 절약.

2.  **동적 장애물과 부분 EDT 업데이트 (Partial EDT)**
    - 에피소드 시작 시 정적 장애물에 대해 1회 전체 EDT(Euclidean Distance Transform)를 계산한다 (~1초).
    - 파이프 추가 등 동적 변화 발생 시, 해당 파이프의 AABB(Bounding Box) 주변 반경만 로컬로 **부분 업데이트(Partial EDT, ~0.01초)**하여 매우 빠르게 실시간 장애물을 반영한다.

3.  **RL 에이전트 관측 (Observation): 150-dim 고정 (Zero-padding)**
    - 전이학습 호환성을 위해 **관측 벡터를 150차원으로 전 Step(1~6) 고정**한다. 각 Step에서 미사용 슬롯은 `0.0`으로 채운다(Zero-padding).
    - **중복 피처 금지**: 동일 정보를 다른 인덱스에 중복 저장하지 않는다.
    - 입력값의 스케일 차이는 `VecNormalize`를 통한 자동 정규화로 대응하며, MLP [256, 256] 구조로 학습한다.
    - **Step 7 이후**: GNN 전환으로 노드 피처 차원으로 별도 관리.

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
| Long Radius Elbow | 1.5D | D = 호칭경(Nominal Diameter) |
| Short Radius Elbow | 1.0D | 공간 제약 시 |
| Hot Bend (열간 굽힘) | 5.0D | 현장 벤딩 |
| Cold Bend (냉간 굽힘) | 6.0D | 소구경 |

### 5.2 벤딩 각도
- 90° 엘보우 (표준)
- 45° 엘보우
- 기타 각도 (Step 후반에서 고려)

### 5.3 벤딩 시 추가 공간
벤딩 구간에서는 직선 구간보다 더 큰 공간이 필요하며,  
유효 외경 + 벤딩 반경을 고려한 점유 영역을 계산해야 한다.

---

## 6. 중력관 규격 (Step 3 이후 적용)

### 6.1 최소 구배 (Slope/Gradient)
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
- 수직 낙차(수직관) 구간은 허용
- 트랩 방지를 위한 연속 하향 보장

---

## 7. 밸브 & 인간 접근성 (Step 5 이후 적용)

### 7.1 밸브 설치 위치 제약
| 조건 | 값 | 비고 |
|------|-----|------|
| 최소 높이 (바닥면 기준) | 600mm | 허리 아래 한계 |
| 최대 높이 (바닥면 기준) | 1800mm | 손 닿는 한계 |
| 권장 높이 | 900~1500mm | 조작 최적 |
| 전면 여유 공간 | 최소 600mm | 사람 접근 |
| 측면 여유 공간 | 최소 300mm | 공구 사용 |

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

### 8.2 서포트 비용 모델

> **⚠️ 본 §8.2 는 v2 spec 에서 폐기됨 (2026-06-24 r1 표시).**
>
> 폐기 사유: 본 절의 추상 비용 모델 (벽 1.0, 행거 1.5, 바닥 2.0, 독립 5.0) 은 단위가 모호하고 다른 reward 항목과 자릿수 정합이 어려움. v2 spec 은 **물리 단위 (kg, mm)** 채택.
>
> 대체 위치: `CLAUDE.md` §16.3 baseline reward function w₁₆ (support_interval_violation), w₁₇ (total_system_weight_excess) 및 §16.3.3 상대 스케일 원칙 (Hard ~20 / Soft ~0.5) 참조.
>
> 본문은 역사적 사실로 보존.

- 벽 부착 서포트,천장 행거: 1.0 (기준 비용)
- 천장 행거: 1.5
- 바닥 스탠드: 2.0
- 독립 구조물: 5.0
- 벽/천장에서 떨어질수록 비용 증가

### 8.3 핵심 Trade-off
최단거리 (공간 중앙 직선) ←─── Trade-off ───→ 서포트 비용 최소 (벽/천장 근접)

---

## 9. 공통 KPI 정의

> **⚠️ 본 §9 전체는 v2 spec 에서 폐기됨 (2026-06-24 r1 표시).**
>
> 폐기 사유: 단일 scalar KPI (성공률 80~95%) 가 Step 4~6 의 "절대 최적 정의 불가능" 영역에서 작동 불가. v2 spec 은 **Layer 1 (binary hard constraint) + Layer 2 (relative quality)** 분리 평가 채택 (의사결정 8, 2026-05-14).
>
> 대체 위치:
> - Step 1~3: `CLAUDE.md` §11.2~11.4 (A* benchmark 대비 length_ratio, distance_to_pareto 등)
> - Step 4~6: `CLAUDE.md` §11.5 (Layer 1/2 분리)
> - 임계값 (L-A3'): Phase 1 결과 분포 본 후 후행 결정
> - Step 1~10 누적 성공 판정도 폐기 → 각 Step 의 자기 평가 metric 으로 대체
>
> 본문은 역사적 사실로 보존.

모든 단계 구역(Phase/Step)에서 1차적으로 **80% ~ 95% 수준의 성공률** 달성을 목표 임계점(Threshold)으로 잡습니다.
해당 KPI를 돌파해야 자율학습이 멈추고 전문가의 검증 단계(HitL)로 전환(Graduate)될 자격이 부여됩니다.

### 9.1 KPI 산출 공식

KPI = (성공 에피소드 수 / 전체 평가 에피소드 수) x 100%

### 9.2 성공 판정 기준 (Step별 누적)
| Step | 성공 조건 |
|------|-----------|
| 1 | 출발→도착 연결 + 간섭 없음 |
| 2 | + 모든 벤딩이 ASME B31.3 준수 |
| 3 | + 중력관 구배 조건 충족 |
| 4 | + 분기점 정상 연결 + 총 길이 최적화 |
| 5 | + 밸브 접근성 충족 |
| 6 | + 서포트 비용 기준값 이내 |
| 7 | + 2개 파이프 모두 성공 |
| 8 | + 8개 타입 모두 성공 |
| 9 | + 50개 파이프 모두 성공 |
| 10 | + 300개 파이프 모두 성공 |

---

## 10. 공통 입출력 형식

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
        {"type": "wall_bracket", "position": [x,y,z], "cost": 1.0}
      ],
      "metrics": {
        "total_length": 12500,
        "num_bends": 3,
        "slope_maintained": true,
        "valve_accessible": true,
        "support_cost": 15.5
      }
    }
  ],
  "global_metrics": {
    "total_pipes": 1,
    "success_count": 1,
    "kpi": 100.0,
    "total_pipe_cost": 125000,
    "total_support_cost": 15.5,
    "computation_time_sec": 5.2
  }
}
```

---

## 11. 공통 좌표계 & 단위

| 항목 | 상세 내용 |
|------|-----------|
| 좌표계 | 오른손 좌표계 (Right-handed System) |
| X축 | 가로 (동-서) |
| Y축 | 세로 (남-북) |
| Z축 | 높이 (상-하) |
| 기본 단위 | mm (밀리미터) |
| 각도 단위 | degree (도) |
| 구배 표시 | 비율 (예: 1/100) |
| 비용 단위 | 상대 비용 (무차원) |

---

## 12. 모델 상속 및 전이학습 전략

각 Step의 모델은 이전 Step의 가중치를 상속받아 학습 효율을 극대화한다. 이를 위해 **Backbone 네트워크의 구조는 전 Step에 걸쳐 동결(Frozen)되거나 확장 가능**해야 한다.

- Step 1 (기본 경로) → Step 2 (벤딩 추가): 경로 탐색 지능 유지
- Step 2 (압력관) → Step 3 (중력관): 구배 개념 학습 추가
- **필수 규칙**: 하위 Step 모델 로드 실패 시 해당 Step 학습을 시작하지 않음

### 12.1 [절충안] Backbone Hidden Dimension 전략

> **⚠️ 본 §12.1 은 v2 spec 에서 폐기됨 (2026-06-24 r1 표시).**
>
> 폐기 사유: 2026-05-14 의사결정 3-r1 에서 Hierarchical (Macro+Micro 별도 네트워크) 구조 폐기 + 단일 에이전트 + 전이학습 구조 채택. Step 7 에서 GNN (GCN→GAT→Transformer-XL) 진화 경로 자체가 폐기됨.
>
> v2 채택 구조:
> - **Backbone hidden dim 256 (Step 1~10 전체 고정)** — Projection 없음
> - Observation dim 150 (zero-padding) 으로 통일
> - 단일 에이전트로 다중 파이프 환경도 처리 (다른 파이프 정보를 obs[136:150] 활용)
>
> 대체 위치: `CLAUDE.md` §12 (단일 에이전트 + 전이학습) + SKILL §3.1 절대 변경 금지 항목.
>
> 본문은 역사적 사실로 보존 (Hierarchical 시절의 사고 흐름 archive).

> 두 AI 제안의 절충안으로 확정된 Backbone 차원 설계 기준.
> 핵심 원칙: **전이학습 연속성 최우선, 아키텍처 전환 시점에서만 1번 확장**

| 구간 | Hidden Dim | 아키텍처 | Projection | 근거 |
|------|:---:|---------|:---:|-----|
| **Step 1~6** | **256** | **MLP Fusion** | ❌ 불필요 | 1D Vector(SDF 센싱 등 ~294dim) 처리. 전이학습 완전 보장. |
| **Step 7** | **256→512** | MLP + **GNN(GCN)** 도입 | ✅ **1번만** | GNN이라는 근본적 아키텍처 전환. 유일하게 합리적인 확장 시점. |
| **Step 8~10** | **512** | GNN(GCN→GAT→Transformer-XL) | ❌ 불필요 | 노드 수 증가에 상관없이 GNN 파라미터 고정. |

```
[비교 대상 두 제안 요약]

상대 AI 제안: Step 3→4에서 256→512, Step 7에서 512→1024 (총 2번 Projection)
  문제: 분기(Branch) 9dim 추가로 2배 확장은 근거 부족.
        GNN에서 1024는 2~50노드 환경에 과파라미터화.

내 초기 제안: 256 전구간 고정 (Projection 없음)
  문제: GNN 도입이라는 근본적 아키텍처 변화를 반영 못함.

[절충안 핵심]: Projection은 Step 7(GNN 도입) 1번만.
  - Step 1~6: 256 유지 → 전이학습 손실 0
  - Step 7: nn.Linear(256, 512) 1개로 단순 매핑
  - Step 8~10: 512 유지 → 노드 수 증가에 파라미터 무관
```

### 12.2 [전 Step 공통] 출발/목표 연결 방향 제약 (Face 6방향 한정)

> **배관 공학 물리 원칙**: 파이프는 장비 노즐(Nozzle) 또는 다른 파이프와 연결될 때,  
> 반드시 **6개의 축 정렬 방향(+X/-X/+Y/-Y/+Z/-Z) 중 하나**로 직각 접속해야 한다.  
> 대각선(Edge/Vertex) 방향으로의 시작/종료 연결은 물리적으로 존재하지 않는다.  
> **이 원칙은 Step 1~10 전체에 예외 없이 적용된다.**

| 제약 항목 | 적용 방식 | 유지 여부 |
|-----------|-----------|:---------:|
| **출발 방향** (`start_dir_idx`) | `action_masks()` Hard Constraint — 물리적 선택 불가 | **전 Step 유지** |
| **목표 도달 방향** (`goal_dir_idx`) | `_calc_reward()` Soft Constraint — 페널티로 유도 | **전 Step 유지** |
| **Face 방향 인덱스** | `0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z` | **전 Step 동일** |

```python
# 에피소드 reset() 공통 로직 (Step 1~10 동일)
start_dir_idx = rng.integers(0, 6)          # 출발 방향: 6방향 중 1개 랜덤
goal_dir_idx  = rng.choice([i for i in range(6)
                             if i != start_dir_idx                              # 출발 방향 제외
                             and not np.array_equal(FACE_DIRS[i],
                                                    -FACE_DIRS[start_dir_idx])  # 직선 U형 제외
                            ])               # 나머지 4개 중 1개 랜덤
```

| 보상 항목 | 값 | 발동 조건 |
|---|---:|---|
| `direction_align_bonus` | **+5.0** | 첫 스텝이 `start_dir_idx` 방향과 일치 |
| `wrong_start_dir_penalty` | **-15.0** | 첫 스텝이 `start_dir_idx`와 다름 (Fallback) |
| `wrong_goal_dir_penalty` | **-15.0** | 목표 도달 시 `goal_dir_idx` 불일치 (Face) |
| `wrong_goal_dir_penalty × 1.5` | **-22.5** | 목표 도달 시 Edge/Vertex로 진입 |

> ⚠️ `start_dir_idx`, `goal_dir_idx`는 에피소드마다 랜덤하게 배정되며,  
> 에이전트는 **observation[91:94]** (출발방향 단위벡터) 과 **observation[94:97]** (목표방향 단위벡터)를 통해 이를 인식한다.

### 12.3 [전 Step 공통] Action Masking + MaskablePPO

> 모든 Step에서 **sb3-contrib의 MaskablePPO**를 사용하고, 환경은 `action_masks()` 메서드를 구현하여 불가능한 행동을 사전 차단한다.

| Step | 마스킹 규칙 (누적) |
|------|-------------------|
| **Step 1** | 충돌(장애물/벽) + 경계 초과 + 자기 경로 역주행 차단 + **출발 방향 강제** |
| **Step 2** | + 벤딩 반경 위반 방향 차단 |
| **Step 3** | + 중력관 Z 상승 소프트 마스킹 |
| **Step 4** | + 분기점 전용 행동 제약 (DynamicHook 기반) |
| **Step 5** | + 밸브 접근성 확보 불가 방향 차단 |
| **Step 6** | + 서포트 비용 과다 경로 소프트 마스킹 |
| **Step 7~10** | GNN 노드별 독립 마스킹 (Joint Action의 각 파이프마다 독립 마스크) |

```
MaskablePPO 전이학습 호환성:
  MaskablePPO의 Actor/Critic은 PPO와 동일한 구조.
  → Step N → N+1 전이 시 가중치 무손실 전이 보장.
  → 마스킹은 정책 네트워크의 출력 logit에 -inf를 적용하는 방식.
  → 가중치 자체는 변경되지 않으므로 전이학습에 간섭 없음.
```

---

## 13. 모델 핸드오프(Handoff) 및 내부 규격

Step 간 모델을 주고받을 때, 다음 6개 구성 요소가 하나의 패키지로 전달되어야 하며, 각 파일은 정해진 내부 구조를 준수해야 한다.

### 13.1 핸드오프 패키지 구성 (10개 파일)
| 파일명 | 역할 | 필수 포함 내용 |
|-------|------|--------------|
| `best_model.pt` | 코어 가중치 | State Dict (Backbone, Actor, Critic 가중치) |
| `best_model_meta.json` | 네트워크 아키텍처 | 레이어 깊이, Hidden Dimension, 입출력 차원 정보 |
| `stepN_normalizer.json` | 정규화 파라미터 | 공간 좌표, 파이프 직경 등에 대한 Min-Max 혹은 Mean-Std 값 |
| `stepN_task_embedding.pt` | 테스크 임베딩 | 각 Step별 특성(밸브유무 등)을 정의하는 벡터 지표 |
| `stepN_kpi_report.json` | 품질 검증서 | 이전 Step의 최종 KPI 및 상세 성공률 데이터 |
| `stepN_training_log.csv` | 학습 이력 요약 | 누적 학습 Episode, Learning Rate 설정값 등 |
| `stepN_action_space_map.json` | 행동 공간 정의 | 이산 행동 인덱스와 실제 물리적 방향 매핑 |
| `stepN_reward_config.json` | 보상 설정 | Step별 보상 가중치 및 페널티 기준 |
| `stepN_env_config.json` | 환경 설정 | 그리드 스케일, 채널 구성 등 환경 초기화 정보 |
| `stepN_validation_set.json` | 검증 시나리오 | 회귀 테스트를 위한 고정 시나리오 및 Step 결과 |

### 13.2 내부 텐서(Tensor) 규격 통일
- **Input Dimension**: 모든 Step의 에이전트는 동일한 관측 공간(Observation Space) 크기를 기본으로 하며, 확장이 필요할 경우 기존 인덱스를 유지하고 뒤에 추가한다.
- **Normalization 기준**: `normalizer.json`에 정의된 값을 모든 Step의 환경(Environment)에서 동일하게 참조한다. (예: Step 1에서 10m를 1.0으로 정규화했다면, Step 2에서도 동일 기준 적용)
- **Weight Key Mapping**: `state_dict` 내의 레이어 이름(Key)을 통일하여 `load_state_dict` 시 `strict=False` 없이도 호환되도록 관리한다.

> [!NOTE]
> 단, 상기항목중 건물에는 벽걸이식관리에 해당됩니다. 해당 설비들은 별도관리이며 기타설비의 관할과는 무관합니다.

---

## 14. 공통 시각화 및 평가 스택

전체 파이프 자동배치 시스템(Step 1 ~ Step 10)의 학습 평가 및 디버깅을 위한 시각화는 **Matplotlib + IPython HTML** 방식을 사용한다.

- **렌더링 방식**: Matplotlib(mpl_toolkits.mplot3d)으로 72프레임을 사전 렌더링(pre-render)하여 base64로 인코딩하고, `IPython.display.HTML`을 통해 Colab 셀에 인터랙티브 HTML 뷰어로 출력한다. 가상 디스플레이(`xvfb`, `pyvirtualdisplay`) 불필요.
- **필수 시각화 형태**: 렌더링 시 진행 경로를 명확히 파악할 수 있도록 항상 **파이프의 중심선(Centerline)과 두께가 반영된 볼륨(Cylinder)**을 겹쳐서 동시에 렌더링한다.
- **뷰어 기능**: 에피소드 선택, 3D 회전 애니메이션, Plan(X-Y) / Section(X-Z) / Profile(Y-Z) 2D 탭, 재생/정지/속도/줌 컨트롤 제공.
- **노트북 사용법 (2줄):**
  ```python
  from visualization.visualization import run_visualization
  run_visualization(model=model, phase="phase_1_3")
  ```
## 15. 구동 환경 및 실행 (Execution Environment)

전체 프로젝트 시스템은 **Python**으로 구현되며, 학습 및 추론, 시각화의 실제 구동은 주로 **Google Colab** 환경 내에서 수행되도록 설계된다.

- **Jupyter Notebook 구동**: 각 Step(Step 01 ~ Step 10) 시뮬레이션 환경 구축, 강화학습 모델 트레이닝, 평가 및 PyVista 등을 이용한 실시간/Interactive 시각화의 전 과정은 **Jupyter Notebook (`.ipynb`)** 형태로 구현되어 배포/실행된다.
- **Colab 특화 환경 고려**: GPU(T4/A100 등) 할당, 가상 디스플레이(xvfb, pyvirtualdisplay) 구성 등 Colab에서 3D 렌더링을 원활하게 수행하기 위한 셀 설정 코드가 각 과정의 노트북 초반부에 기본 명기 및 포함되어 구동된다.
---

## 16. 숙련공 피드백 루프 (Expert Feedback Loop)

> **⚠️ 본 §16 전체는 v2 spec 에서 폐기됨 (2026-06-24 r1 표시).**
>
> 폐기 사유: v2 spec 의 평가 철학 §11.1 ("숙련공 시연 = 정답 채택 X") 와 정면 충돌. 숙련공 시연을 정답으로 삼으면 시연을 천장으로 만들기 때문에 v2 의 "엔지니어를 뛰어넘는 것" 목표와 양립 불가.
>
> 폐기된 메커니즘:
> - ② Graduated Model → 성공 에피소드 10개 추출 (v2 는 autoresearch 의 wandb / Optuna 분석으로 대체)
> - ③ 숙련공 클릭 시연 (v2 는 시연 자체를 사용하지 않음)
> - ④ IRL 역산 (§17 폐기, 아래 참조)
> - ⑤ reward_config.json 업데이트 (v2 는 autoresearch sweep + 4-Gate 검증으로 대체)
> - ⑥ 숙련공 OK/NG 판정 (v2 는 회귀 임계값 자동 판정으로 대체, `CLAUDE.md` §11.0.7)
>
> 대체 위치:
> - Reward 업데이트: `CLAUDE.md` §16 (autoresearch + 4-Gate + Dense Reward) 전체
> - 학습 종료 판정: `CLAUDE.md` §16.6 종료 조건 (Round 5 / plateau / 사용자 강제)
> - 평가 metric: `CLAUDE.md` §11.0.7 (Stage 1 / Stage 2 임계값)
>
> 본문은 역사적 사실로 보존 (v1 시절 HitL 설계의 archive).

전체 학습 시스템은 RL 자율 탐색과 숙련공 피드백을 순환시켜 reward function을 점진적으로 고도화한다.
숙련공은 언어 코멘트가 아닌 **직접 경로 시연(클릭)**으로 피드백을 제공하며, 시스템이 그 행동에서 reward를 역산한다.

### 16.1 전체 순환 구조

```
① RL 학습 (MaskablePPO + reward_config.json)
        ↓
② Graduated Model → 성공 에피소드 10개 추출 및 3D 시각화
        ↓
③ 숙련공이 같은 격자/시작점/끝점에서 클릭으로 경로 직접 시연 (1~3개)
        ↓
④ 모델 경로 vs 숙련공 경로 비교 → IRL로 reward 역산
        ↓
⑤ reward_config.json 업데이트 제안 → 숙련공 최종 확인
        ↓
⑥ 숙련공 OK → 다음 Step/Phase 이동
   숙련공 NG → ①로 순환 (재학습)
```

### 16.2 숙련공 직접 시연 방식 채택 이유

| 방식 | 문제점 |
|------|--------|
| 언어 코멘트 → LLM 수식 변환 | 숙련공이 수식 검증 불가. reward hacking 위험 |
| OK/NG 판정만 | 왜 NG인지 정보 없음. reward 방향 불명확 |
| **직접 클릭 시연** | 행동 자체가 정답. 암묵지도 데이터화. 수식 검증 불필요 |

숙련공이 언어로 설명하지 못하는 암묵지(tacit knowledge)도 **행동으로는 표현 가능**하다.
클릭 시퀀스 = `(state, action)` 쌍의 나열이므로 별도 가공 없이 IRL 입력 데이터로 직접 사용된다.

### 16.3 성공 에피소드 10개 제시 기준

- Graduated Model이 생성한 에피소드 중 `success=True`인 것만 추출
- 다양성 확보: 경로 길이, 장애물 수, 파이프 사이즈가 겹치지 않도록 샘플링
- 모델의 반복 패턴(전략)이 드러나도록 유사한 구간을 포함한 것 우선
- 시각화: 모델 경로를 반투명으로 표시하여 숙련공이 차이를 인지하며 시연 가능

### 16.4 숙련공 OK 판정 기준

| 판정 | 조건 | 다음 행동 |
|------|------|----------|
| OK | 10개 에피소드 중 8개 이상에서 숙련공 경로와 모델 경로의 핵심 구간 일치 | 다음 Phase/Step 이동 |
| NG | 반복적으로 같은 구간에서 모델이 잘못된 선택 | reward 업데이트 후 재학습 |

---

## 17. IRL (Inverse Reinforcement Learning) 전략

> **⚠️ 본 §17 전체는 v2 spec 에서 폐기됨 (2026-06-24 r1 표시).**
>
> 폐기 사유: §16 폐기의 직접 부수효과. IRL 자체가 "숙련공 경로 = 정답 → reward 역산" 메커니즘인데, v2 평가 철학 §11.1 이 "숙련공 시연 = 정답 채택 X" 이므로 IRL 의 전제가 무너짐.
>
> 대체 메커니즘 (v2):
> - **Dense Reward (PBS)**: 학습 신호 부족 문제를 IRL 이 아닌 Potential-Based Shaping 으로 해결 (`CLAUDE.md` §16.7, Ng et al. 1999 의 정책 보존 이론적 보장).
> - **autoresearch + 4-Gate**: 새 reward 항목은 LLM 제안 + 4-Gate 자동 검증 (`CLAUDE.md` §16.4) + autoresearch sweep 으로 결정. 숙련공 승인 절차 없음.
> - **FAILURE_LOG dynamic source**: 실패 케이스가 수작업 seed pool 로 누적 (`CLAUDE.md` §11.0.5) — 숙련공 시연 없이도 어려운 케이스 학습 가능.
>
> 본문은 역사적 사실로 보존.

### 17.1 기본 원리

```
같은 환경 (격자, 장애물, 시작/끝점)
    모델 경로:    A → B → C → D → G
    숙련공 경로:  A → B → F → H → G

차이 구간: C-D vs F-H
    → 숙련공이 C를 피하고 F를 선택한 이유를 역산
    → 해당 구간의 SDF값, 접근공간, 인접 구조물 정보로 reward 항목 도출
```

### 17.2 역산 절차

```python
# 1. 경로 차이 구간 추출
diff_segments = extract_diff(model_path, expert_path)

# 2. 각 차이 구간의 격자 속성 비교
for seg in diff_segments:
    model_cell_features  = get_cell_features(seg.model_pos)
    expert_cell_features = get_cell_features(seg.expert_pos)
    feature_diff = expert_cell_features - model_cell_features

# 3. LLM에 feature_diff + 건축 배관 컨텍스트 전달
#    → reward 항목 및 계수 변경 제안 생성

# 4. 숙련공 확인 후 reward_config.json 반영
```

### 17.3 새로운 Reward 항목 생성 거버넌스

IRL 역산 과정에서 초기 설계에 없던 새로운 reward 항목이 생성될 수 있다.
이는 숙련공의 암묵지가 명시화되는 과정이므로 허용하되, 반드시 아래 절차를 따른다.

```
LLM이 새 reward 항목 제안
        ↓
거버넌스 체크 (자동):
  1. 기존 항목과 충돌 여부 (예: straightness vs collision_avoidance 충돌)
  2. 격자 좌표로 측정 가능한 수식인지
  3. reward_config.json에 추가 시 기존 계수 합산 범위 초과 여부
        ↓
숙련공 최종 승인 (필수)
        ↓
reward_config.json에 추가 + 변경 이력 기록
```

**항목 생성 금지 조건:**
- 격자 좌표로 측정 불가능한 항목 (주관적 미감 등)
- 기존 항목과 반대 방향으로 작동하는 항목 (사전 충돌 해소 필요)
- 숙련공 미승인 항목

### 17.4 reward_config.json 버전 관리

```json
{
  "version": "step1_phase3_v2",
  "updated_at": "ISO8601",
  "updated_by": "expert_feedback_loop",
  "changelog": [
    {
      "version": "step1_phase3_v1",
      "changed_items": ["bend_angle_penalty"],
      "reason": "숙련공 시연 #3: 급격한 꺾임 구간 회피 패턴 확인",
      "expert_approved": true
    }
  ],
  "reward_weights": {}
}
```

Step이 진행될수록 reward_config.json은 **숙련공 암묵지의 명문화 문서**로 축적된다.

---

## 18. grid_config.json — Single Source of Truth

RL 환경, 3D 뷰어, 클릭 인터페이스 세 모듈이 **반드시 동일한 격자 정의**를 참조해야 한다.
격자 불일치는 숙련공 클릭 좌표와 RL 학습 좌표가 어긋나는 치명적 버그를 유발한다.

### 18.1 grid_config.json 스키마

```json
{
  "grid_size": {
    "x": 40,
    "y": 30,
    "z": 10,
    "unit": "cells"
  },
  "cell_size_mm": 500,
  "origin_mm": [0, 0, 0],
  "coordinate_system": "right_handed",
  "axes": {
    "x": "east_west",
    "y": "north_south",
    "z": "vertical"
  }
}
```

### 18.2 모듈별 사용 규칙

| 모듈 | 역할 | grid_config 사용 방식 |
|------|------|----------------------|
| 모듈 A — RL 환경 | MaskablePPO 학습 | 격자 크기, 셀 단위로 state/action 정의 |
| 모듈 B — 3D 뷰어 | 에피소드 재생 | 격자 좌표 → mm 변환하여 렌더링 |
| 모듈 C — 클릭 UI | 숙련공 경로 시연 | 클릭 픽셀 → 격자 좌표 역변환 |
| 모듈 D — IRL 역산 | reward 추출 | 모델/숙련공 경로를 동일 격자 좌표로 비교 |

### 18.3 출력 파일 스키마

```json
// expert_episode.json — 숙련공 시연 저장 형식
{
  "episode_id": "EXP_001",
  "reference_model_episode": "MODEL_042",
  "grid_config_version": "v1.0",
  "pipe_spec": {
    "nominal_size": "100A",
    "insulation": true
  },
  "start": [2, 2, 4],
  "goal": [38, 28, 4],
  "obstacles": [],
  "expert_path": [[2,2,4], [3,2,4], [3,3,4], "..."],
  "timestamp": "ISO8601",
  "annotation": "숙련공 메모 (선택)"
}
```

---

## 19. 피드백 루프 종료 조건 및 폴더 구조

> **⚠️ 본 §19 전체는 v2 spec 에서 폐기됨 (2026-06-24 r1 표시).**
>
> 폐기 사유: §16, §17 폐기의 직접 부수효과. 피드백 루프 자체가 존재하지 않으므로 종료 조건 / step1/feedback/ 폴더 구조 모두 무관.
>
> 대체 위치:
> - 학습 종료 조건: `CLAUDE.md` §16.6 (Round 5 도달, plateau 감지, 사용자 강제 종료, 2 Round 연속 best 동일)
> - 폴더 구조: `CLAUDE.md` SKILL §5 졸업판 폴더 구조 (단일 에이전트 + autoresearch + Optuna 기반)
>
> 본문은 역사적 사실로 보존.

### 19.1 루프 종료 조건 (확정)

숙련공이 성공 에피소드 10개를 확인한 후 **아무 코멘트 없이 OK**를 선택할 때 루프가 종료된다.
코멘트 유무가 유일한 종료 기준이며, KPI 수치나 별도 임계값으로 자동 종료하지 않는다.

```
[루프 지속 조건] 숙련공이 코멘트를 입력하고 시연을 진행
[루프 종료 조건] 숙련공이 10개 에피소드를 보고 코멘트 없이 OK 선택
```

### 19.2 숙련공 판단 인터페이스 (단순 2분기)

복잡한 UI 없이 두 가지 선택지만 제시한다. 판단 인터페이스가 복잡하면 피드백 품질이 떨어진다.

```python
print("10개 에피소드를 확인하셨습니까?")
print("[1] OK — 다음 Phase로 이동")
print("[2] 코멘트 있음 — 시연 및 수정 진행")

choice = input("선택 (1 or 2): ").strip()

if choice == "1":
    generate_handoff_package()   # 루프 종료 → 핸드오프 패키지 생성
elif choice == "2":
    run_expert_demonstration()   # 클릭 시연 → IRL → 재학습
```

### 19.3 step1/feedback/ 폴더 구조 (신규 생성)

Phase Graduate 시점에 자동으로 구동되는 피드백 루프 프로그램 모음.

```
step1/
├── env/
│   └── pipe_routing_env_step1.py       # RL 환경
├── train/
│   └── train_step1.py                  # 학습 파이프라인
├── feedback/                            # ★ 신규 생성
│   ├── 01_visualize_episodes.ipynb     # 성공 에피소드 10개 3D 시각화
│   ├── 02_expert_demonstration.ipynb   # 숙련공 클릭 시연 + OK/코멘트 판단
│   ├── 03_irl_reward_update.ipynb      # IRL 역산 + reward_config.json 업데이트
│   └── run_feedback_loop.ipynb         # 01→02→03 순서대로 실행하는 마스터 노트북
└── outputs/
    ├── best_model.pt
    ├── reward_config.json               # 피드백 루프마다 버전 업
    └── expert_episodes/
        └── EXP_001.json                 # 숙련공 시연 저장
```

### 19.4 각 노트북 역할

| 노트북 | 역할 | 핵심 출력 |
|--------|------|----------|
| `01_visualize_episodes.ipynb` | Graduated Model로 성공 에피소드 10개 추출 및 3D 시각화 | HTML 뷰어 (Colab 인라인) |
| `02_expert_demonstration.ipynb` | 숙련공 OK/코멘트 판단 + 클릭 시연 | `expert_episode.json` 또는 OK 신호 |
| `03_irl_reward_update.ipynb` | IRL 역산 → reward 변경 제안 → 숙련공 승인 → 재학습 | 업데이트된 `reward_config.json` |
| `run_feedback_loop.ipynb` | 위 3개를 순서대로 실행, 루프 제어 | Phase Graduate 또는 재학습 트리거 |

### 19.5 run_feedback_loop 마스터 루프

```python
while True:
    # 01: 성공 에피소드 10개 시각화
    episodes = extract_success_episodes(model, n=10)
    run_visualization(episodes)

    # 02: 숙련공 판단 (단순 2분기)
    choice = input("선택 (1=OK / 2=코멘트): ").strip()

    if choice == "1":
        print(f"Phase {current_phase} OK — 다음 Phase로 이동")
        generate_handoff_package(model, current_phase)
        break  # 루프 종료

    # 03: 클릭 시연 → IRL → reward 업데이트 → 재학습
    expert_path = run_expert_demonstration(episodes)
    proposal    = run_irl_analysis(expert_path, current_reward_config)
    new_config  = apply_reward_update(proposal, expert_approved=True)
    model       = retrain(model, new_config)
```

### 19.6 피드백 루프 발동 시점

| 시점 | 조건 | 동작 |
|------|------|------|
| Phase 1-1 Graduate | (KPI ≥ 95%) 달성 | `run_feedback_loop` 구동 |
| Phase 1-2 Graduate | (KPI ≥ 90%) 달성 | `run_feedback_loop` 구동 |
| Phase 1-3 Graduate | (KPI ≥ 85%) 달성 | `run_feedback_loop` 구동 |
| Phase 1-4 Graduate | (KPI ≥ 80%) 달성 | `run_feedback_loop` 구동 |
| Phase 1-5 Graduate | (KPI ≥ 80%) 달성 | `run_feedback_loop` 구동 → 평가 OK 시 Step 2 핸드오프 |

---

## 20. step1 파일 구조 확정

> **⚠️ 본 §20 전체는 v2 spec 에서 폐기됨 (2026-06-24 r1 표시).**
>
> 폐기 사유: 본 절의 step1 파일 구조 (notebooks/main_step1.ipynb + feedback/ 노트북 + IRL 셀 A~E) 는 §16 숙련공 피드백 루프 메커니즘에 종속. §16, §17, §19 폐기로 본 구조 전체 무효.
>
> 대체 위치: `CLAUDE.md` SKILL §5 졸업판 폴더 구조 (envs/ + shaping/ + generators/ + scenarios/ + autoresearch/ + training/ + tests/ + handoffs/). 노트북 중심 → 모듈 분리 코드 베이스로 전환.
>
> 본문은 역사적 사실로 보존 (v1 시절 Colab 노트북 워크플로우 archive).

### 20.1 전체 파일 구조

```
step1/
├── step1_main.ipynb               ← 기존 노트북 (셀 추가)
│
├── feedback/                       ← 신규 폴더 (.py 함수 모음)
│   ├── visualize.py               # 에피소드 추출 및 3D 시각화 함수
│   ├── expert_demo.py             # 숙련공 클릭 시연 함수
│   └── irl_reward.py              # IRL 역산 + reward 업데이트 함수
│
└── outputs/                        ← 신규 폴더 (산출물)
    ├── best_model.pt
    ├── reward_config.json
    ├── expert_episodes/
    │   └── EXP_001.json
    └── (핸드오프 패키지 10개 파일)
```

### 20.2 설계 원칙

- **노트북은 1개만** — `step1_main.ipynb` 하나에서 학습부터 피드백 루프까지 전부 제어
- **로직은 .py로** — feedback/ 폴더의 .py 파일에 함수로 구현, 노트북은 import해서 호출만
- **여러 노트북 동시 구동 금지** — 숙련공 혼란 방지, 노트북 1개 위에서 아래로 순서 실행

### 20.3 기존 노트북에 추가되는 셀 구조

기존 학습/평가 셀 뒤에 아래 셀들이 순서대로 추가된다.

```python
# ─────────────────────────────────────────
# [추가 셀 A] Phase Graduate 감지 → 피드백 루프 진입
# ─────────────────────────────────────────
from feedback.visualize  import extract_success_episodes, run_visualization
from feedback.expert_demo import run_expert_demonstration
from feedback.irl_reward  import run_irl_analysis, apply_reward_update

if kpi >= config["promotion_threshold"]:
    print(f"✅ Phase {current_phase} Graduate (KPI={kpi:.1f}%) — 피드백 루프 시작")
    feedback_loop_active = True
else:
    feedback_loop_active = False

# ─────────────────────────────────────────
# [추가 셀 B] 성공 에피소드 10개 추출 및 시각화
# ─────────────────────────────────────────
# ※ 피드백 루프 재진입 시 이 셀부터 다시 실행
if feedback_loop_active:
    episodes = extract_success_episodes(model, env, n=10)
    run_visualization(episodes=episodes, phase=current_phase)

# ─────────────────────────────────────────
# [추가 셀 C] 숙련공 판단 (OK / 코멘트) — 2분기만
# ─────────────────────────────────────────
if feedback_loop_active:
    print("\n10개 에피소드를 확인하셨습니까?")
    print("[1] OK — 다음 Phase로 이동")
    print("[2] 코멘트 있음 — 시연 및 수정 진행")
    choice = input("선택 (1 or 2): ").strip()

    if choice == "1":
        feedback_loop_active = False   # 루프 종료 → 셀 E로
    # choice == "2" 이면 셀 D 실행

# ─────────────────────────────────────────
# [추가 셀 D] 클릭 시연 → IRL 역산 → reward 업데이트 → 재학습
# (choice == "2" 일 때만 실행)
# ─────────────────────────────────────────
if feedback_loop_active:
    # 클릭 시연
    expert_path = run_expert_demonstration(episodes, current_phase)

    # IRL 역산 + 숙련공 승인
    proposal = run_irl_analysis(expert_path, episodes[0], current_reward_config)
    print_proposal(proposal)
    approved = input("위 변경안을 적용하시겠습니까? (yes/no): ").strip()

    if approved == "yes":
        current_reward_config = apply_reward_update(
            proposal, current_reward_config, expert_approved=True
        )
        # 재학습 (Graduated Model에서 이어서)
        env = PipeRoutingEnvStep1(reward_config=current_reward_config)
        model = MaskablePPO.load("outputs/best_model.pt", env=env)
        model.learn(total_timesteps=FEEDBACK_RETRAIN_STEPS,
                    reset_num_timesteps=False)
        print("재학습 완료 — 셀 B부터 다시 실행하세요")

# ─────────────────────────────────────────
# [추가 셀 E] OK 확정 → 핸드오프 패키지 생성 → 다음 Phase
# (choice == "1" 일 때만 실행)
# ─────────────────────────────────────────
if not feedback_loop_active:
    generate_handoff_package(model, current_phase)
    print(f"✅ Phase {current_phase} 핸드오프 패키지 생성 완료")
    print(f"➡️  다음 Phase: {next_phase}")
```

### 20.4 각 .py 파일 역할

| 파일 | 포함 함수 | 역할 |
|------|----------|------|
| `feedback/visualize.py` | `extract_success_episodes()`, `run_visualization()` | 성공 에피소드 추출 및 3D HTML 뷰어 출력 |
| `feedback/expert_demo.py` | `run_expert_demonstration()`, `save_expert_episode()` | 숙련공 클릭 시연 인터페이스 및 저장 |
| `feedback/irl_reward.py` | `run_irl_analysis()`, `apply_reward_update()`, `validate_reward_proposal()` | IRL 역산, reward 거버넌스, config 업데이트 |

### 20.5 재학습 시 노트북 실행 규칙

```
최초 실행:  셀 A → B → C → D → E  (순서대로)
재학습 시:  셀 B부터 다시 실행      (셀 A 재실행 불필요)
OK 확정:   셀 E만 실행             (핸드오프 패키지 생성)
```

> ⚠️ 재학습 후 반드시 셀 B(시각화)부터 다시 실행해야 한다.
> 셀 D(재학습) 완료 후 곧바로 셀 E를 실행하면 숙련공 검증 없이 Graduate되는 버그 발생.

---

## 부록. r1 변경 작업 요약 (2026-06-24)

본 r1 revision 에서 추가된 폐기 안내 박스 총 **7곳** (§ 헤더 직후):

| § | 폐기 안내 추가 위치 | 대체 위치 |
|---|---------------------|----------|
| §8.2 | 추상 비용 모델 | `CLAUDE.md` §16.3 (물리 단위) |
| §9   | 공통 KPI | `CLAUDE.md` §11 (Layer 1/2) |
| §12.1 | Hidden Dim Projection | `CLAUDE.md` SKILL §3.1 (256 고정) |
| §16  | 숙련공 피드백 루프 | `CLAUDE.md` §16 (autoresearch) |
| §17  | IRL 전략 | `CLAUDE.md` §16.7 (Dense Reward) |
| §19  | 피드백 루프 종료 조건 | `CLAUDE.md` §16.6 (autoresearch 종료) |
| §20  | step1 파일 구조 | `CLAUDE.md` SKILL §5 (졸업판 폴더 구조) |

**원본 (r0) 본문은 모두 보존**. 본 revision 의 목적은 폐기 영역을 명시화하여 6개월 후 또는 새 협업자가 본 reference 를 읽을 때 혼란을 방지하는 것.

**향후 r2 가 필요한 시점**: 새로운 폐기 영역 발생 시 (예: Phase 1 결과 후 §7 밸브 spec 의 일부가 추가 폐기되는 경우 등). r2 작성 시에도 본 r1 의 변경 이력 + 원본 보존 원칙 유지.

**문서 버전:** r1 (2026-06-24)
**원본:** r0 (`common_spec.md`, v1 시절 작성)
**상위 spec 참조:** `CLAUDE.md` v1.0 (2026-06-24 졸업판), `SKILL.md` v1.0, `PROGRESS.md` v1.0