# 파이프 공학 스펙 참조 문서

> CLAUDE.md에서 분리된 참조 문서. 파이프 타입/사이즈/벤딩/중력관/밸브/서포트 물리 스펙과 입출력 JSON 형식을 구현할 때 로드한다.
> 본문 서술("Step N 이후 적용" 등)의 Step 기준점은 CLAUDE.md §2 아키텍처를 따른다.

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

