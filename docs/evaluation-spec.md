# 평가 체계 참조 문서

> CLAUDE.md에서 분리된 참조 문서. 평가 시나리오 구성(4분할/난이도/seed range), Procedural Generator 알고리즘, Step별 평가 metric을 구현하거나 참조할 때 로드한다.

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

