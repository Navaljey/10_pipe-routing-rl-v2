# PROGRESS.md — 의사결정 이력 및 근거 보존

> 본 문서는 파이프 자동배치 강화학습 v2 프로젝트의 **WHY** 를 기록한다.
> CLAUDE.md (WHAT) 와 SKILL.md (HOW) 의 결정 근거가 흩어지지 않도록 시간순으로 보존한다.
> 다음 세션에서 "왜 이렇게 결정했지?" 라고 헷갈릴 때 본 문서를 참조한다.
>
> **🎓 2026-06-24 졸업**: 본 archive 는 `_ing` 시스템에서 졸업했다 (`PROGRESS_ing.md` → `PROGRESS.md`).
> 졸업 전 archive entry 안의 cross-reference 도 새 파일명으로 일괄 갱신됨 (역사적 정확성은 git history 로 추적 가능).

---

## 0. 본 문서 사용법

```
새 Lock 해제 / 결정이 발생하면:
  1. 해당 Session에 결정 사항 기록
  2. 기각된 옵션은 "기각된 대안" 에 보존 (나중에 재검토 가능)
  3. 결정의 근거 (왜) 명시
  4. 다음 세션 진입점 명시
```

---

## Session 2026-04-30 — 프로젝트 v2 구조 설계 (전체 골격)

### 세션 배경

기존 pipe-routing-rl 프로젝트(common_spec.md, Step 1~10 단일 에이전트 curriculum)에서 **Phase 1부터 결과 품질이 만족스럽지 않은 상황**.

> 사용자 표현: "내가 원하는 결과물은 숙련된 engineer라면 저렇게 하지 않을 거라는 거지."

이 문제를 해결하기 위해 reward tuning이나 hyperparameter 조정이 아닌 **구조적 개선**을 모색.

---

### 핵심 의사결정 1 — 숙련공 시연 = 정답 채택 폐기

#### 배경

- 처음에는 RLHF (숙련공 피드백) 을 고려
- 기존 common_spec.md 16~17장에 숙련공 클릭 시연 + IRL 역산 체계 존재

#### 결론

**숙련공 시연을 정답으로 삼는 방식 전면 폐기**

#### 근거 (사용자 통찰)

```
"분기점, 밸브, 서포트에 대한 정답은 숙련엔지니어마다도 실제로 다를 수 있어"
```

이 통찰이 결정적이었음. 시연을 정답으로 삼으면:
1. 시연자 A와 B가 다른 답을 주면 어느 것이 정답인지 결정 불가
2. RLHF는 본질적으로 지도학습 → 시연이 천장(ceiling)이 됨
3. 본 프로젝트의 목적인 "숙련 엔지니어를 뛰어넘는다" 와 정면 충돌

#### 채택된 대안

**물리적 측정 가능 metric 기반 평가**

- Step 1~3: A* benchmark 와 비교
- Step 4~6: 물리 metric (분기 길이 / 밸브 제약 / 총 중량)
- Step 7~10: 다중 파이프 metric (§99-Locked)

#### 부수 효과

- common_spec.md 16~17장 (숙련공 피드백 루프) 폐기
- 19~20장 (피드백 루프 인프라) 폐기
- 9장 (KPI 정의) 새 평가 체계로 대체

---

### 핵심 의사결정 2 — A* 의 위치 = 평가 도구이지 학습 신호 아님

#### 배경

- 사용자가 "Micro RL 학습을 위한 모범답안으로 A*를 활용" 아이디어 제시
- Claude가 처음에는 이를 reward 신호로 쓰는 것으로 오해 → 우려 표명

#### 결론

A* 는 **학습 시점에는 절대 사용하지 않음**.
**평가 시점에만** RL 모델의 output 과 비교 baseline으로 사용.

#### 근거 (사용자 정정)

```
"학습 evaluation시 reward의 총량이 아닌 A*의 route와 RL의 route를 비교한다는 게 내 생각이야."
```

이게 표준 ML 관행 (학습 loss와 평가 metric의 분리). 그리고 이 방식은:

1. **Reward hacking 탐지** 가능 (총량은 높은데 실제 경로가 이상한 케이스)
2. **서로 다른 reward 함수 비교** 가능 (공정한 외부 기준)
3. **편향(bias) 식별** 가능 (어떤 reward가 어떤 trade-off에 갇혔는지)

#### 기각된 대안 1: A* 를 reward 항목에 포함

```
reward = ... - α × distance(rl_path, astar_path)
```

기각 이유: RL 이 A* 를 모방하게 되어 **A* 가 천장이 됨**. 사용자의 원래 목적과 충돌.

#### 기각된 대안 2: A* 경로를 BC (Behavioral Cloning) 초기값으로 사용

기각 이유: 일부 가치 있으나 A* 편향이 잔존. 사용자가 "evaluation 도구로만" 명시하여 폐기.

#### 한계 인식

A* 만으로 모든 step을 평가할 수 없음:

| Step | A* 적용성 | 이유 |
|------|:--------:|------|
| 1 | ★★★★★ | 단일 cost (length) |
| 2 | ★★★★☆ | bend 가중치 자의성 → Pareto front로 해결 |
| 3 | ★★★★☆ | 구배는 hard constraint, A* 자연 반영 |
| 4 | ★★★☆☆ | 분기점 위치는 본질적 multi-objective |
| 5 | ★★☆☆☆ | 밸브 위치는 추상 개념 (사람 접근성) |
| 6 | ★★☆☆☆ | 서포트는 경로 전체 평가, A* local cost와 안 맞음 |

→ Step 4~6 은 **물리 metric** 사용 (의사결정 4 참조)

---

### 핵심 의사결정 3 — Hierarchical 구조 (Macro + Micro)

#### 배경

- 단일 에이전트 curriculum (Step 1~10)이 Phase 1부터 한계
- "global하게 본다", "전략 후 실행" 의 엔지니어 사고 모방 필요

#### 결론

**Macro + Micro 2-Level 계층 구조 채택**

```
Macro: A* 후보 K개 중 1개 선택 (전략적 경로 선택)
Micro: 선택된 waypoint 구간을 셀 단위로 채움
```

#### 근거

1. **Phase 1 실패 원인 = local 결정만 하는 구조**
   - 현재 Micro는 SDF/Raycast로 근방만 감지
   - 전체 공간 전략 부재 → "막히면 돌아가는" 반응적 행동
   - 엔지니어는 "전략 먼저, 실행 나중" 사고

2. **Sparse reward 문제 구조적 완화**
   - 200~500 step → 20~50 step 구간들로 분해
   - 각 구간 완료 시 즉시 reward

3. **Step 7~10 (다중 파이프) 자연스러운 흡수**
   - Macro가 "공간 자원 배분" 역할 → 다중 파이프 문제의 본질

4. **디버깅 / 개선 용이**
   - 문제가 Macro/Micro 어느 층위인지 즉시 분리
   - autoresearch가 각 층위 독립 튜닝 가능

#### 기각된 대안 1: Generative 접근 (Diffusion/Transformer)

기각 이유: 학습 데이터 (좋은 라우팅 예시) 가 부족. 사용자 표현:
```
"숙련된 엔지니어가 한 예시들도 결국에는 시간에 쫓겨서 충분하다고 할 수는 없어"
```

#### 기각된 대안 2: Graph 기반 표현만 (Macro 없이)

기각 이유: Graph 표현은 Macro 의 일부 (visibility graph) 로 흡수됨.
Macro 없이 graph만 쓰면 전략 결정 주체 부재.

---

### 핵심 의사결정 4 — Step 4~6 평가 = 물리 metric

#### 배경

- A* 가 multi-objective 문제에서 한계
- Claude가 "숙련공 시연을 부분 활용" 제안 → 사용자가 정면 거절

#### 결론 (사용자 주도 결정)

각 step 별 **물리적 측정 metric** 사용:

| Step | Metric |
|------|--------|
| Step 4 (분기) | 분기 여부 + main+branch 전체 길이 최소 |
| Step 5 (밸브) | 밸브 설치 여부 + 높이 (700~1500mm) + 전면 1m³ + 길이 최소 |
| Step 6 (서포트) | 파이프 + 서포트 **총 중량 (kg)** |

#### 근거 (사용자 통찰)

```
"다른 evaluation 지표를 찾는 것이 더 현실적이면서도 물리적 현실에 가까운 것이 아닌가"
```

이 metric 들의 우수성:

1. **객관성**: 측정 가능, 일관됨 (사람마다 다르지 않음)
2. **autoresearch 친화성**: 자동 평가 가능 (라벨링 불필요)
3. **확장성**: 시나리오만 있으면 됨 (시연 수집 불필요)
4. **물리적 단위**: 추상 비용 가중치 자의성 제거 (예: 서포트 = kg)

#### Claude의 평가

이 결정은 **Claude의 제안보다 더 좋음**. 솔직하게 인정:
- Claude 제안: 숙련공 시연 부분 활용
- 사용자 제안: 물리 metric
- 사용자 안이 우월한 이유: 시연 의존성 제거, 자동화 가능

#### 부수 결정 — 서포트 비용 모델 변경

common_spec.md 8.2 의 추상 비용 모델 (벽 1.0, 행거 1.5 등) 폐기.
**파이프 + 서포트 실제 중량 (kg)** 으로 대체.

근거: kg 은 물리적 실체. 추상 가중치보다 의미 명확.

---

### 핵심 의사결정 5 — autoresearch 운영 방침

#### 배경

- 사용자가 기존 프로젝트에서 Karpathy autoresearch 적용 경험
- 본 프로젝트에서는 **각 step별 reward 최적화 도구**로 활용

#### 결론

| 항목 | 결정 |
|------|------|
| Reward 가중치 변형 | 자유 (Claude API 판단) |
| Reward 항목 추가/제거 | 가능, 단 4-Gate 자동 검증 통과 |
| Hyperparameter 변형 | 자유 |
| 네트워크 구조 변경 | Case by case (안전/금지/조건부 영역 분류) |
| 동시 실행 | **한 번에 한 인자만** (ablation 원칙) |
| 학습 시간 | 2-Stage Screening |
| 변형 제안 방식 | Bayesian Optimization + Claude API |

#### 근거 — "한 번에 한 인자만"

사용자 통찰:
```
"한 번에 여러 개를 하면 어떤 인자가 효과를 발휘했는지 알 수 있나?"
```

이는 **ablation study** 원칙과 일치. 인과관계 추적의 정공법.

다만 Claude가 보강:
- 같은 인자의 다른 값 동시 학습은 허용 (예: w_length=[0.5, 1.0, 1.5, 2.0])
- 인자 간 interaction 효과는 sequential sweep 후 local grid search 로 포착

#### 근거 — 4-Gate 자동 검증

새 reward 항목 발명을 허용하되, **명백한 헛소리만 거름**:

```
Gate 1: 측정 가능한가 (격자 좌표로 계산)
Gate 2: 기존 항목과 충돌 안 하는가
Gate 3: 평가 metric 정렬되는가
Gate 4: 부호 일관성 있는가
```

게이트 통과 != 효과 보장. 학습 후 평가 결과로 자동 폐기.

#### 근거 — 2-Stage Screening

학습 시간 현실 계산:
```
변형 20개 × 정상 학습 4시간 = 80시간 (3.3일)
Step 1~6 누적 = 약 20일/라운드
→ 비현실적
```

해결:
```
Stage 1 (짧은 학습): 변형 N개 모두 → 하위 50% 제거
Stage 2 (정상 학습): 살아남은 후보만 → 1개 채택
```

#### 기각된 대안: 동시 다인자 변형

기각 이유: 인과관계 추적 불가능. 어느 인자가 효과 냈는지 모름.

---

### 핵심 의사결정 6 — 기존 spec 처리

#### 결론

```
common_spec.md → reference 로 보존
새 CLAUDE.md → 자체 완결 문서로 작성
```

폐기: 16, 17, 19, 20장 (숙련공 피드백 인프라)
대체: 9장 (KPI), 12장 (전이학습 - Hierarchical 반영), 13장 (핸드오프 - Macro 추가)
직접 계승: 3, 4, 5, 6, 7, 8, 11, 14, 15, 18장

#### 근거 (사용자 지시)

```
"계승은 계승해서 작성, 폐기는 그냥 미작성, 새정의는 새정의해서 작성"
"chapter mapping 안 적어도 추적 가능"
```

---

### 핵심 의사결정 7 — Step 6 → 7 전환 프로토콜

#### 결론

3-stage 전환:
```
Stage 1 (0~30%): Pure Freeze - Micro 동결, Macro 학습
Stage 2 (30~70%): Selective Unfreeze - Micro 마지막 layer만, lr 1/10
Stage 3 (70~100%): 안정화 후 fine-tune 비율 점진 증가
```

매 N 에피소드마다 **Step 1~6 회귀 검증**, 회귀 시 Stage 1로 rollback.

#### 근거

Hierarchical RL의 함정:
1. **Pure Joint 학습**: non-stationarity 문제 (Micro 변하면 Macro 입장에서 환경 계속 바뀜)
2. **Pure Freeze**: 다중 파이프 양보 학습 불가 (Step 1~6은 단일 파이프 환경이라 다른 파이프 = 정적 장애물 인식)

해결: **단계적 unfreeze + 회귀 검증**

#### 기각된 대안: Adapter (LoRA) 패턴

기각 이유: 구현 복잡도. **현 단계에서는 불필요**. 만약 selective unfreeze 로도 안 되면 그때 도입 검토.

---

### 12개 Lock 항목 식별

본 세션 종료 시점 기준 미해결 항목:

```
[Critical - 다음 세션 우선]
L-A3': Step 1~3 평가 임계값 구체값
L-A5: 평가 시나리오 구성
L-16.3-w: baseline reward 가중치 초기값/sweep 범위

[Important]
L-A4: Step 4~6 hack 방지 hard constraint 세부
L-B1: Macro Observation/Action Space 구체 정의
L-B2: Macro Reward 설계
L-B3: 다중 파이프 양보 학습 방식
L-B4: Macro 학습용 시나리오 정의
L-12.2: Macro 아키텍처 결정 (GNN vs Transformer vs MLP)
L-17.1: Macro K값 및 다양성 threshold

[Infrastructure]
L-C2: autoresearch 메타 구조 (스크리닝 비율, 동시 학습 수, 종료 조건)
L-C4: grid 좌표계 ↔ graph 좌표계 통합
```

---

### 본 세션의 작업 산출물

```
/mnt/user-data/outputs/
├── CLAUDE.md   (31KB, 프로젝트 spec 본체)
├── SKILL.md    (8.7KB, Claude 행동 규칙 + Hard Gate)
└── PROGRESS.md (본 문서)
```

---

### 다음 세션 진입점

```
1. 본 PROGRESS.md + CLAUDE.md + SKILL.md 첨부
2. 권장 시작 멘트:
   "지난번 결정 사항 확인 후 L-XXX 부터 풀자"
3. 권장 우선순위:
   L-A3' → L-A5 → L-16.3-w (Step 1 학습 시작 조건 충족)
```

---

### 본 세션의 사용자 통찰 모음 (다시 읽을 가치)

> "내가 원하는 결과물은 숙련된 engineer라면 저렇게 하지 않을 거라는 거지."

> "현 시점 autoresearch는 방법은 맞는데, Pipe routing RL의 요소 개선이 아니라, 구조상에서의 개선이 맞는 것이 아닐까"

> "숙련된 엔지니어가 한 예시들도 결국에는 시간에 쫓겨서 충분하다고 할 수는 없어."

> "macro에서 다른 pipe 배치까지 고려하여 waypoints를 정해주는 거 아냐?"

> "Micro 학습을 위한 모범답안으로 A* 알고리즘을 생각했어"

> "학습 evaluation시 reward의 총량이 아닌 A*의 route와 RL의 route를 비교한다는 게 내 생각이야."

> "분기점, 밸브, 서포트에 대한 정답은 숙련엔지니어마다도 실제로 다를 수 있어"

> "다른 evaluation 지표를 찾는 것이 더 현실적이면서도 물리적 현실에 가까운 것이 아닌가"

> "한 번에 여러 개를 하면 어떤 인자가 효과를 발휘했는지 알 수 있나?"

> "오늘은 여기까지 공부한다고 생각을 하면 그 날 한 부분을 .md 파일에 충분히 논점들을 모두 정리해서 입력하고 (풀어야 할 부분도 명기) 다음에 다시 그 파일을 기준으로 clear되지 않은 부분을 풀어내는 게 맞는 거 같아."

> "그냥 vibe coding을 시작하지 않도록 하자."

이 통찰들이 본 프로젝트의 방향을 결정함. 기억할 것.

---

## Session 2026-05-14 — APR 구조 재검토 및 Hierarchical 폐기 결정

### 세션 배경

2026-04-30 세션에서 합의된 Hierarchical (Macro+Micro 별도 네트워크) 구조에 대해 사용자가 확신이 들지 않아 재검토 요청.

> 사용자 표현: "ing파일에 있는 lock을 푸는 것보다 계층구조에 대하여 확신이 안 들고 있슴. 계속 여러가지 질문을 하면서 복합적으로 생각하고 결정할려고 하는중. 현재 검토사항으로는 굳이 macro, micro 나눌것이 아니라 전이학습으로 전체를 구성완료 하는것이 맞다고 봄."

본 세션은 Lock 해제 세션이 아니라 **상위 의사결정 (의사결정 3, 7) 재검토 세션** 임. 그 과정에서 평가 체계, 도구 도입, 신규 기법 도입까지 함께 검토됨.

---

### 핵심 의사결정 3-r1 — Hierarchical 별도 네트워크 구조 폐기

#### 배경 및 재진단

2026-04-30 의사결정 3 당시 Phase 1 실패 원인을 다음과 같이 진단:

> "Phase 1 실패 원인 = local 결정만 하는 구조 (global view 부재)"

본 세션에서 사용자에게 Phase 1 실패 양상을 직접 확인한 결과:

| 질문 | 사용자 답변 |
|------|------------|
| 실패 양상은? | **비효율 경로** (갇힘/발산 아님) |
| Local 잘 됐다면 Global도 잘 됐을지? | **"LOCAL VIEW단계에서 잘되었다면 GLOBAL VIEW도 잘 되었을것으로 여겨짐. LOCAL VIEW에서 부터 만족되지 않음."** |
| Observation/Reward/Action 어디를 만졌나? | autoresearch로 전 분야 조금씩, 개선은 됐으나 만족 미달 |

이 답변이 **결정적**. 의사결정 3의 진단이 오진(誤診)임이 확인됨:

- 원래 진단: "Local은 잘 되는데 Global이 부재" → Hierarchical 처방 정당
- 실제: "Local부터 부실" → Hierarchical은 잘못된 처방

Hierarchical RL의 표준 가정 ("low-level이 reasonably well 학습된 후 high-level이 조립") 이 깨진 상태이므로 Macro+Micro 별도 네트워크 도입의 정당성이 약함.

#### 전이학습 관점 추가 검토

사용자 질문: "Step 6→7로 넘어가는 단계에서 Hierarchical이 전이학습에 도움이 될까?"

분석 결과:

| 항목 | Hierarchical의 효과 |
|------|------------------|
| 단일 파이프 능력 보존 | ✅ Micro freeze로 구조적 보장 |
| 다중 파이프 신규 능력 학습 | ❌ 어차피 새로 학습해야 함 |
| 양보(yielding) 행동 학습 | ❌ Micro freeze가 막음, Selective Unfreeze는 non-stationarity 불안정 |

→ Hierarchical의 진짜 이득은 **"Step 1~6 능력 보존"** 한 가지뿐.
→ 그것조차 단일 에이전트 + 다른 기법 (mixed replay, EWC, LoRA Adapter 등) 으로 대체 가능.

#### 결론

**Hierarchical (Macro+Micro 별도 네트워크) 구조 폐기.**

#### 기각된 대안

- **별도 네트워크 유지 + non-stationarity 해법 추가 연구**: 비용 대비 효과 부정적
- **현재 spec 그대로 진행**: Phase 1 실패 진단 오류를 그대로 안고 가는 것 → 같은 실패 재발 가능

#### 부수 결정

의사결정 7 (Step 6→7 전환 프로토콜 3-stage) 도 자동 폐기. Macro 자체가 없으므로 전환 의식이 사라짐. 대신 Step 간 전이학습은 의사결정 3-r1 후속 구체화 필요 (L-D1 항목 신규).

---

### 핵심 의사결정 3-r2 — 단일 에이전트 + 전이학습 방향

#### 결론

Step 1~10 전체를 **단일 에이전트 구조**로 진행. Step 간은 전이학습으로 연결.

#### 전이학습 전략 후보 (구체 선택은 L-D1 Lock)

| 전략 | 설명 | 평가 |
|------|------|------|
| A. 순차적 fine-tuning | Step N 정책으로 Step N+1 초기화 | 단순, catastrophic forgetting 위험 |
| B. Mixed curriculum | 매 iteration에 Step 1~N 시나리오 일정 비율 혼합 | forgetting 방지 강력, 비용 증가 |
| C. EWC regularization | 중요 가중치 보호 정규화 | 우아하나 RL에서 까다로움 |
| D. LoRA Adapter | Backbone freeze + 작은 adapter | 구현 명확, 표현력 제약 |

#### 근거

- Phase 1 실패가 "Local 부실"이므로 단일 에이전트의 Micro 능력 자체를 더 끌어올리는 것이 우선
- Step 1~10의 본질적 변화가 작음 (관측/행동 공간 150-dim/7-direction 고정, reward 항목만 누적)
- 단일 에이전트가 디버깅/평가/autoresearch 모두 단순

#### v2 spec에 미치는 영향

- CLAUDE.md §2 (전체 아키텍처) 전면 재작성 필요
- §12.2 (Step 6→7 전환), §12.3 (Macro 신규 정의) 폐기
- §13.2 (Step 6→7 핸드오프), §13.3 (Macro 핸드오프) 폐기
- §17 전체 (Macro 학습 환경) 폐기
- §99 Lock 항목 중 Macro 관련 모두 폐기: L-B1, L-B2, L-B3, L-B4, L-12.2, L-17.1

---

### 핵심 의사결정 8 — Step 4~6 평가의 2-Layer 분리

#### 배경

사용자 질문: "phase1의 결과가 나쁜것은 a*알고리즘과 비교하면서 결정하면 되는데, phase 2부터는 a*알고리즘이라는 정답지가 없는데 어떻게 해야할지 고민됨."

#### 본질적 인식

Step 4 이후 (분기/밸브/서포트) 는 **절대 최적의 정의가 수학적으로 불가능**한 영역. 분기점/밸브/서포트의 정답은 multi-objective + 도메인 판단의 영역. 의사결정 1에서 합의한 "숙련공 시연 = 정답 금지" 원칙과 일관되게, **절대 정답이라는 개념 자체를 포기**.

#### 결론

Step 4~6 평가를 두 층으로 분리:

```
Layer 1. Hard Constraint 충족 여부 (Binary)
  - 분기점이 valid 위치인가
  - 밸브 높이가 700~1500mm 범위인가
  - 서포트 간격이 §8.1 max 이내인가
  → 통과/실패의 binary 판정

Layer 2. 정량적 Quality (Relative)
  - 물리 metric 절대값 (kg, mm 등)
  - 가능한 경우 Multi-cost A* 대비 비율 (Lower bound estimation)
  - 변형 간 ranking (autoresearch가 자연스럽게 산출)
```

#### 근거

- "절대 최적과의 거리"를 포기하면 평가 체계가 **상대 비교 + 제약 통과**로 단순화
- autoresearch가 본질적으로 변형 간 상대 비교라 자연스러운 정합
- 의사결정 4 (물리 metric 채택) 와 의사결정 1 (시연 정답 거부) 의 자연스러운 연장

#### 보강된 의사결정 4

원래 의사결정 4의 물리 metric (kg, mm) 은 그대로 유지. 다만 **"절대 최적 대비 평가가 불가능함을 명시적으로 수용"** 하는 Layer 분리를 추가.

#### v2 spec에 미치는 영향

- CLAUDE.md §11.5 (Step 4~6 평가) 에 Layer 1/2 명시
- L-A4 (Step 4~6 hack 방지 hard constraint) 의 의미가 Layer 1 정의로 명확화

---

### 핵심 의사결정 9 — wandb 도입 (autoresearch 가시성 확보)

#### 배경

사용자가 autoresearch의 효과에 대한 의문 제기:

> "AUTORESEARCH를 통햇서 전 분야를 조금씩 만지면서 좋아지기는 햇지만, 내가 생각하는 궁극적인 결과보다 아직 못하나 개선이 될것으로 보임"

이 답변이 의미하는 바: 천장(Plateau)에 닿았는지, search space가 좁은지, search strategy가 비효율인지 **현재 진단 불가**. 진단 도구가 없는 상태.

#### 결론

**Weights & Biases (wandb) 즉시 도입.**

#### 도입 이유

| 비교 | TensorBoard | wandb |
|------|-------------|-------|
| 단일 실험 시각화 | ✅ | ✅ |
| 다중 실험 비교 | ❌ 약함 | ✅ 강함 |
| Colab 세션 끊김 내성 | ❌ | ✅ 클라우드 저장 |
| sb3 native 통합 | △ | ✅ callback 1줄 |
| Parallel coordinates plot | ❌ | ✅ |
| 비용 | 무료 | 무료 (개인 plan) |

#### 핵심 기능: Parallel Coordinates Plot

100회 변형 실험을 한 그림에 표시. 각 변형이 하나의 선, 여러 축에 인자 값 배치. final_reward 기준 색칠. 드래그/필터/정렬로 패턴 즉시 인식.

```
효과:
  - "best 10개 변형의 공통 인자 영역" 5초 만에 파악
  - "어떤 인자 영역이 worst와 연관" 즉시 시각화
  - autoresearch의 천장 vs search 부족 진단에 직접 활용
```

#### 추가 검토 (별도 결정 아님)

Optuna importance analysis 도 autoresearch가 Optuna 기반이 될 경우 자연스럽게 따라옴. wandb와 결합 시:

1. Optuna가 "어떤 인자가 중요한지" 정량적 추출
2. wandb parallel coordinates 가 "그 인자의 어떤 영역이 좋은지" 시각화

#### 기각된 대안

- **MLflow**: 범용성 강점이나 RL native 통합 약함
- **Neptune/Comet**: 기능 유사하나 RL 생태계에서 wandb 우세
- **자체 분석 노트북만 유지**: autoresearch 규모 (수백~수천 회) 에 부담 큼

#### v2 spec에 미치는 영향

- SKILL.md §3 (코딩 규칙) 에 wandb 통합 의무 명시
- autoresearch 메타 구조 (L-C2) 설계 시 wandb 전제

---

### 핵심 의사결정 10 — Dense Reward (Potential-Based Shaping) 도입

#### 배경

사용자가 Tan & Mu (2024) "Hierarchical Reinforcement Learning for Chip-Macro Placement in Integrated Circuit" (Pattern Recognition Letters) 논문의 dense reward 및 options 기법 적용 가능성 제기.

#### 논문 핵심 (HRLP)

- **Sparse reward, large search space, unstable training** 문제 해결 목표
- 두 가지 기여:
  1. Hierarchical RL framework with options (sub-task 추상화)
  2. **Dense reward**: 각 episode마다 local wirelength와 congestion의 difference 계산
- Wirelength 최대 14.25% 감소

#### 본 프로젝트 매핑

| 차원 | HRLP | 본 프로젝트 |
|------|------|------------|
| 문제 본질 | 객체 배치 (one-shot 시퀀스) | 경로 생성 (sequential cell 이동) |
| Sparse 문제 | 모든 macro 배치 후만 reward | Goal 도달까지만 reward |
| Dense 해법 적용성 | wirelength/congestion 변화량 | goal potential / congestion proxy 변화량 |

#### 결론

**Dense reward (Potential-Based Shaping, PBS 형식) 도입.**

#### 도입 형식

```
[기존 sparse 구조]
- Goal 도달 시: +goal_bonus
- 매 step: -1 (length penalty, 정보량 낮음)
- 충돌 시: -collision_penalty
- 일부 SDF 기반 shaping

[Dense 보강]
매 step에 추가:
  + γ·Φ(s_{t+1}) - Φ(s_t)   ← Potential-Based Shaping (Ng et al. 1999)
  
  Φ(s) 후보:
    - Φ_goal(s) = -distance_to_goal(s)
    - Φ_cong(s) = -local_congestion(s)
    - Φ_future(s) = future_feasibility_estimate(s)
```

#### 핵심 안전장치 (PBS 이론적 보장)

Potential-Based Shaping은 다음 조건 하에서 **optimal policy 보존 (정책 왜곡 없음)** 이 이론적으로 증명됨:

1. Φ는 potential function (state만의 함수, action 의존 금지)
2. Φ(terminal_state) = 0
3. Shaping reward = γ·Φ(s') - Φ(s) 형식

이 조건만 지키면 reward hacking 위험 없이 학습 속도만 빨라짐. autoresearch의 4-Gate 검증과 정합.

#### 근거

- Phase 1 실패 양상 ("Local부터 부실") 의 직접 처방
- 의사결정 3-r1에서 폐기한 Hierarchical 의 dense reward 효과만 흡수 (구조 복잡도 없이)
- HRLP가 chip placement에서 14.25% 개선 달성 → 본 프로젝트에서도 유의미 개선 기대 가능
- 절대 변경 금지 항목 (observation/action) 안 건드림
- 표준 RL 기법 (PBS), 검증된 안전성

#### 기각된 대안

- **HRLP options 원형 (별도 high-level/low-level 네트워크)**: 의사결정 3-r1과 충돌. Hierarchical을 폐기했는데 options 도입은 이름만 다른 동일 구조 도입.
- **단순 cost 변화량 reward (PBS 형식 아님)**: 이론적 보장 없음. reward hacking 위험.

#### 신규 Lock

- **L-D2**: Φ (potential function) 의 구체 정의
  - Φ_goal: distance metric (Manhattan / Euclidean / SDF-aware)?
  - Φ_cong: local_congestion proxy 정의
  - Φ_future: 사용 여부 및 정의
  - 각 Φ의 상대 가중치
  - autoresearch sweep 범위

---

### 핵심 의사결정 11 — Macro-action 도입 검토 (HRLP Options의 가벼운 변형)

#### 배경

의사결정 10에서 HRLP options 원형은 기각 (의사결정 3-r1과 충돌). 그러나 options의 **temporal abstraction 정신**은 단일 네트워크 안에서 macro-action 형태로 차용 가능.

#### 차이 (Hierarchical vs Macro-action)

| 항목 | Hierarchical (별도 네트워크) | Macro-action (단일 네트워크) |
|------|------------------------|------------------------|
| 네트워크 수 | 2개 (Macro + Micro) | 1개 |
| 학습 의식 (freeze/unfreeze) | 필요 | 불필요 |
| Non-stationarity | 큰 문제 | 없음 |
| Action space | 분리 (전략 + 실행) | 확장 (primitive + macro) |
| 구현 복잡도 | 높음 | 중간 |

#### 잠정 결론

**Macro-action 도입 검토 (즉시 도입은 아님).**

#### 도입 형식 (잠정)

```
[현재 action space]
7-direction discrete (±X, ±Y, ±Z, stay)

[macro-action 확장 (잠정)]
기존 7개 + 추가:
  - "현재 방향으로 직선 N칸 진행" (N = obstacle까지 거리)
  - "현재 방향 유지하며 가능한 만큼 진행"

네트워크는 매 decision step에서 primitive action 또는 macro-action 선택.
Macro-action 선택 시 자동으로 여러 primitive step 수행 후 다시 결정.
```

#### 도입 단계 (잠정)

1. **Phase 1**: Dense reward (의사결정 10) 단독 도입. Phase 1 결과 재평가.
2. **Phase 1 후 결정**: Dense reward만으로 Phase 1 천장 돌파 충분하면 macro-action 보류.
3. **Phase 1 천장 확인 시**: Macro-action 도입 검토 본격화.

#### 근거

- SKILL.md §3.1 "Action space 정의 불변, 마스킹 외 변경 불가" 원칙과 부분 충돌. 도입 시 spec 재해석 필요.
- Phase 1에서 dense reward 만으로 충분한 개선 가능성 → macro-action 도입을 일단 보류
- 도입 시 구조 변경이라 자체 검증 필요

#### 신규 Lock

- **L-D3**: Macro-action 구체 설계 (도입 결정 시)
  - 추가 action 종류
  - Termination condition
  - Action mask 처리
  - SKILL §3.1 재해석안

---

### 본 세션에서 검토했으나 채택하지 않은 사항들

#### BMad 프레임워크 도입 검토 → 비채택

사용자가 BMad 프레임워크 적용 여부를 질문. 분석 결과:

- BMad는 **vibe coding의 반대** 개념 (spec-driven framework)
- 본 프로젝트는 이미 `_ing` 구조로 BMad 핵심 가치 (PRD, ADR, Quality Gate) 자체 구축
- BMad의 선형 워크플로우 (PRD→Architecture→Stories) 는 RL의 반복 실험 루프에 부적합
- Colab 분산 환경과의 호환성 약함
- **"좋은 도구가 잘못된 설계를 보완하지 않는다"** — Hierarchical이 잘못된 처방이라면 BMad로 빠르게 만들어도 잘못된 결과

→ 본 프로젝트 구현 단계 진입 시 **Claude Code** 사용 권고. BMad는 부적합.

#### 일반적 RL 프레임워크 도입 검토 → 비채택

사용자가 RL 전용 framework 존재 여부 질문. 분석 결과:

- 계층 1 (알고리즘): sb3-contrib MaskablePPO 이미 spec 고정
- 계층 2 (환경): Gymnasium de facto 표준, 선택 여지 없음
- 계층 3 (실험 추적): **wandb 도입 (의사결정 9)**
- 계층 4 (HP 최적화): autoresearch 자체 구축 + Optuna 보조 가능
- 계층 5 (워크플로우 framework): **RL 전용 산업 표준 없음**

본 프로젝트의 `_ing` 시스템 자체가 RL용 spec-driven framework prototype. 추가 framework 도입 가치 없음.

---

### 본 세션 종료 시점 Lock 항목 정리

#### 폐기된 Lock (Hierarchical 폐기 부수효과)

| 폐기 Lock | 폐기 사유 |
|----------|----------|
| L-B1 (Macro Observation/Action) | Macro 폐기 |
| L-B2 (Macro Reward) | Macro 폐기 |
| L-B3 (다중 파이프 양보 학습 방식) | Macro 의존 항목 |
| L-B4 (Macro 학습용 시나리오) | Macro 폐기 |
| L-12.2 (Macro 아키텍처) | Macro 폐기 |
| L-17.1 (Macro K값/다양성) | Macro 폐기 |

다중 파이프 양보 학습은 **L-D1 (전이학습 전략) + 단일 에이전트 reward 설계** 로 재정의 필요.

#### 유지되는 Lock

| Lock | 상태 | 비고 |
|------|------|------|
| L-A3' (Step 1~3 평가 임계값) | 유지 | Critical, 최우선 |
| L-A4 (Step 4~6 hard constraint) | 유지 | Layer 1 정의로 의미 명확화 |
| L-A5 (평가 시나리오 구성) | 유지 | Critical |
| L-C2 (autoresearch 메타 구조) | 유지 | wandb 도입 전제 추가 |
| L-C4 (좌표계 통합) | 단순화 | Macro 폐기로 grid 좌표계만 처리 |
| L-16.3-w (baseline reward 가중치) | 유지 | Critical |

#### 신규 Lock

| Lock | 내용 |
|------|------|
| L-D1 | 단일 에이전트 전이학습 전략 (A/B/C/D 중 선택 + 구체 메커니즘) |
| L-D2 | Dense reward의 Potential function Φ 구체 정의 |
| L-D3 | Macro-action 구체 설계 (도입 결정 시) |

---

### 다음 세션 진입점

1. 본 PROGRESS.md + CLAUDE.md + SKILL.md + FAILURE_LOG.md 첨부
2. 권장 시작 멘트: "지난 세션 결론들이 머릿속에 묵힌 결과 어떻게 검증되는지 확인"
3. 권장 우선순위:
   - **L-D1 (전이학습 전략)** 우선 — 단일 에이전트 구조의 핵심
   - **L-D2 (Dense reward Φ 정의)** — 즉시 효과 있을 가능성 큰 항목
   - **L-A3' + L-A5 + L-16.3-w** — Phase 1 재학습 시작 조건
   - **L-C2 (autoresearch 메타 + wandb 통합)** — 도구 인프라

---

### 본 세션의 사용자 통찰 모음

> "굳이 macro, micro 나눌것이 아니라 전이학습으로 전체를 구성완료 하는것이 맞다고 봄."

> "LOCAL VIEW단계에서 잘되었다면 GLOBAL VIEW도 잘 되었을것으로 여겨짐. LOCAL VIEW에서 부터 만족되지 않음."

> "phase 2부터는 a*알고리즘이라는 정답지가 없는데 어떻게 해야할디 고민되."

> "Hierarchical Reinforcement Learning for Chip-Macro Placement in Integrated Circuit 2024 논문상에서 말하는 dense reward, option을 반영하면 어떤가?"

> "더이상 논의가 길어지면 초반에 언급된 부분을 재현될거 같어."

마지막 통찰이 핵심: **세션 길이가 길어지면 같은 논의가 반복**됨을 인식하고 마무리. 이것이 \_ing 시스템의 정상 작동 신호.

---

### 본 세션의 Claude 자기 비판

- 2026-04-30 의사결정 3 (Hierarchical 채택) 시 HRLP 같은 도메인 인접 문헌 충분히 참조 안 함
- 만약 그때 HRLP를 알았다면 "Macro+Micro 별도 네트워크" 대신 "Dense reward + Macro-action" 조합을 먼저 제안했을 것
- 사용자가 HRLP 논문을 가져온 것이 본 세션을 한 단계 끌어올림
- 다음 세션 진입 시 새로운 결정을 내릴 때 인접 문헌 사전 검색 강화 필요

---

### 본 세션의 작업 산출물

```
/mnt/user-data/outputs/
├── CLAUDE.md   (Hierarchical 관련 §2/§12.2/§12.3/§13.2/§13.3/§17 폐기 반영)
├── SKILL.md    (Macro 관련 규칙 폐기, 신규 규칙 추가)
├── PROGRESS.md (본 Session 2026-05-14 추가)
└── FAILURE_LOG.md  (첫 entry 추가: Phase 1 비효율 경로 + Hierarchical 오진단)
```

---

## Session 2026-06-11 — Lock 명명 체계 명시화 + 회귀 검증 체계 명시화

### 세션 배경

사용자가 두 가지 질문을 제기:

1. "L-D3와 같이 LOCK의 NO.에서 D가 의미하는게 뭐지? 전체 LOCK에 대해 설명해줘."
2. "커리큘럼학습과 전이학습의 차이점이 뭐지? 두 학습방법의 기본 논리는 기존 모델가중치가 다음 단계에서도 사라지지 않는 개념으로 보이는데, (1) 어떻게 확신하는가? (2) 평가 지표가 있는가? (3) 얼마만큼 학습했을 때 전 단계 가중치가 살아있는지 확인할 수 있는가?"

첫 질문은 spec 문서의 **암묵적 규칙이 문서화되지 않은 결함** 을 드러냄. 두 번째 질문은 spec 의 **회귀 검증 체계가 임계값/지표/주기 측면에서 미완성** 임을 드러냄. 두 질문 모두 본 \_ing 시스템 운영 중 추가 발견된 spec gap.

---

### 핵심 의사결정 12 — Lock 명명 규칙 명시화 (범례 spec 내장)

#### 배경

Lock ID 의 접두사 (L-A / L-B / L-C / L-D / 섹션 직참조) 의 의미가 spec 어디에도 명시 안 됨. 항목 내용과 시간순 추이로 역추론은 가능하나, 6개월 후 자기 자신 또는 새 협업자가 spec 을 읽을 때 혼란 가능.

#### 결론

**CLAUDE.md §99 맨 앞에 "Lock 명명 규칙" 범례 표를 내장.**

```
L-A{n}     : Assessment — 평가 체계 (활성)
L-B{n}     : Block-level / Hierarchical (2026-05-14 전면 폐기)
L-C{n}     : Control / Operations (활성)
L-D{n}     : Dynamics — 단일 에이전트 + 전이학습 구조의 학습 동역학 결정 (2026-05-14 신규)
L-{§n.m}-{tag} : spec 특정 섹션 직참조 (예: L-16.3-w = §16.3 weight)
```

#### 부수 결정

- 폐기된 카테고리 (L-B) 의 ID 재할당 금지. PROGRESS 이력 추적 안정성 확보.
- Lock 범례 single source of truth 는 CLAUDE.md §99 한 곳. SKILL.md / PROGRESS.md / FAILURE_LOG.md 는 참조만 한다.

#### 근거

- spec 문서는 **자기 설명적 (self-documenting)** 이어야 한다 — 외부 사람이 spec 만 읽고도 명명 체계를 이해 가능해야 함.
- _ing 시스템의 핵심 원칙 "암묵적 규칙은 명시적 spec 으로 끌어올린다" 와 정합.

#### Lock 변화

- 신규 Lock 없음 (메타 문서화 결정)
- CLAUDE.md §99 에 범례 표 신설

---

### 핵심 의사결정 13 — 회귀 검증 체계 명시화 (BWT / Forgetting / 주기 감시)

#### 배경

CLAUDE.md §12.3 ("매 N 에피소드마다 Step 1~6 시나리오 재평가, 회귀 시 학습 중단/조정") 과 §13.1 (`stepN_regression_report.json` 핸드오프) 가 회귀 검증의 인프라는 갖추었으나, **다음 세 가지가 모든 step 에서 미정**:

1. **회귀 지표의 정의** — 단순 "성능 떨어지면 회귀" 가 아니라 어떤 metric 으로 어떻게 계산하는지
2. **측정 주기** — 학습 종료 후 1회? 학습 도중 주기적? 주기는 얼마?
3. **트리거 임계값과 대응** — 어느 수준 이하로 떨어지면 무엇을 하는지

사용자 질문 "어떻게 확신하는가 / 지표가 있는가 / 얼마나 학습했을 때 확인 가능한가" 가 이 세 빈 곳을 정확히 가리킴.

#### 결론

**회귀 검증을 "학습 종료 후 1회 확인" 이 아니라 "학습 중 주기적 모니터링" 으로 운영. 세 가지 지표를 모든 step ≥ 2 학습에서 의무 기록.**

```
지표 1 — 각 과거 step 의 평가 metric 직접 측정
  Step 1: success_rate, length_ratio
  Step 2: + bend_count_ratio, ASME compliance
  Step 3: + slope_violation (hard=0 유지)
  Step 4~6: Layer 1 통과율 + Layer 2 절대값

지표 2 — Backward Transfer (BWT)
  BWT_k = perf_k(end_of_stepN) - perf_k(end_of_stepK)
  Step N 학습 후 과거 step k 의 성능이 step k 학습 종료 시점 대비 어떻게 변했는가
  음수=망각, 0=보존, 양수=후속 학습이 과거 도움

지표 3 — Forgetting measure
  F_k = max_{t ≤ now} perf_k(t) - perf_k(now)
  과거 step k 의 역대 최고 성능 대비 현재 격차
```

**측정 주기:** wandb logging 주기와 정합. 매 K timestep (K 는 L-C2 해제 시 결정) 마다 회귀 시나리오 set 실행.

**트리거:** BWT 또는 회귀 통과율이 임계값 아래로 떨어지면 학습 중단. 대응 우선순위 — (1) mixed curriculum 비율 상향, (2) 학습률 감소, (3) 이전 체크포인트 복귀.

#### 핵심 통찰 (사용자 질문이 끌어낸 것)

> "기존 모델가중치가 다음 단계에서도 사라지지 않는다는 개념" 이 자명한 전제처럼 보이지만, **실제로는 보장되지 않는다**. Catastrophic forgetting 은 신경망 순차 학습의 디폴트 결과이며, 본 프로젝트의 §12.2 후보 A/B/C/D 가 전부 "forgetting 방지" 를 명분으로 존재한다.

> 따라서 "가중치가 살아있는가" 라는 질문은 **확신의 영역이 아니라 검증의 영역** 이고, 검증은 가중치 자체가 아니라 **행동(성능)** 으로 한다.

#### 근거

- Catastrophic forgetting 은 RL continual learning 의 표준 실패 모드 (FAILURE_LOG §2.1 에도 예상 실패 카테고리로 명시됨)
- BWT 와 Forgetting measure 는 continual learning 문헌의 표준 지표 (Lopez-Paz & Ranzato 2017, Chaudhry et al. 2018)
- 망각은 새 step 학습 **초반에 가장 빠르게** 진행되므로 종료 후 1회 확인은 too late
- wandb 도입 (의사결정 9) 이 학습 reward 곡선과 회귀 곡선의 동시 시각화를 가능케 함 — 직접 시너지

#### 기각된 대안

- **가중치 변화량 norm `‖θ_N − θ_{N-1}‖` 기반 판정**: 가중치 변화량 ≠ 능력 변화량. Redundancy 로 인해 가중치 많이 변해도 성능 유지 가능, 민감한 방향으로 조금만 변해도 성능 무너짐. 보조 진단 panel 로는 가능하나 판정 기준으로는 부적합.
- **종료 후 1회 확인**: 망각은 학습 초반에 빠르게 진행 → 종료 후 발견은 이미 손실.

#### 신규 Lock 없음, 기존 Lock 의미 확장

- **L-D1 의 결정 범위 확장**: "전이학습 전략 (A/B/C/D 중 선택)" 만이 아니라 **회귀 감시 임계값 / 측정 주기 / 트리거 대응을 동시 결정** 으로 의미 확장. L-D1 해제 세션이 이를 모두 다뤄야 함.
- **L-A5 의 우선순위 격상**: 회귀 시나리오 set 의 신뢰성이 회귀 검증 자체의 신뢰성을 결정. L-A5 해제 전에는 회귀 감시 의미 약함.

#### SKILL 영향

- SKILL §3.7 신설: 회귀 감시 의무 (3종 지표 / 주기 / 트리거 명시)
- SKILL §6.4 절대 하지 말 것: "회귀 검증 임계값/주기/트리거 미정 상태 Step N+1 학습 시작 금지", "가중치 변화량 norm 만으로 능력 보존 판정 금지" 추가
- SKILL §0 차단 표: "Step 2 이상 학습 시작 → L-D1 미해제로 차단" 추가

---

### 본 세션의 사용자 통찰

> "L-D3와 같이 LOCK의 NO.에서 D가 의미하는게 뭐지?"

→ spec 문서의 자기 설명성 결함 지적. 의사결정 12 의 직접 trigger.

> "두 학습방법의 기본 논리는 기존 모델가중치가 다음 단계에서도 사라지지 않는 개념으로 보이는데, 그것에 대하여 어떻게 확신을 하지?"

→ "확신" 이라는 단어가 핵심. 본 spec 이 암묵적으로 "확신" 영역에 두었던 가중치 보존을 "검증" 영역으로 끌어내림. 의사결정 13 의 직접 trigger.

---

### 본 세션의 Claude 자기 비판

- 의사결정 8 (Layer 1/2 평가 분리) 작성 시 회귀 검증의 임계값 / 주기 / 지표를 같이 정의하지 못함. "회귀 시 학습 중단/조정" 이라고 적어두고 어떤 지표로 어떤 임계값에서 중단인지를 비워둠.
- §12.2 후보 A/B/C/D 가 모두 "forgetting 방지" 가 명분인데, 정작 "forgetting 자체를 어떻게 측정하는가" 는 spec 에 없었음. **수단을 나열하면서 측정 도구를 빠뜨림.**
- 사용자가 "어떻게 확신하지?" 라고 묻기 전까지 이 gap 을 자체 발견 못 함. _ing 시스템이 다시 한 번 사용자 질문의 힘으로 작동.

---

### 본 세션의 작업 산출물

```
/mnt/user-data/outputs/
├── CLAUDE.md   (§99 Lock 명명 규칙 범례 신설 - 의사결정 12)
├── SKILL.md    (§3.7 회귀 감시 의무 신설, §6.4 절대 금지 항목 추가,
│                    §6.5 Lock 명명 규칙 참조 신설, §0 차단 표 확장 - 의사결정 13)
├── PROGRESS.md (본 Session 2026-06-11 추가)
└── FAILURE_LOG.md  (entry 추가: spec 의 회귀 검증 체계 미완성 발견)
```

---

### 다음 세션 진입점

1. 본 PROGRESS.md + CLAUDE.md + SKILL.md + FAILURE_LOG.md 첨부
2. 권장 시작 멘트: "L-D1 해제 진행 — 전이학습 전략 + 회귀 감시 체계 동시 결정"
3. 권장 우선순위 (이전 권고 보강):
   - **L-D1 (전이학습 전략 + 회귀 감시 체계 동시 결정)** — 의사결정 13으로 범위 확장됨
   - **L-A5 (평가 시나리오 구성)** — L-D1 의 회귀 시나리오 set 의 토대
   - **L-D2 (Dense reward Φ 정의)** + **L-16.3-w (baseline 가중치)** — Phase 1 재학습 조건
   - **L-C2 (autoresearch 메타 + wandb)** — 도구 인프라

---

## Session 2026-06-18 — L-A5 / L-D1 / L-D2 / L-A6 / L-C2 / L-16.3-w 해제 (Phase 1 시작 조건 충족)

### 세션 배경

이전 Session 2026-06-11 에서 도출된 Lock 해제 우선순위 (L-A5 → L-D1 → L-D2) 에 따라 본 세션에서 **L-A5, L-D1, L-D2, L-A6, L-C2, L-16.3-w 총 6개 Lock 해제**. L-A6 신설 후 같은 세션에서 해제.

세션 흐름:
1. Claude 가 영향력 분석 → L-A5 가 가장 영향력 큰 Lock 으로 판정
2. 사용자가 L-A5 해제 세션 요청 → 5개 sub-question 진행
3. 사용자 confirm 후 L-A5 해제 + L-A6 신설 + L-D1 채택
4. 4개 파일 일괄 반영 후 사용자가 L-D2 권장안 confirm → L-D2 해제
5. 4개 파일 추가 반영
6. 사용자가 L-16.3-w / L-C2 / L-A6 해제 권고안 요청 → Claude 권고안 제시
7. 사용자 confirm 후 3개 Lock 동시 해제
8. 4개 파일 최종 반영 (본 entry)

**결과: Phase 1 학습 시작 조건 충족**. 남은 4개 Lock (L-A3', L-A4, L-C4, L-D3) 은 모두 Phase 1 시작에 영향 없는 후행 결정.

---

### 핵심 의사결정 14 — L-A5 해제 (평가 시나리오 구성)

#### 결정 내용

##### (1) 시나리오 set 4분할

| 용도 | 개수 | 생성 방식 |
|------|------|----------|
| 학습용 | 무한대 | Procedural, 매 episode 새 seed |
| Screening | Step별 75개 고정 | Procedural + 고정 seed list |
| Regression | Step별 150개 고정 | Procedural (초기) + 수작업 누적 |
| Final Eval | Step별 500개 고정 | Procedural (초기) + 수작업 누적 |

용도 간 seed range 절대 비겹침 (데이터 누수 방지).

##### (2) 난이도 분포 Easy / Medium / Hard = 3 : 5 : 2

각 용도 내에서 동일 비율. 학습용 procedural generator 는 micro-curriculum (Easy 60% → Hard 30%) 적용 가능.

##### (3) Step별 난이도 축 차별화

각 Step 의 학습 목표가 다르므로 난이도 축도 Step 별 차별화. 예: Step 1 은 장애물 밀도 + start-goal 거리, Step 3 은 중력관 비율 + 슬로프 압박, Step 6 은 파이프 길이 + 서포트 가능면. 구체는 CLAUDE.md §11.0.3 참조.

##### (4) Seed Range 분리 체계

Step N 의 seed range 는 N\*100000 시작. 그 안에서 용도별 sub-range (screening 10000~19999 / regression 20000~29999 / final eval 30000~39999 / 학습용 50000~99999). 충돌 절대 금지.

##### (5) Procedural 100% 시작, FAILURE_LOG 누적 시 수작업 점진 추가

수작업 비율은 0% 시작. **FAILURE_LOG entry 추가 시마다 해당 실패 케이스를 수작업 seed pool 로 영구 등록**. 시간 경과 시 자연 누적 (최대 30% 권장). 이로써 FAILURE_LOG 는 단순 archive 가 아니라 **regression set 의 동적 source** 가 됨.

##### (6) Seed 관리 인프라

- YAML 기반 저장, Git 추적
- `numpy.random.default_rng(seed)` (PCG64) 만 사용. legacy API 금지
- Generator version 관리 (algorithm 수정 시 version 증가 + hash 보관)

##### (7) 회귀 검증 통과 임계값 (Stage 1 vs Stage 2 분리)

6개 지표 (success_rate, length_ratio, bend_count_ratio, Layer 1 통과율, BWT_k, F_k) 의 임시 임계값 채택. Phase 1 결과 분포 보고 v0.4 spec 에서 조정 예정. 구체는 CLAUDE.md §11.0.7 참조.

#### 근거

- ProcGen / MetaWorld / MiniGrid 등 표준 RL 벤치마크의 관행: 학습용 무한대 / 평가용 고정 set 분리
- 시나리오 수 (75/150/500) 는 통계적 신뢰도와 계산비용의 trade-off 결과. 회귀 100~200 범위 중앙값, screening 50~100 범위 중앙값
- 난이도 3:5:2 는 continual learning 문헌의 표준 비율
- Step 별 난이도 축 차별화는 사용자 통찰 직접 채택 ("각 STAGE의 목표에 맞게 난이도를 다르게 정의")
- FAILURE_LOG → 수작업 seed 메커니즘은 regression test 의 가장 효과적 운영 방식 (실패한 적 있는 케이스가 영구 평가 대상으로 승격)

#### 기각된 대안

- **시나리오 모두 수작업**: 도메인 의미는 좋으나 개수 제한 + 작성자 편향 + 학습용 부적합
- **screening 50개 미만 / regression 100개 미만**: 통계 노이즈가 신호 압도
- **seed range 겹침 허용**: 사실상 평가 데이터 누수 (autoresearch 결과 신뢰도 붕괴)
- **autoresearch 로 회귀 임계값 학습**: 순환 의존 (objective 안에 임계값이 있어야 함). 대신 autoresearch 가 생산하는 데이터를 사람이 후행 조정

---

### 핵심 의사결정 15 — L-A6 신설 (Procedural Scenario Generator 알고리즘 spec)

#### 배경

L-A5 해제 시 시나리오 set 의 **메타 구성** (개수, 난이도 분포, seed range, 임계값) 은 결정되었으나, **시나리오 자체의 내용을 생성하는 알고리즘** 이 별도 spec 필요. 사용자가 직접 요청 ("장애물 배치 알고리즘, start-goal sampling 알고리즘, 난이도 파라미터 매핑에 대한 별도 spec lock 을 추가해줘").

#### 결정

신규 Lock **L-A6** 설립. 카테고리는 A (Assessment, 평가 체계의 일부) 로 분류.

#### L-A6 결정 항목

- 장애물 배치 알고리즘 (Poisson disk / Perlin noise / uniform with min-distance / 도메인 형상 비율)
- Start-Goal sampling 알고리즘
- 난이도 파라미터 매핑 (Easy/Medium/Hard 추상 정의 → generator 입력 값)
- Step 4~6 sub-algorithm (분기점 / 밸브 / 서포트 가능면)
- Step 7~10 sub-algorithm (L-D1 후속 영향)
- Generator version 관리 규칙

#### 차단 행위 이동

SKILL §0 차단 표의 "시나리오 생성기 작성" 의존이 L-A5 → L-A6 로 이동. L-A5 해제로 차단 해제되지 않고 L-A6 신설로 차단 유지.

#### 해제 시점 권장

implementation 단계 초기. Phase 1 학습 시작 전 결정. L-D2 / L-16.3-w / L-C2 해제 후 진행 가능 (병렬 작업 가능).

---

### 핵심 의사결정 16 — L-D1 해제 (전이학습 전략 + 회귀 감시 체계)

#### 결정 내용

##### (1) 전이학습 전략 — 후보 A + B 혼합

§12.2 의 4개 후보 중 A (순차 fine-tuning) + B (mixed curriculum) 혼합 채택. C (EWC) 와 D (LoRA) 비채택.

**기각 근거**:
- **C (EWC)**: PPO 의 on-policy 특성 + sample inefficiency 와 Fisher 정보 추정의 충돌
- **D (LoRA)**: backbone hidden_dim=256 이 LoRA 효과 보기엔 작음

##### (2) 학습 데이터 mix 비율

```
Warm-up phase (학습 초기 100K timestep):
  50%  Step N 신규 시나리오
  40%  Step 1 ~ Step N-1 시나리오 (mixed curriculum)
  10%  FAILURE_LOG 누적 실패 케이스

정상 phase:
  60% / 30% / 10%
```

##### (3) 학습률 — fine-tuning 표준 관행

신규 학습의 50% 로 시작.

##### (4) 회귀 감시 측정 주기

매 25 rollout (≒ 50K timestep). Step N 총 학습 2M timestep 가정 시 40번 측정 (충분한 해상도).

##### (5) 트리거 escalation (4-level)

- Level 0 (즉시 중단): Layer 1 hard 5% 이상 위반
- Level 1 (경고): 1번째 soft 위반 → wandb alert
- Level 2 (자동 조정): 2번째 연속 위반 → mixed curriculum 30→45%, 학습률 50% 감소
- Level 3 (학습 중단): 3번째 연속 위반 → 체크포인트 복귀

##### (6) 회귀 임계값

CLAUDE.md §11.0.7 표 (Stage 1 vs Stage 2 분리) 그대로 채택. L-A5 해제로 임계값의 의미가 정합 (어떤 시나리오 set 위의 어떤 통과율인지 명확).

#### 근거

- continual RL 문헌의 표준 관행 (mixed curriculum 30% 정도가 forgetting 방지 최소선)
- Warm-up 의 mix 비율 강화 (50→60% 신규) 는 catastrophic forgetting 의 학습 초반 집중 현상 대응
- Level 0 즉시 중단의 5% 임계값은 Layer 1 (slope, ASME) 의 hard 성격 반영 — 1% 위반도 문제이나 측정 노이즈 고려해 5%
- 50K timestep 주기는 wandb logging 주기와 정합. 너무 잦으면 (10K) 학습 속도 손실, 너무 드물면 (200K) 회귀 발견 지연

#### 의사결정 13 (2026-06-11) 과의 관계

의사결정 13 에서 L-D1 의 범위를 "전이 전략 선택" 에서 "회귀 감시 체계 동시 결정" 으로 확장. 본 의사결정 16 이 그 확장된 범위를 모두 채움.

---

### 핵심 의사결정 17 — L-D2 해제 (Dense Reward Φ 구체 정의)

#### 결정 내용

##### (1) Phase 1 Φ 구성

```
Φ(s) = α · Φ_goal(s) + β · Φ_cong(s)

  Φ_goal(s) = -d_manhattan(s, goal) / d_initial    범위 [-1, 0]
  Φ_cong(s) = -1 / (mean_SDF_5x5x5(s) + 1.0)       범위 음수, |Φ_cong| ~ O(1)
  Φ_future  = Phase 1 미사용 (Phase 2+ 검토)
```

##### (2) Distance metric 선택 — Manhattan

7-direction discrete action 과 정합. Euclidean 은 대각선 단축이 가능하다는 거짓 신호 위험. SDF-aware (장애물 우회 거리) 는 매력적이나 계산량과 진단 복잡도 증가 → Phase 1 에서는 단순 Manhattan, Phase 1 결과 보고 SDF-aware 를 autoresearch sweep 대상에 추가 검토.

##### (3) Φ_goal 정규화 — d_initial 로 나눔

raw d_goal 은 grid 단위 (0~30) 로 너무 큼 → episode 초기 거리로 나눠 |Φ_goal| ≤ 1 보장. baseline reward 의 다른 항목과 자릿수 정합.

##### (4) Terminal 처리 — 명시적 0 set 의무

```
goal_reached:    Φ = 0  (자연스러움, d_goal=0)
collision:        Φ = 0  (명시적 set, PBS 조건 2)
timeout:          Φ = 0  (명시적 set)
out_of_bounds:    Φ = 0  (명시적 set)
```

구현 시 unit test 의무.

##### (5) 초기값과 sweep 범위

```
초기값:  α = 1.0  (Φ_goal 가중치, 핵심)
        β = 0.1  (Φ_cong 가중치, 보수적 시작)

autoresearch Stage 1:
  α ∈ {0.5, 1.0, 2.0, 5.0}
  β ∈ {0.0, 0.1, 0.3}
  → 12 variants

autoresearch Stage 2 (살아남은 후보):
  α, β ± 50% local grid
```

##### (6) PBS 안전조건 4번째 신설 — γ_PBS == γ_PPO

기존 3대 조건 (state-only / terminal=0 / γΦ(s')-Φ(s) 형식) 에 4번째 조건 추가:

```
조건 4. γ_PBS == γ_PPO (PPO 의 discount factor 와 PBS 의 γ 동일 값)
```

autoresearch sweep 으로 γ 변경 시 PBS γ 도 함께 변경하는 sync constraint 신설 (`pbs_gamma_sync` constraint).

##### (7) `PotentialBasedShaping` helper class 강제

PBS 4대 조건을 type level + runtime check 로 강제하는 wrapper class. 직접 `r_shape` 작성 금지. SKILL §3.5 / §3.1 에 의무 명시.

##### (8) wandb 진단 panel 정의

```
r_baseline_mean / r_shape_mean             : 분리 logging
r_shape_to_baseline_ratio                  : 정상 0.1~1.0, > 2.0 위험
phi_goal_at_step0 / phi_goal_at_terminal   : episode 진행 곡선
phi_cong_distribution                       : histogram
action_value_phi_corr                       : Q-value gradient 와 Φ gradient 코사인 유사도
```

진단 우선순위: ratio → corr → trajectory.

#### 근거

- **Manhattan 채택**: 7-direction discrete action 과 정합. ProcGen 등 표준 grid 환경 관행
- **d_initial 정규화**: baseline reward 항목과 자릿수 맞춤. shape 가 baseline 압도 방지
- **Φ_future 보류**: 정의 모호성 미해결 (mini A* / value approximation / heuristic 모두 trade-off 있음). Phase 1 결과 보고 도입 검토
- **γ_PBS == γ_PPO 신설**: Ng et al. 1999 의 PBS 이론적 보장은 γ 일치 전제. autoresearch 가 γ 만 sweep 하고 PBS γ 를 안 바꾸면 보장 깨짐 → 코드 강제로 방지
- **12 variants Stage 1**: continual RL hyperparameter sweep 의 표준 규모. 단, Colab GPU 동시 학습 모델 수 (L-C2 미정) 에 따라 wall-clock 시간 산출 후 조정 가능

#### 기각된 대안

- **Euclidean distance**: 대각선 단축 거짓 신호. 7-direction action 과 부정합
- **SDF-aware distance (Phase 1 도입)**: 계산량 증가 + 진단 복잡도. Phase 1 에서는 단순함 우선
- **Φ_future Phase 1 도입**: 정의 모호. 미숙한 정의로 도입 시 reward hacking 위험
- **α=2.0, β=0.5 (높은 시작값)**: shape 가 baseline 압도 위험. autoresearch 가 발견하도록 보수적 시작

#### autoresearch wall-clock 위험 검토

12 variants × Stage 1 학습 시간 = 본 프로젝트 Colab T4 GPU 1개 가정 시 약 12 시간 (variant 당 1시간 가정). 동시 실행 가능 모델 수가 4개라면 3 시간. L-C2 해제 시 명확해짐. 12 가 과도하면 sequential sweep (α 4개 → β 3개) 으로 7 variants 로 축소 가능.

#### Phase 1 적용 spec 완성

L-D2 해제로 다음 spec 완성:
- §11.0: 평가 시나리오 set (L-A5)
- §12.2~12.3: 전이학습 + 회귀 감시 (L-D1)
- §16.7: Dense Reward Phase 1 적용 (L-D2)

**남은 Phase 1 시작 전제**: L-16.3-w (baseline 가중치), L-C2 (autoresearch 메타), L-A6 (procedural generator 알고리즘). 이 3개 해제 시 Phase 1 학습 시작 가능.

---

### 핵심 의사결정 18 — L-C2 해제 (autoresearch 메타 구조)

#### 결정 내용

##### (1) 1차 스크리닝 학습 시간 비율 — 1/8

```
정상 학습 시간 (Phase 1 Step 1):  2,000,000 timestep
Stage 1 (1차 스크리닝):             250,000 timestep (1/8)
```

**근거**: 1/4 (500K) 은 너무 길어 Stage 2 와 차이 약화. 1/8 (250K) 가 (a) 학습 곡선 초기 trajectory 안정화 + (b) 12 variants 합리적 시간 (T4 기준 약 2시간) 의 균형점. ProcGen / Optuna 사례 관행도 1/8~1/10.

##### (2) 1차 스크리닝 제거 비율 — 하위 50%

successive halving 의 표준 비율. 70% 제거는 짧은 학습 노이즈로 좋은 variant 잘려나갈 위험.

##### (3) 동시 학습 가능 모델 수 — T4 1개 기준 2~3개

```
MaskablePPO + hidden_dim 256 + obs 150 + grid 30³
  → 단일 학습 VRAM ~3GB
  → T4 16GB 에서 동시 2~3개 (안전 margin 포함)
```

Pro+ / A100 환경에서 4~6개 가능. 사용자 환경에 따라 wall-clock 산출.

##### (4) autoresearch 종료 조건 (Round 단위 정의)

```
1 Round = Stage 1 + Stage 2 + 평가 ≒ 1.5~3 시간 (T4 GPU, 동시 3개)

종료 조건 (OR):
  1. Round 5 도달 (절대 상한)
  2. 최근 2 Round best 변형 동일 (개선 정체)
  3. wandb best variant plateau (마지막 200K success_rate 증가 < 2%p)
  4. 사용자 강제 종료
```

**근거**: Round 5 는 reward tuning 의 표준 깊이. plateau 임계값 2%p 는 통계 노이즈 (~1%p) 보다 약간 큰 값.

##### (5) wandb naming 체계

```
project: "pipe-routing-rl"
name:    f"step{N}_round{R}_stage{S}_var{V:03d}"  (예: step1_round1_stage1_var007)
config:  step, round, stage, variant_id, sweep 대상 + git_commit + generator_version
tags:    [step{N}, round{R}, stage{S}, autoresearch, phase1|phase2plus]
```

##### (6) Optuna 채택 — TPE + MedianPruner + sqlite

```python
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(n_startup_trials=12),  # Stage 1 grid 보장
    pruner=optuna.pruners.MedianPruner(),                      # plateau 자동 차단
    storage="sqlite:///autoresearch.db",                        # 영구 보관
    study_name=f"step{N}_round{R}",
)
```

`get_param_importances()` 로 천장 vs search 부족 진단 가능.

#### 근거

- 1/8 + 50% 는 hyperparameter optimization 의 표준 successive halving
- TPE 가 단순 grid 보다 효율적 (Optuna 표준)
- MedianPruner 가 plateau variant 자동 차단으로 시간 절약
- sqlite storage 가 Round 간 study 공유 + 재개 가능

#### 기각된 대안

- **Ray Tune**: Optuna 보다 강력하나 본 프로젝트 규모에 과함. 학습 곡선 길지 않아 ASHA 같은 정교한 pruner 불필요
- **WandB Sweep**: wandb 내장 sweep 도구이나 importance analysis 약함. Optuna + wandb logging 조합이 우월
- **Grid only (Optuna 없이)**: importance analysis 못함 → 천장 진단 도구 부재

---

### 핵심 의사결정 19 — L-16.3-w 해제 (baseline reward 가중치)

#### 결정 내용

##### (1) Step 1 가중치 초기값

```
w1 (length)         = 0.1     ← step 당 가벼운 길이 penalty
w2 (collision)      = 2.0     ← 마스킹 외 충돌 억제, length 2배 자릿수
w3 (goal_bonus)     = 50.0    ← episode penalty 대비 약 3배
w4 (direction_align) = 5.0    ← §12.4 이미 고정값 (sweep 대상 아님)
w5 (wrong_dir)      = 15.0    ← §12.4 이미 고정값 (sweep 대상 아님)
```

##### (2) 단위 분석 근거

```
Medium 난이도, 학습 초기 episode 평균:
  length 50 step, collision 5회, goal 도달 시:
    r = -0.1·50 - 2.0·5 + 50.0·1 + 5·1 - 15·0
      = -5 - 10 + 50 + 5 = +40  (net positive 확보)
  
  goal 미도달 (timeout):
    r = -0.1·100 - 2.0·10 + 0 = -30
  
  도달/미도달 차이 = 70 → 충분한 학습 신호
```

FAILURE_LOG §6.5 placeholder 예시 (goal_bonus 100 → 500 의 sparse trap) 회피.

##### (3) Step 1 sweep 범위

```
Round 2 sweep (Round 1 의 α/β best 고정 후):
  w1 ∈ {0.05, 0.1, 0.2, 0.5}     # 4 values
  w2 ∈ {1.0, 2.0, 5.0}            # 3 values
  w3 ∈ {20, 50, 100}              # 3 values
  → 4 × 3 × 3 = 27 variants
```

##### (4) Step 2~6 가중치 — 상대 스케일 원칙

w6 ~ w17 의 구체값은 각 Step 진입 직전 결정. 원칙만 spec 화:

```
1. Hard constraint (slope_violation, bend_radius_violation 등) w: ~20 (net negative 보장)
2. Soft constraint (excess_bend_count, branch_total_length_excess 등) w: ~0.5 (length 와 자릿수 비슷)
3. 새 Step w 도입 시 이전 Step r 합산 크게 변경 안 함 (회귀 검증 통과)
```

#### 근거

- 단위 분석 기반 도달/미도달 reward 차이 70 ≫ 학습 노이즈
- w4, w5 는 §12.4 이미 고정이므로 sweep 대상 제외 (spec 일관성)
- sweep 범위는 자릿수 단위 변화 (0.05 → 0.5 = 10배) 포함, 안전 범위
- Step 2~6 후행 결정은 각 Step 진입 시점에 실제 reward scale 측정 후 결정 가능

#### 기각된 대안

- **w3 = 100 이상 큰 값**: shape reward 와 baseline 자릿수 불일치 위험 (PBS 분석)
- **w3 = 500 (FAILURE_LOG 예시값)**: episode 누적 penalty 대비 과도. 50 으로 충분
- **w1 = 1.0 같은 큰 length penalty**: episode 길어질수록 penalty 누적 폭주
- **모든 w 한 번에 sweep**: L-D2 의 α/β 와 동시 시 432 variants → 비현실적. Sequential sweep 채택

---

### 핵심 의사결정 20 — L-A6 해제 (Procedural Scenario Generator 알고리즘)

#### 결정 내용

##### (1) 장애물 배치 — Poisson Disk Sampling + Cuboid 합성

```
density = {"easy": 0.10, "medium": 0.20, "hard": 0.35}
도메인 형상 비율: tank 0.4 / structural 0.3 / small_eq 0.3
```

**Poisson disk 채택**: 장애물 간 최소 거리 보장 → 통과 corridor 자동 보장.

**기각**: uniform voxel (비현실적 단일 voxel 흩어짐), Perlin noise (angular 형상과 부정합), Voronoi (cell 경계 모호).

##### (2) Start-Goal Sampling — Rejection Sampling + A* Reachable 검증

```
rejection sampling 으로 거리 + ratio 제약 만족하는 (start, goal) 쌍 찾기
A* reachable 검증으로 풀이 불가능 시나리오 제거
```

**A* 사용 안전성**: 본 A* 는 **시나리오 검증 전용**이며 학습 신호로 사용 안 됨. §11.1 평가 철학 ("A* = 평가 도구") 과 충돌 없음.

##### (3) 난이도 파라미터 매핑 — `DIFFICULTY_PARAMS` dict + `_inherits`

```python
DIFFICULTY_PARAMS = {
    "step1": {"easy": {...}, "medium": {...}, "hard": {...}},
    "step2": {"easy": {"_inherits": "step1.easy", "min_bend_count": 0, ...}, ...},
    # ...
}
```

`_inherits` 키로 Step 상속 명시. spec 변경 추적 용이.

##### (4) Step 4~6 sub-algorithm — 개요만, 각 Step 진입 직전 구체화

```
Step 4: 분기점 후보 + 목적지 분산 + tee/elbow fitting 검증
Step 5: 밸브 위치 후보 + 간격 + clearance
Step 6: 서포트 가능면 + 파이프 길이 → 서포트 수 + 무게 분포
```

Phase 1 직접 영향 없음 → 후행 결정.

##### (5) Step 7~10 sub-algorithm — Step 7 진입 시점 별도 세션

다중 파이프의 공간 경합 / 우선순위 분리 등은 Step 7 진입 시 다시 결정 필요.

##### (6) Generator Version 관리 — semver + git tag + hash 보관

```
v1.0.0 → v1.0.1: 버그 수정, 동작 동일
v1.0.0 → v1.1.0: 새 sub-algorithm 추가 (기존 시나리오 재현 가능)
v1.0.0 → v2.0.0: 기존 알고리즘 변경 (재현 안 됨)
```

major version 증가 시 시나리오 hash 보관 + PROGRESS entry 의무.

#### 근거

- Poisson disk 는 procedural content generation 의 표준 (장애물 배치, 게임 레벨 생성에서 검증됨)
- Cuboid 합성은 선박 배관 환경의 angular 형상 (탱크, 구조재, 장비) 과 정합
- A* reachable 검증은 학습 신호가 아니므로 평가 철학 충돌 없음 — generator 가 trivial 한 풀이 불가 시나리오 방지
- semver 는 인터넷 표준, 도구 (poetry, npm) 지원 풍부
- Step 4~6 후행은 Phase 1 진입에 영향 없으므로 합리적 deferral

#### 기각된 대안

- **Uniform random voxel**: 단일 voxel 흩어져 비현실적
- **Perlin noise**: 자연 noise 이나 선박 배관 환경과 부정합
- **수작업 100% 시나리오**: 학습용 무한대 supply 불가
- **A* reachable 검증 생략**: 풀이 불가 시나리오 학습 데이터 오염

---

### Phase 1 시작 조건 충족 — 완성된 spec map

```
[Phase 1 학습에 필요한 모든 spec]

시나리오:
  §11.0     — 시나리오 set 4분할 + 난이도 + seed range (L-A5)
  §11.0.8   — Generator 알고리즘 v1.0.0 (L-A6)

Reward:
  §16.3.1   — baseline 가중치 초기값 (L-16.3-w)
  §16.3.2   — Step 1 sweep 범위 (L-16.3-w)
  §16.7     — Dense Reward Φ Phase 1 spec (L-D2)

autoresearch 운영:
  §16.5     — 2-Stage screening 구체값 (L-C2)
  §16.6     — 종료 조건 (L-C2)
  §16.6.5   — Optuna 통합 (L-C2)
  §16.6.6   — wandb naming (L-C2)
  §16.6.7   — Phase 1 Round 운영 순서 (L-D2 + L-16.3-w + L-C2 정합)

전이학습 / 회귀 (Step 2 이상에서 적용, Phase 1 자체는 미사용):
  §12.2     — 전이학습 메커니즘 (L-D1)
  §12.3     — 회귀 감시 체계 (L-D1)
```

**남은 활성 Lock 4개** (모두 Phase 1 시작에 영향 없음):

| Lock | 상태 | 해제 시점 |
|------|------|----------|
| L-A3' | 활성 | Phase 1 결과 분포 본 후 후행 결정 |
| L-A4 | 활성 | Step 4 진입 시점 |
| L-C4 | 활성 (경량) | unit test 위임 가능 (사실상 자동 해제) |
| L-D3 | 활성 (보류) | Phase 1 결과 후 macro-action 도입 검토 |

---

### 본 세션의 사용자 통찰

> "각 STAGE별로 난이도를 다르게 정의하면 되는 거 아니야?"

→ 의사결정 14 의 핵심. 단일 난이도 정의 (장애물 밀도 등) 의 함정을 사용자가 먼저 지적. Claude 가 Step 별 학습 목표에 맞춰 난이도 축을 차별화하는 방향으로 답.

> "수작업 비율은 시작은 0%로 두고, FAILURE_LOG에 entry가 쌓일 때마다 그 실패 케이스를 수작업 seed로 추가하는 방식이 자연스럽습니다."

→ 사용자가 Claude 의 권고를 그대로 인용 채택. 본 메커니즘이 FAILURE_LOG 의 역할 자체를 확장 — passive archive → regression set 의 active source.

> "임시값을 사용하고 autoresearch 기법으로 찾아낼수 있는거 아닌가?"

→ Claude 가 "autoresearch 가 임계값을 직접 학습할 수는 없으나 (순환), 임시값으로 시작해 데이터로 후행 조정 가능" 으로 답. 임시 임계값 표 채택.

---

### 본 세션의 Claude 자기 비판

- 의사결정 14 의 시나리오 수 (75/150/500) 는 표준 관행 기반이나, 본 프로젝트의 grid 30³ + observation 150-dim 특성에 맞춘 정밀 검토는 부족. Phase 1 결과로 적정성 사후 검증 필요.
- Step 7~10 의 난이도 축은 L-D1 후속으로 미루었으나, L-D1 본문에서도 다중 파이프 부분이 잠정. L-D1 의 다중 파이프 측면은 향후 다시 다룰 필요.
- 회귀 임계값 6개를 임시값으로 정한 것은 합리적이나, 6개를 한꺼번에 trigger 로 보는 OR 로직 vs 가중 합 vs 우선순위 차등 등의 합산 규칙이 spec 에 미정. 운영 중 정의 필요.

---

### 본 세션의 작업 산출물

```
/mnt/user-data/outputs/
├── CLAUDE.md   (§11.0 시나리오 spec 신설, §12.2 전이학습 채움, §12.3 회귀 감시 체계 신설,
│                    §99 L-A5/L-D1 해제 + L-A6 신설, §101 갱신)
├── SKILL.md    (§0 차단 표 갱신, §3.7 회귀 감시 구체값 명시)
├── PROGRESS.md (본 Session 2026-06-18 추가)
└── FAILURE_LOG.md  (§0.1 신설 — L-A5 해제 반영, regression set dynamic source 역할)

[L-D2 해제 반영 추가 작업]:
├── CLAUDE.md   (§16.7 Dense Reward Phase 1 spec 채움, §99 L-D2 해제 표시, §101 갱신)
├── SKILL.md    (§3.1 PBS 4대 조건 + helper class 의무, §3.5 구체화,
│                    §4.1 PBS 변형 인자 / FORBIDDEN 갱신, §4.2 PBS Gate 4/5 추가,
│                    §6.4 helper 우회 금지)
└── PROGRESS.md (의사결정 17 추가, 본 entry 갱신)

[L-A6 / L-C2 / L-16.3-w 해제 반영 최종 작업]:
├── CLAUDE.md   (§11.0.8 Generator 알고리즘 v1.0.0 신설 [L-A6],
│                    §16.3.1~3 가중치 초기값/sweep/상대 스케일 [L-16.3-w],
│                    §16.5~16.6.7 autoresearch 운영 spec 채움 [L-C2],
│                    §99 3개 Lock 해제 표시, §101 갱신, 문서 버전 v0.6)
├── SKILL.md    (§0 차단 표 Phase 1 시작 가능 표시,
│                    §3.8 Generator 의무, §3.9 가중치 초기값,
│                    §4.4 Stage 구체값, §4.7 Optuna, §4.8 wandb naming, §4.9 Round 운영,
│                    문서 버전 v0.6)
├── PROGRESS.md (의사결정 18/19/20 추가, 본 entry 최종 갱신)
└── FAILURE_LOG.md  (버전 정보 갱신만)
```

---

### 다음 세션 진입점

#### Phase 1 학습 시작 단계

본 세션으로 Phase 1 spec 모두 채워짐. 다음 세션은 **implementation 단계** 진입.

```
다음 세션 가능 진입점 (택 1):

[A] harness engineering 시작
    - SKILL.md / CLAUDE.md → SKILL.md / CLAUDE.md 로 rename
    - §0 차단 Lock 의 단일 에이전트 학습 코드 / Dense Reward / autoresearch /
      시나리오 생성기 차단 모두 해제 상태
    - env class 구현 시작 가능 (L-C4 미해제이나 unit test 로 위임 가능)

[B] L-C4 단독 해제 (좌표계 unit test 명시화)
    - 환경 구현 차단 완전 해제
    - 매우 가벼운 세션 (1시간 이내 예상)

[C] L-A3' 사전 검토 (Phase 1 결과 임계값 사전 설계)
    - Phase 1 결과 보고 후행 결정이 정석이나, 사전 골격 잡기 가능
    - "어떤 metric 에서 어느 percentile 이 cutoff" 의 의사결정 프레임

[D] L-A4 사전 검토 (Step 4~6 hard constraint 사전 설계)
    - Step 4 임박 시점 결정이 정석이나, 사전 골격 잡기 가능

권장: [A] harness engineering 시작.
      Phase 1 학습 진행 중 발견되는 issue 가 L-A3' / L-A4 의사결정의 데이터가 됨.
```

#### Phase 1 학습 실패 시 복귀 경로

```
1. FAILURE_LOG entry 추가 (예: "Phase 1 Round 1 plateau")
2. wandb / Optuna importance analysis 로 천장 vs search 부족 진단
3. 천장이면 → L-D3 (macro-action) 해제 검토 또는 §16.7 Φ_future 도입 검토
4. search 부족이면 → autoresearch sweep 범위 확장 (Round 3+ 적극 진행)
5. spec 결함 발견 시 → 해당 Lock 재논의 / 새 Lock 신설
```

#### 남은 활성 Lock 4개

| Lock | 권장 해제 시점 |
|------|--------------|
| L-A3' | Phase 1 학습 결과 분포 확인 후 |
| L-A4 | Step 4 진입 시점 (가까운 시일 아님) |
| L-C4 | implementation 직전 (단독 세션 1시간 또는 unit test 위임) |
| L-D3 | Phase 1 결과 후 macro-action 도입 필요 시

---

## Session 2026-06-24 — 🎓 \_ing 시스템 졸업

### 세션 배경

이전 Session 2026-06-18 에서 Phase 1 학습 시작 조건이 충족됨 (7개 Lock 해제 + L-A6 신설→해제). 사용자가 "harness engineering 은 Claude Code CLI 에서 하는 게 맞는가" 질문 → Claude 가 그렇다고 확정 + "그 전에 \_ing 시스템 졸업 작업이 필요" 제안. 사용자가 졸업 작업 진행 요청.

본 세션의 목적: **로컬 Claude Code CLI 환경으로 넘어가기 전 spec 파일들을 졸업판 상태로 정비**.

---

### 핵심 의사결정 21 — \_ing 시스템 졸업 (CLAUDE_ing/SKILL_ing/PROGRESS_ing → CLAUDE/SKILL/PROGRESS)

#### 결정 내용

##### (1) 파일 rename

```
CLAUDE_ing.md   → CLAUDE.md
SKILL_ing.md    → SKILL.md
PROGRESS_ing.md → PROGRESS.md
FAILURE_LOG.md  → FAILURE_LOG.md  (이미 _ing 없음, rename 없음)
```

##### (2) 각 파일 헤더 갱신

- 첫 헤더 줄에 졸업 안내 (🎓 2026-06-24 졸업)
- 작성 중 (`_ing`) 안내 제거
- 차단 의미 → 운영 안내 의미로 전환

##### (3) SKILL.md §0 차단 표 → §0 운영 상태 안내로 전환

- 차단 행위 표 → "harness engineering 진행 가능" 명시
- 남은 4개 활성 Lock (L-A3', L-A4, L-C4, L-D3) 의 자기 자연 시점 해제 가이드
- vibe coding 금지 원칙은 유지

##### (4) Cross-reference 일괄 갱신

- 모든 spec 본체 (CLAUDE.md) 및 archive (PROGRESS.md, FAILURE_LOG.md) 의 cross-reference 새 이름으로 갱신
- 역사적 정확성은 git history 로 추적 (commit `docs: graduate _ing system`)

##### (5) 졸업 후 운영 모드 spec

```
[일상 운영 - Claude Code CLI]
- 코드 변경 시 SKILL.md §3 코딩 규칙 자동 적용
- §3.1 절대 변경 금지 항목 임의 변경 차단
- git commit 으로 변경 추적

[새 의사결정 - Claude AI 세션]
- 4개 파일 일관 갱신 (CLAUDE/SKILL/PROGRESS/FAILURE_LOG)
- 별도 Lock 해제 절차 거침
- harness engineering 환경에서는 spec 변경 권한 없음

[학습 결과 분석 후 spec 재논의 - Claude AI 세션]
- wandb screenshot 첨부, 분석 후 spec 변경 결정
- FAILURE_LOG entry 추가 시 수작업 seed pool 영구 등록
```

##### (6) 문서 버전 일괄 갱신

모든 파일 v0.6 → **v1.0 (졸업판)**.

#### 근거

- **\_ing 시스템 의 목적 완료**: vibe coding 차단, spec drift 방지, Lock 시스템 운영. 본 세션 시점에 모든 기능 안정. 차단 의미가 유지되면 오히려 harness engineering 진행 방해.
- **Phase 1 시작 가능 상태 명시화**: 졸업판은 "지금부터 코드 작성 OK" 의 spec 화. \_ing 상태 유지 시 모호함.
- **CLI 환경 정합성**: Claude Code CLI 가 폴더에 들어가면 SKILL.md / CLAUDE.md 를 자동 읽음. \_ing 이름이면 자동 인식 방해.
- **남은 4개 Lock 의 성격 변화**: 차단 Lock 이 아니라 "자기 자연 시점에 풀리는 후행 결정" 으로 의미 변화. \_ing 시스템에서 머무를 이유 없음.

#### 기각된 대안

- **모든 Lock 해제 후 졸업**: 남은 4개 Lock 은 모두 후행 결정 성격. Phase 1 시작 자체에 영향 없음. 4개 모두 풀 때까지 \_ing 유지는 비효율.
- **단계적 졸업 (CLAUDE.md 만 먼저 졸업, 나머지 \_ing 유지)**: 파일 간 일관성 깨짐. 일괄 졸업이 정상.
- **\_ing 유지 + 차단 표만 해제**: 이름과 실체 불일치 → 혼란 가중.

#### 영향

- **차단 의미 전환**: SKILL §0 의 "금지" → "운영 안내"
- **vibe coding 금지 원칙 유지**: 졸업 후에도 spec 결함 발견 시 → FAILURE_LOG → 재논의 절차
- **Claude Code CLI 시작 가능**: 사용자가 로컬 폴더에 4개 파일 + reference/common_spec.md 배치 후 CLI 진입

---

### 본 세션의 사용자 통찰

> "harness engineering 은 Claude AI 세션에서 하는 게 아니라 Claude Code CLI 에서 해야 하는 거 아닌가"

→ 정확한 지적. Claude 가 spec 결정과 코드 구현의 적절한 도구 분리를 명시화. 향후 운영 흐름 (spec → CLI / 학습 → CLI / 결과 분석 → AI 세션 / 재결정 → AI 세션) 의 토대.

> "최종 파일을 로컬 컴퓨터에 저장하고 cli를 시작할려고 해"

→ 졸업 작업의 trigger. 본 세션이 \_ing 시스템에서의 마지막 작업.

---

### 본 세션의 작업 산출물

```
/mnt/user-data/outputs/
├── CLAUDE.md       (졸업판 v1.0, _ing 첫 헤더 갱신, §0 작성 상태 요약 갱신,
│                    §99 헤더 차단 의미 전환, §101 의사결정 21 추가)
├── SKILL.md        (졸업판 v1.0, frontmatter 갱신, §0 차단 표 → 운영 안내,
│                    §3.10 모듈 분리 규칙 채움, §5 파일 구조 졸업판 폴더 구조,
│                    §6.1 첫 인사 운영 모드, §7 졸업 후 운영 규칙)
├── PROGRESS.md     (졸업판, 첫 헤더 갱신, Session 2026-06-24 추가, 의사결정 21)
└── FAILURE_LOG.md  (졸업판, 첫 헤더 갱신만, entry 추가 없음 — 실패 사례 아님)
```

---

### 다음 세션 진입점

#### Claude Code CLI 첫 작업 권장

```bash
cd pipe-routing-rl-v2  # 로컬 프로젝트 폴더
claude                 # Claude Code CLI 진입
```

CLI 가 폴더 진입 시 자동으로 CLAUDE.md / SKILL.md 읽어들임. 권장 첫 명령:

```
"본 프로젝트 spec 졸업판 검토 완료. harness engineering 시작.
첫 작업: shaping/potential_based.py — PotentialBasedShaping helper class.
CLAUDE.md §16.7.5 spec 1:1 변환. unit test 도 함께 작성하여
PBS 4 대 조건 (state-only / terminal=0 / γΦ(s')-Φ(s) / γ_PBS == γ_PPO) 강제."
```

#### 권장 구현 순서 (의존성 흐름)

```
1. shaping/potential_based.py + tests/test_pbs_safety.py
   → 가장 격리된 모듈. 다른 모듈 의존 없음.

2. generators/poisson_disk.py + tests/test_generator.py
   → 다음으로 격리된 모듈.

3. generators/scenario_generator.py + scenarios/step1/*_seeds.yaml
   → poisson_disk 사용. Generator v1.0.0 완성.

4. envs/base_env.py + envs/step1_env.py + tests/test_coord_conversion.py
   → shaping, generator 사용. L-C4 unit test 동시 작성.

5. training/regression_callback.py + training/transfer_learner.py
   → 회귀 감시 + mixed curriculum. Step 1 단독에선 미사용이나 미리 구현.

6. autoresearch/optuna_study.py + wandb_callback.py + stage_runner.py
   → autoresearch 인프라.

7. training/train_step1.py + autoresearch/round_orchestrator.py
   → 모든 모듈 통합. Phase 1 Round 1 시작 가능 상태.

8. Phase 1 Round 1 실행 (L-D2 α × β 12 variants sweep)
9. 결과 분석 (wandb / Optuna importance)
10. Round 2 진행 (L-16.3-w 27 variants) 또는 spec 재논의
```

#### 학습 진행 중 다시 Claude AI 세션으로 돌아오는 시점

```
- Phase 1 Round 1 완료 후 결과 분석 (plateau 진단)
- spec 결함 발견 시 (FAILURE_LOG entry 추가, Lock 재논의)
- L-A3' 해제 (Phase 1 결과 분포 기반 임계값 결정)
- Phase 1 plateau 시 L-D3 해제 검토 (macro-action 도입 또는 Φ_future)
```

---

## Session 2026-06-25 — harness engineering 첫 세션: skill 충돌 + PBS 검증 강화

### 세션 배경

2026-06-24 \_ing 졸업 직후 Claude Code CLI 에서 harness engineering 첫 작업 진행.
권장 첫 작업인 `shaping/potential_based.py` (PotentialBasedShaping helper class) 구현 중
**CLI 자동 로드 skill 의 spec (120-dim / 27-action) 이 졸업판 SKILL.md (150-dim / 7-direction) 와 충돌**하는 사태 발견.
진단 3단계 거쳐 원인 확정 후 해결 방안 수립.
부수적으로 PBS state-only 검증 방법 보강안 도출.

FAILURE_LOG 2026-06-25 entry 에 전체 분석 기록됨.

---

### 핵심 의사결정 22 — PBS state-only 검증 강화

#### 배경

SKILL.md §3.5 의 `PotentialBasedShaping` helper class 가 PBS 조건 1 (state-only 시그니처) 을
`inspect.signature()` 의 parameter 수 (== 1) 로 검증함. 이 검증 방식의 한계:
- parameter 수만 체크하면 클로저 (closure) 로 외부 상태를 캡처해도 통과 가능
- `lambda s: f(s, some_action)` 같은 action-dependent 함수도 단일 parameter 이면 통과

#### 결론

**`inspect.signature()` 단독 검증 → 런타임 추가 검사 + unit test 의무화 조합으로 강화.**

```python
# 추가 검증 (helper class __init__ 내):
# 1. co_freevars 에 action-related 이름 포함 시 경고 로깅
if any('action' in v or 'act' in v for v in phi_fn.__code__.co_freevars):
    warnings.warn(
        "PBS 조건 1 주의: phi_fn 이 action-related 이름을 closure 로 캡처 중. "
        "state-only 여부를 수동 검증하시오.",
        PBSSafetyWarning
    )
```

`tests/test_pbs_safety.py` 에 클로저 케이스 명시적 추가 의무:
```python
# 추가해야 할 test case
def test_closure_capturing_action_raises_warning():
    action = 1
    phi_with_action = lambda s: s[action]  # action-dependent closure
    with pytest.warns(PBSSafetyWarning):
        PotentialBasedShaping(phi_fn=phi_with_action, gamma_ppo=0.99)
```

#### 근거

PBS 조건 1 위반은 shaping reward 가 action-dependent 가 되어 정책 보존 이론 붕괴 (Ng 1999).
`len(sig.parameters) == 1` 은 필요 조건이나 충분 조건 아님. 클로저 경고 + unit test 로 보강.

#### 기각된 대안

- **mypy type checking 단독**: 런타임 lambda / closure 에는 적용 불가
- **완전한 정적 AST 분석**: 클로저 의존성 분석은 AST 로도 불완전 (동적 binding 추적 불가)

#### SKILL / 코드 영향

- `shaping/potential_based.py`: `__init__` 에 `co_freevars` 경고 추가
- `tests/test_pbs_safety.py`: 클로저 케이스 추가
- SKILL.md §6.4 절대 하지 말 것: "signature parameter 수 1개 == state-only 확정으로 단정" 금지 추가

---

### 핵심 의사결정 23 — skill 시스템 충돌 해결 (Option A-1 채택)

#### 배경

CLI 첫 진입 시 자동 로드된 서버 등록 skill (v1 시절 등록, 120-dim / 27-action) 이
졸업판 SKILL.md (150-dim / 7-direction) 와 충돌.
전체 분석은 FAILURE_LOG 2026-06-25 entry 참조.

**핵심 구조적 원인**: 졸업 작업 (의사결정 21) 이 로컬 파일 갱신만 포함하고
서버 skill 갱신을 체크리스트에 포함하지 않음.

#### 후보 검토

| 옵션 | 내용 | 장점 | 단점 |
|------|------|------|------|
| **A-1. In-place 갱신** | 서버 skill → SKILL.md 내용으로 업데이트 | 단일 출처, 관리 단순 | 수동 작업 의무 |
| B. 병행 운영 | 두 출처 공존, spec 버전으로 구별 | 롤백 쉬움 | 관리 복잡도 2배 |
| C. 서버 skill 삭제 | 서버 skill 제거, 로컬 SKILL.md 만 | 충돌 원천 제거 | CLI 자동 로드 비활성화 위험 |

#### 결론

**Option A-1 (In-place 갱신) 채택.**

수립된 운영 원칙:

```
[로컬 SKILL.md = source of truth. 서버 skill = mirror.]

SKILL 갱신 순서 (절대 준수):
1. 로컬 SKILL.md 수정
2. git commit (로컬 이력 보존)
3. 서버 skill in-place 갱신 (SKILL.md 내용 그대로)
4. CLI 새 세션 진입 → 로드 skill 검증
   (obs-dim 150 / action 7-direction / 졸업 날짜 / 활성 Lock 목록)
```

#### 근거

- 단일 출처가 관리 복잡도 최소화 (B 옵션 대비)
- CLI 자동 로드 유지가 사용자 경험에 중요 (C 옵션 대비)
- 수동 작업 의무는 §0.5 체크리스트로 강제

#### SKILL.md / CLAUDE.md 영향

- SKILL.md §0.5 신설 — 서버 등록 skill 동기 의무 + 갱신 순서 + 충돌 감지 체크
- CLAUDE.md §101 졸업 후 운영 원칙에 항목 6 추가 — 서버 skill 동기화

---

### 핵심 의사결정 24 — 졸업 범위 확장 (서버 skill 동기화 의무 추가)

#### 배경

의사결정 21 의 \_ing 졸업 체크리스트:
```
1. 로컬 파일 rename
2. 각 파일 헤더 갱신
3. Cross-reference 일괄 갱신
4. git commit
```
**서버 측 등록 skill 갱신 누락** → 의사결정 23 충돌의 근본 원인.

#### 결론

**\_ing 시스템 졸업 체크리스트 v2 (의사결정 24 반영):**

```
[v2 체크리스트]
1. 로컬 파일 rename (CLAUDE_ing → CLAUDE, SKILL_ing → SKILL, PROGRESS_ing → PROGRESS)
2. 각 파일 헤더 갱신 (졸업 안내, v1.0 버전)
3. Cross-reference 일괄 갱신
4. git commit
5. ★ 서버 등록 skill in-place 갱신 (로컬 SKILL.md 내용으로)  ← 신규 추가
6. ★ CLI 새 세션 진입 → skill 로드 확인 (obs-dim / action-space / 졸업 날짜)  ← 신규 추가
```

항목 5, 6 이 신규 추가. 본 세션이 항목 6 의 첫 실증.

#### 근거

- 서버 skill 갱신은 수동 작업이므로 체크리스트 없이는 반드시 누락됨
- 항목 6 (CLI 검증) 이 동기화 완료를 자기 확인 (self-verification) 가능하게 함
- 본 사례 (의사결정 23) 가 체크리스트 필요성의 직접 증거

#### 기각된 대안

- **자동 동기화 hook (git post-commit)**: 구현 복잡도 과도. 수동 체크리스트가 현 규모에 적합.
- **로컬 SKILL.md 만 사용 (서버 skill 비등록)**: CLI 자동 로드 비활성화 위험.

---

### 본 세션의 Claude 자기 비판

- 의사결정 21 (졸업) 당시 harness engineering 환경에서 로컬 파일과 서버 skill 이 분리된다는 것을 인식하지 못해 서버 skill 갱신을 체크리스트에 누락.
- 진단 1차에서 가정 기반 추론 (디스크 확인 없이 "자동 갱신됐을 것"). 3차 이후에야 디스크 증거로 확정. **디스크 증거 우선 원칙** 의 소중함 재확인.

---

### 본 세션의 작업 산출물

```
├── FAILURE_LOG.md  (2026-06-25 entry 추가 — skill 충돌 사례 전체 기록)
├── PROGRESS.md     (본 Session 2026-06-25 추가 — 의사결정 22/23/24)
├── SKILL.md        (§0.5 신설 — 서버 등록 skill 동기 의무 + 갱신 순서)
└── CLAUDE.md       (§101 의사결정 22/23/24 추가 + 졸업 후 운영 원칙 항목 6 추가)
```

---

### 다음 세션 진입점

1. 서버 skill in-place 갱신 완료 후 (의사결정 23 Option A-1 실행)
2. harness engineering 재개: `shaping/potential_based.py` + `tests/test_pbs_safety.py`
   - 의사결정 22 반영: PBS state-only closure 경고 + unit test 클로저 케이스 추가
3. 이후 권장 구현 순서는 PROGRESS.md Session 2026-06-24 § "권장 구현 순서" 참조

---

## §101. Lock 해제 로그 (CLAUDE.md §101 미러)

| 일자 | 항목 | 결론 요약 | 본 문서 참조 |
|------|------|----------|-------------|
| 2026-04-30 | 의사결정 1~7 | 초기 v2 구조 설계 | Session 2026-04-30 |
| 2026-05-14 | 의사결정 3 폐기 → 3-r1, 3-r2 | Hierarchical 폐기, 단일 에이전트 + 전이학습 | Session 2026-05-14 |
| 2026-05-14 | 의사결정 7 폐기 | Macro 자체 폐기로 자동 폐기 | Session 2026-05-14 |
| 2026-05-14 | 의사결정 8 신규 | Step 4~6 평가 Layer 1/2 분리 | Session 2026-05-14 |
| 2026-05-14 | 의사결정 9 신규 | wandb 도입 | Session 2026-05-14 |
| 2026-05-14 | 의사결정 10 신규 | Dense reward (PBS) 도입 | Session 2026-05-14 |
| 2026-05-14 | 의사결정 11 신규 | Macro-action 도입 검토 (Phase 1 후 결정) | Session 2026-05-14 |
| 2026-06-11 | 의사결정 12 신규 | Lock 명명 규칙 spec 내장 (CLAUDE.md §99 범례) | Session 2026-06-11 |
| 2026-06-11 | 의사결정 13 신규 | 회귀 검증 체계 명시화 (BWT/Forgetting/주기 감시) — L-D1 범위 확장 | Session 2026-06-11 |
| 2026-06-18 | 의사결정 14 신규 | L-A5 해제 — 시나리오 4분할 / 난이도 축 차별화 / seed range / 임시 임계값 | Session 2026-06-18 |
| 2026-06-18 | 의사결정 15 신규 | L-A6 신설 — Procedural Scenario Generator 알고리즘 spec | Session 2026-06-18 |
| 2026-06-18 | 의사결정 16 신규 | L-D1 해제 — 전이학습 A+B 혼합 + 회귀 감시 체계 구체화 | Session 2026-06-18 |
| 2026-06-18 | 의사결정 17 신규 | L-D2 해제 — Dense Reward Φ Phase 1 spec 확정 (Φ_goal + Φ_cong, α=1.0/β=0.1, PBS 4대 조건, helper class) | Session 2026-06-18 |
| 2026-06-18 | 의사결정 18 신규 | L-C2 해제 — autoresearch 메타 (1/8 + 50%, 동시 2~3개, Round 5 상한, Optuna TPE) | Session 2026-06-18 |
| 2026-06-18 | 의사결정 19 신규 | L-16.3-w 해제 — baseline 가중치 (w1=0.1, w2=2.0, w3=50.0, sweep 27 variants) | Session 2026-06-18 |
| 2026-06-18 | 의사결정 20 신규 | L-A6 해제 — Generator v1.0.0 (Poisson disk + cuboid + rejection sampling) | Session 2026-06-18 |
| 2026-06-24 | 의사결정 21 신규 | 🎓 \_ing 시스템 졸업 — CLAUDE_ing/SKILL_ing/PROGRESS_ing → CLAUDE/SKILL/PROGRESS. 차단 의미 전환. 문서 v1.0. | Session 2026-06-24 |
| 2026-06-25 | 의사결정 22 신규 | PBS state-only 검증 강화 — closure 경고 + unit test 클로저 케이스 추가 | Session 2026-06-25 |
| 2026-06-25 | 의사결정 23 신규 | skill 시스템 충돌 해결 — Option A-1 (in-place 갱신) 채택, 로컬 SKILL.md = source of truth | Session 2026-06-25 |
| 2026-06-25 | 의사결정 24 신규 | 졸업 범위 확장 — 체크리스트 v2 (서버 skill 동기화 + CLI 검증 항목 추가) | Session 2026-06-25 |

> Lock **해제** 가 아니라 상위 **의사결정 재검토** 결과임에 유의. L-B1, L-B2, L-B3, L-B4, L-12.2, L-17.1은 "해제"가 아니라 "Hierarchical 폐기 부수효과로 항목 자체 폐기". 2026-06-18 의사결정 14, 16, 17, 18, 19, 20 은 진짜 Lock 해제.

---

**문서 버전:** v1.1
**졸업일:** 2026-06-24
**마지막 갱신:** 2026-06-25 (Session 2026-06-25 추가: 의사결정 22/23/24 — skill 충돌 + PBS 검증 강화)
**이전 갱신:** 2026-06-24 v1.0 (의사결정 21 \_ing 졸업), 2026-06-18 v0.6 (의사결정 18/19/20), 2026-06-18 v0.5 (의사결정 17), 2026-06-11, 2026-05-14, 2026-04-30 (프로젝트 시작)
**다음 갱신 시점:** Phase 1 결과 분석 후 의사결정 25 (L-A3' 해제 등)
