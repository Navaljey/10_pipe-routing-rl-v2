# 시각화 및 구동 환경 참조 문서

> CLAUDE.md에서 분리된 참조 문서. 학습 결과 시각화 스택과 구동 환경(Colab 등) 세부사항을 다룰 때 로드한다.
> Colab 작업 시 지켜야 할 경로 원칙(코드 수정 금지, git pull 선행 등)은 CLAUDE.md §18을 따른다.

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

