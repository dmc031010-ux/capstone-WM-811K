# 01. 동료 저장소 (Hanbandoche/wafer-defect) 방향 평가 — Claude

동료 코드 사본: `debate/peer-wafer-defect/peer_repo/` (README.md, EXPERIMENTS.md, ROADMAP.md, wafixmatch.py, wafer/models.py, wafer/selfsup.py).
동료 저장소에는 metrics.json·로그가 없다. 아래 수치는 동료 문서(README / EXPERIMENTS)에 적힌 값이며 우리가 재현하지 않았다.
우리 저장소와 같은 코드 기반 (ver7 까지 수치가 우리와 동일: ver7 seed 0 0.8314, seed 1 0.783) 에서 갈라졌다.

## 동료 주장 요약
- `ver9.91` = ver7 인코더 그대로 + fine-tune 단계만 변경:
  1. `OrientedLineBranch` (models.py:250): 96×96 결함 맵에 8방향 9×9 / 15×15 선 필터 + 이방성 + MaxPool 보조 스트림 → 6×6
  2. `IterativeSpatialBeliefAttention`: layer3 (6×6) 에 2D 상대위치 + 9클래스 belief 3회 반복 + 보조 손실 (aux_belief_w 0.3)
  3. `SpatialEvidencePooling` (models.py:318): layer4 3×3 의 평균 풀링을 0-초기화 gated attention 풀링으로
  4. 레시피: sample_pow 0.35, label smoothing 0.05, attn_wd 0.02, warmup 5 + cosine, grad clip 1.0, hflip
- 결과 (문서 값): ver9.91 2-seed **0.8527 ± 0.0001, Scratch 0.6777 ± 0.0000** (ver7 2-seed 0.8074 ± 0.0239, Scratch 0.4601).
  같은 fine-tune 을 SimCLR 인코더에 붙인 ver3.991 = **0.8674** (seed 0). 이후 Phase 5 목표는 "마스킹 사전학습으로 SimCLR 0.8674 돌파".

## 내 평가: 방향은 좋다 (특히 fine-tune 쪽). 단 수치 신뢰성과 목표 설정에 문제가 있다.

### 좋은 점 (우리 분석과 일치)
- G1. 우리 ver8 분석이 찾은 원인과 정확히 맞는 처방이다.
  - 우리: 평균 풀링이 국소 선을 정상 칸에 희석 (#13 로짓 분해, runs/hough_eda/decomp13.py) → 동료 ③ EvidencePooling.
  - 우리: 놓친 Scratch 는 특징 공간에서 이미 none/Loc 이웃 = 인코더 표현 문제 (runs/ver8_eda/numbers.txt) → 동료 ① 은 인코더를 우회하는 선 전용 스트림.
  - 우리: 균형 샘플링 1:1 이 none 오탐을 키움 → 동료 sample_pow 0.35.
- G2. 우리 ver8.3 (클래스별 풀링) 은 효과가 없었는데 (0.836, Scratch 0.469, seed 0), 동료는 풀링 + 선 스트림 + 공간 어텐션을 합쳐 +0.045 를 주장한다. 개선 폭이 seed 잡음 (±0.024) 보다 크다.

### 문제점
- P1. **두 seed 결과가 거의 똑같다.** macro ± 0.0001, Scratch ± 0.0000. Scratch F1 0.6777 = 164/242 이므로 두 seed 모두 TP 82, FN+FP 78 이 정확히 같아야 한다.
  ver7 은 seed 에 따라 0.048 차이가 났다 (사전학습 데이터 10만 장, 인코더, 초기값이 seed 마다 다름). 레시피로 분산이 줄 수는 있어도 소수점 넷째 자리까지 같은 것은 의심스럽다.
  가능성: seed 1 이 seed 0 인코더·결과를 재사용 (캐시 경로), 같은 실험을 두 번 집계, 문서 오기. metrics.json 과 seed 별 혼동행렬로 확인해야 한다.
- P2. **test 로 설계를 고른 흔적.** EXPERIMENTS.md 의 결정 근거가 대부분 test macro 다 (예: layer4/3/2 = 0.825/0.829/0.818 → "layer3 최적 확정", ver9.8 기각, B 시리즈 선택). 30개 가까운 변형을 test 로 비교·선택했으면 최종 0.8527 은 낙관적으로 부풀려진 값이다. val 수치가 거의 보고되지 않는다.
- P3. **대부분 seed 0 한 번.** ver9.9B2 (0.8479), ver9.9A/B, ver3.991 (0.8674), Phase 5 의 L1~L4 모두 seed 0 단일. ver7 seed 잡음이 0.05 인데 0.01~0.02 차이로 채택·기각을 정했다.
- P4. **"순정 학습 100%" 표현이 코드와 다르다.** ver9.9B2 에도 grad_clip 1.0 과 belief 보조 손실 0.3 이 켜져 있다 (wafixmatch.py:376-377). 구조 효과와 학습 트릭이 완전히 분리되지 않았다.
- P5. **목표 설정.** 같은 fine-tune 에서 SimCLR 인코더 (0.8674) 가 우리 마스킹 인코더 (0.8527) 보다 높다. 즉 성능 향상의 주인공은 fine-tune 구조이고, 마스킹 사전학습은 오히려 손해다. Phase 5 (마스킹으로 SimCLR 돌파) 에 사전학습 4시간짜리 실험을 계속 쓰는 것보다,
  (a) SimCLR + ver9.91 을 최종 후보로 두고 seed 를 늘리는 것, (b) "마스킹의 장점 = Scratch" 를 die 수 구간별로 엄밀히 보여주는 것이 더 효율적일 수 있다. (단, 방법론 독창성이 과제 요구라면 Phase 5 의 의미가 있다 — 사용자 판단 필요.)
- P6. **선 필터가 die 수 적은 웨이퍼의 Scratch 를 고치는지 불명.** 선 필터 폭이 1~2px 선에 맞춰져 있다 (models.py:292, σ 0.14 → 9×9 에서 약 0.6px). 우리 조작 실험은 die 가 크면 1 die 선이 약 3.5px 로 그려져 놓친다는 것이었다 (runs/hough_eda/fewdie_test.txt). 동료는 Scratch 정답 58.5 → 82/120 을 보고하지만 die 수 구간별 분석은 없다.

## 제안 (우리가 할 일)
1. 동료에게 요청: ver9.91 seed 0·1 의 metrics.json (val 포함), seed 별 혼동행렬, 인코더 경로 확인 (P1).
2. 우리 환경에서 재현 (ver9.1 로 등록): 동료 models.py 의 세 모듈 + 레시피를 우리 코드에 이식하고, ver9 인코더 (= ver8 = ver7) 로 seed 0·1 fine-tune (seed 당 약 30분, 사전학습 없음).
   - 판단 기준: **val macro** 와 test macro 를 둘 다 보고. die 수 적은 웨이퍼의 Scratch 재현율도 함께 (P6).
   - 같은 seed 두 번 실행해 결정성 확인 (P1 의 원인 분리).
3. 재현되면 seed 2 추가와 SimCLR 인코더 버전 (ver3.991 대응) 으로 P5 판단 근거 마련.
비용: 1~2 는 약 1.5시간 (GPU). 이식 코드는 약 300줄.
