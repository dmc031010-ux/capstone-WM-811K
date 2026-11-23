## 판정: 과장

Frangi를 추가한 fine-tune은 두 seed의 val에서 Scratch를 개선했다. 다만 굵기 민감성이 완화됐다는 해석은 조작 결과와 일관되지 않으며, 사전에 합의한 SSL 단계 진행 기준은 seed 0의 Random F1에서 미달했다.

## 지적

**R1. [심각도: 중요] Random 하락은 사전 기준 미달이며, 표본 수로 환산한 해석은 부정확하다.** seed 0의 `a − b`는 Random **−0.03438**, Near-full **−0.02956**이다. 따라서 Random만 미달이고 Near-full은 통과다. Random 87장 중 정답은 74→69장, 오탐은 6→6장이다. Near-full은 정답 12장으로 같고 오탐이 1→2장으로 늘었다. “약 1.5장 상당” 대신 이 수치를 보고하는 편이 정확하다. `b − ver8`은 입력 구조와 실행 시점이 다른 모델 간 차이여서 동일 조건의 학습 잡음 추정치로 쓰기 어렵다. 기준을 재검토할 경우 현재 판정은 미달로 기록하고, 추가 비교 규칙을 결과 확인 전에 정해야 한다. [진행 기준](/C:/Users/annan/OneDrive/바탕%20화면/umm/debate/y9.2/03_response.md), [seed 0 Frangi 혼동행렬](/C:/Users/annan/OneDrive/바탕%20화면/umm/runs/resnet18/p0.05/seed0/y9.2a_ssl50_ft30/metrics.json:203), [seed 0 대조군 혼동행렬](/C:/Users/annan/OneDrive/바탕%20화면/umm/runs/resnet18/p0.05/seed0/y9.2b_ssl50_ft30/metrics.json:203)

**R2. [심각도: 중요] 굵은 선의 Scratch 확률 상승과 굵기 민감성 완화는 구분해야 한다.** 동일 구조인 `b→a`에서 1 die와 2 die의 평균 확률 격차는 네 조건 중 세 조건에서 커졌다. 예를 들어 seed 0의 300~1199 die 배경에서는 2 die 확률이 `0.04→0.17`로 올랐지만, 1 die 확률은 `0.31→0.58`로 더 많이 올라 격차가 `0.27→0.41`로 벌어졌다. 3 die는 전 조건에서 `0.00~0.01`이다. 같은 배경과 중심선을 쓰고 조작 후 Frangi를 재계산한 설계에는 동의한다. 결론은 “1 die와 일부 2 die에서 Scratch 점수가 상승했다”로 좁히고, 완화 여부는 같은 배경에서 두께 증가에 따른 점수 하락폭이 줄었는지 짝지어 확인해야 한다. 실제 학습 라벨에 3 die Scratch가 드물다는 설명은 추측이다. [조작 결과](/C:/Users/annan/OneDrive/바탕%20화면/umm/runs/y9.2_step1/eval_step1.txt:9), [조작 코드](/C:/Users/annan/OneDrive/바탕%20화면/umm/runs/y9.2_step1/eval_step1.py:70)

**R3. [심각도: 중요] 효과의 근거는 Frangi 대 0 채널 비교이며, 오류 해석에도 같은 대조군을 써야 한다.** 두 조건은 같은 seed의 `masklat8_e50.pt`와 2채널 stem을 쓰고 새 채널 가중치를 0으로 시작한다. 네 실행은 각각 90 epoch이며 `v3_nopl=True`, 전 epoch `Lc=0`이었다. 이 비교에서 Scratch F1은 **+0.154 / +0.131**, AP는 **+0.132 / +0.193**이고, 저 die Scratch 61장 중 정답은 **17→38장 / 21→31장**이다. seed 0의 `b→a`에서는 Scratch→none이 `47→20`, none→Scratch도 `64→56`으로 줄어 정밀도가 `0.394→0.510`으로 올랐다. “none 오탐이 늘어 정밀도 이득이 작다”는 문장은 ver8 대비에만 해당한다. seed 1에서는 `b`와 ver8의 Scratch 정답이 이미 `46 대 59장`으로 달라, macro가 비슷하다는 이유로 두 모델을 동일하게 볼 수 없다. Scratch 짝비교도 `a 대 b`를 직접 보고해야 한다. [인코더 경로](/C:/Users/annan/OneDrive/바탕%20화면/umm/wafixmatch.py:151), [미세조정 초기화](/C:/Users/annan/OneDrive/바탕%20화면/umm/wafer/selfsup.py:185), [val 재평가](/C:/Users/annan/OneDrive/바탕%20화면/umm/runs/y9.2_step1/eval_step1.txt:2)

**R4. [심각도: 중요] 체크포인트 선택과 사후 평가의 수치 출처를 명시해야 한다.** 네 y9.2 `metrics.json`에 `no_test=true`가 있고 test 지표는 없다. 다만 과거 ver8 로그에는 val 최고점 갱신 시 test를 평가한 기록이 있으므로 “test 안 봄”은 **이번 y9.2 실행과 사후 평가**에 한정하는 게 정확하다. 학습 중 val 평가는 FP16 autocast, 사후 평가는 autocast 없이 수행했다. 예를 들어 seed 1 ver8 Scratch F1은 저장된 `metrics.json`에서 `0.4291`, 사후 평가에서 `0.431`이다. seed 1의 `a − b` macro도 저장값 기준 `+0.02729`, 사후 평가 로그 기준 `+0.028`이다. 전체 판정은 두 경로에서 같지만, 문서에는 어느 경로의 값으로 기준을 판정했는지 적어야 한다. 또한 같은 best-epoch 선택 규칙이 같은 크기의 낙관 편향을 보장하지는 않는다. 다만 `metrics.json`의 마지막 30 epoch 평균에서도 `a − b` macro가 `+0.0175 / +0.0298`이므로, 관측된 이득이 최고 epoch 한 점에만 의존하지는 않는다. [학습 중 평가](/C:/Users/annan/OneDrive/바탕%20화면/umm/wafer/engine.py:42), [사후 평가](/C:/Users/annan/OneDrive/바탕%20화면/umm/runs/y9.2_step1/eval_step1.py:38), [과거 ver8 로그](/C:/Users/annan/OneDrive/바탕%20화면/umm/runs/log_ver8_p0.05_s1.txt:128), [seed 1 Frangi metrics](/C:/Users/annan/OneDrive/바탕%20화면/umm/runs/resnet18/p0.05/seed1/y9.2a_ssl50_ft30/metrics.json:188), [seed 1 대조군 metrics](/C:/Users/annan/OneDrive/바탕%20화면/umm/runs/resnet18/p0.05/seed1/y9.2b_ssl50_ft30/metrics.json:188)

## 이 결과로 말할 수 있는 것과 없는 것

- 고정된 val 분할과 두 seed에서 Frangi를 fine-tune 입력에 추가했을 때, 0 채널 대조군보다 Scratch F1·AP·저 die 재현율과 macro F1이 모두 높았다.
- 이 비교로 Frangi가 다른 불량 파생 채널보다 우수한지, 굵기 민감성 완화가 실제 Scratch 개선의 원인인지, 2채널 SSL이 추가 이득을 주는지, WaFixMatch 대비 test 목표와 소수 클래스 하한을 충족하는지는 판정할 수 없다.

## 다음에 꼭 확인할 것

1. SSL 진행 기준은 **미달**로 기록한다. 추가 검증을 결정한다면 test 공개 전에 평가 정밀도와 seed 규칙을 고정하고, 같은 `best.pt`의 `a 대 b` 예측으로 Random·Near-full의 정답과 오탐 변화, 저 die Scratch의 새 정답과 새 누락을 짝지어 집계한다.
2. 굵기 조작은 웨이퍼별 `1→2`와 `1→3` 점수 하락폭을 `a 대 b`로 비교한다. 이 결과를 근거로 2채널 SSL을 검토한다면, 가려진 die만 바꿨을 때 보이는 Frangi 값이 얼마나 달라지는지도 먼저 측정한다.

학습과 추론은 재실행하지 않았다. 혼동행렬에서 복원 가능한 수치는 직접 계산했고, AP와 합성선 확률은 저장된 평가 로그와 평가 코드를 대조했다. 파일은 수정하지 않았다.