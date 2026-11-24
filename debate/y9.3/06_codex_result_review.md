## 판정: 해석 타당 — y9.3a의 사전 기준 미달과 y9.2a 기준 모델 유지는 타당하다. 길이별 재현율과 진폭 대조군에서 끌어낸 원인 해석은 범위를 좁혀야 한다.

## 지적

**R1. [심각도: 경미] 사전 판정은 재현되며, 보고서에 평가 정밀도를 명시해야 한다**

- **문제 및 근거** — `metrics.json` 12개와 y9.3 학습 로그 6개를 확인했다. 같은 seed의 y9.2a와 y9.3a/b 설정 차이는 `method`와 `frangi`였고, 라벨 6,919장, `split_seed=0`, `v3_nopl`, 90 epoch, `no_test=True`가 일치했다. 저장된 val macro F1에서 y9.3a의 seed별 차이는 **−0.00516 / −0.00153 / −0.00328**, 평균 **−0.00333**이었다. 저장된 혼동행렬의 `none→Scratch`는 **156→210**, 사후 평가에서는 **156→212**다. 체크포인트 선택은 FP16 autocast, 사후 평가는 FP32 경로를 사용한다. 사전 판정 1~3번 미달, 4번 통과는 유지된다. [저장 지표 예시](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/resnet18/p0.05/seed1/y9.3a_ssl50_ft30/metrics.json:188>), [학습 로그 예시](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/log_y9.3a_p0.05_s1.txt:101>), [체크포인트 평가](<C:/Users/annan/OneDrive/바탕 화면/umm/wafer/engine.py:42>), [사후 평가 코드](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/y9.3_step1/eval_step1.py:39>)
- **제안** — 05의 판정표에 ‘FP32 사후 평가’를 명시하라. 03의 ‘20% 이상 감소’에 해당하는 정수 상한은 **124건**이다. 평가 코드는 이 조건을 올바르게 적용했다. [사후 판정표](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/y9.3_step1/eval_step1.txt:37>), [판정 코드](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/y9.3_step1/eval_step1.py:94>)

**R2. [심각도: 중요] 길이별 재현율 표는 실제 채널에 들어간 능선 길이와 다른 값을 사용한다**

- **문제 및 근거** — 학습 캐시는 `τ=0.2`인 Frangi 능선에 길이를 부여한다. 사후 구간표는 `Fr.line_weight(nat, return_parts=True)`의 기본값인 **불량 die 연결 성분**의 최대 길이를 사용한다. 이 표에는 val Scratch 119장 중 114장이 들어가며, `10~14 die, die<1200` 구간은 **0.70→0.70**이다. `10 die 미만, die≥1200`의 **0.67→0.48**은 웨이퍼 9장에 대한 세 seed 평균이다. 이 결과는 불량 성분 길이에 따른 재현율 패턴을 보여준다. 모델이 능선 길이를 활용했다거나, 긴 Scratch를 더 잡으면서 생긴 오탐도 긴 성분에서 나왔다는 해석에는 근거가 부족하다. [학습 캐시 구현](<C:/Users/annan/OneDrive/바탕 화면/umm/wafer/frangi.py:190>), [사후 길이 계산](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/y9.3_step1/eval_step1.py:52>), [구간별 분모와 재현율](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/y9.3_step1/eval_step1.txt:29>)
- **제안** — 현재 표에는 측정 정의와 분모를 붙여라. 기전을 확인하려면 실제 능선 정의로 Scratch 구간을 다시 집계하고, 신규 및 해소된 `none→Scratch`와 `Loc→Scratch`의 능선 길이와 채널 응답도 비교하라.

**R3. [심각도: 중요] 같은 재현율 비교는 Scratch 확률의 순위 진단이다**

- **문제 및 근거** — `P(Scratch)`에 단일 임계값을 적용했을 때 오탐은 y9.2a 대비 **57→48 / 35→20 / 67→78**이다. AP 차이도 **+0.028 / +0.041 / −0.011**로 방향이 엇갈린다. 세 seed가 같은 val 웨이퍼를 평가하므로 합계 **159→146**은 seed별 결과를 요약한 값이다. 이 비교는 `P(Scratch)`의 순위를 진단한다. 9클래스 argmax에서 Scratch 로짓을 보정했을 때의 오탐이나 macro F1은 이 수치로 확인할 수 없다. [임계값 계산](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/y9.3_step1/eval_step1.py:78>), [seed별 원자료](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/y9.3_step1/eval_step1.txt:2>)
- **제안** — 05의 ‘순위 품질은 조금 나을 수 있다’는 해석은 탐색 관찰로 유지하라. 후속으로 로짓 보정을 평가한다면 Scratch 로짓과 가장 높은 경쟁 클래스 로짓의 차이를 사용하고, 달성 재현율과 none·Loc 오탐, 9클래스 macro F1을 함께 보고하라.

**R4. [심각도: 중요] `k·V`의 성능 하락은 이번 감쇠 설정에 한정된다**

- **문제 및 근거** — `k=0.0757`은 5% 학습 세트의 원본 불량 die에서 평균을 맞춘 값이다. 캐시 점검에서 전체 픽셀 평균은 길이 가중 **0.0037**, 균일 감쇠 **0.0036**으로 비슷하지만, 비영 웨이퍼 비율은 **93% / 100%**였고 두 채널의 국소 대비와 양자화 후 값 분포도 달라질 수 있다. y9.3b의 Scratch F1 **−0.033**과 소형 Scratch 재현율 **−0.164**는 이 감쇠·양자화 설정의 결과다. y9.3a의 y9.3b 대비 macro 이득 **+0.0076**도 seed 2의 **+0.0195**에 크게 의존한다. [감쇠 및 양자화 구현](<C:/Users/annan/OneDrive/바탕 화면/umm/wafer/frangi.py:190>), [캐시 점검](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/y9.3_step1/check_cache.txt:1>), [대조군 판정](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/y9.3_step1/eval_step1.txt:44>)
- **제안** — “이번 `k·V` 설정에서는 Scratch 검출 성능이 떨어졌다”로 좁혀라. 진폭 자체의 효과를 후속 주장에 쓰려면 감쇠 강도별 결과와 양자화 후 응답 분포를 확인하라.

## 이 결과로 말할 수 있는 것과 없는 것

- **말할 수 있음** — 0단계에서 불량 die 성분 후보는 끊긴 선 기준에 미달했다. 통과한 능선 후보 중 5% 학습 세트의 Scratch 대 전체 AP로 `τ=0.2`를 골랐고, 이 채널을 쓴 y9.3a는 1단계에서 미달했다. FP32 사후 평가에서 `none→Scratch`는 seed별로 **20 / 17 / 45건 해소**, **83 / 39 / 16건 신규**였다. 기존 오탐을 일부 해소했지만 전체 오탐은 늘었다. [0단계 후보 선택](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/y9.3_step0/verify_length.txt:26>), [짝지은 오탐 비교](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/y9.3_step1/eval_step1.txt:8>)
- **판단 보류** — 절대 길이 정보의 일반적 효용, 3채널 방식의 성능, WaFixMatch 재현값 대비 test macro F1 및 소수 클래스 하한 충족 여부는 이번 실험으로 판단할 수 없다. 소수 클래스 기준 통과는 y9.2a 대비 val 평균 차이에 한정된다.

## 다음에 꼭 확인할 것

- 길이 가설을 이어갈 경우, 추가 학습 전에 저장된 체크포인트로 **신규·해소 오탐의 실제 능선 길이와 응답**을 비교하라. 짧은 오탐이 해소됐는지, 어떤 사례가 그 자리를 채웠는지부터 확인할 수 있다.
- 프로젝트 최종 비교는 모델과 설정을 고정한 뒤 동일 split의 test에서 WaFixMatch 재현값 대비 macro F1과 소수 5개 클래스의 F1을 함께 보고하라.

검토에는 저장된 지표, 로그, 코드와 캐시 점검 기록을 사용했다. 파일 수정, 재학습, 체크포인트 재추론은 수행하지 않았다.