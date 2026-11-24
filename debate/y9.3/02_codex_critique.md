## 판정: 수정 후 진행

길이 가중 Frangi는 검증할 가치가 있다. 0단계에서 실제 채널에 사용하는 길이 정의를 검증하고, 1단계에서 짧은 선 오탐 감소와 Scratch 검출 기준의 이동을 구분할 수 있게 판정 기준을 보완해야 한다.

y9.2의 출발점은 확인했다. 6개 `metrics.json`과 동일한 사후 val 평가에서 Frangi의 0채널 대비 평균 차이는 macro F1 **+0.024**, Scratch F1 **+0.143**, die<1200 Scratch 재현율 **+0.284**였다. y9.3의 주 대조군으로 y9.2a를 재사용하는 데 동의한다. [seed 0 y9.2a metrics](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/resnet18/p0.05/seed0/y9.2a_ssl50_ft30/metrics.json:188>), [3-seed 사후 평가](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/y9.2_step1/eval_step1_s012.txt:38>)

## 지적

**C1. [심각도: 중요] 30 die의 근거가 새 채널의 길이 정의에 그대로 적용되지 않는다**

- 문제 — 제안 채널은 모든 8-연결 불량 성분에 길이를 부여한다. 기존 분석은 최소 3점과 길쭉함 ≥3을 만족하는 성분만 남기고 PCA 투영 범위에 `+1`을 더했다. 둥근 덩어리나 선이 붙은 큰 군집도 새 정의에서는 큰 가중치를 받을 수 있다. Frangi 능선으로 성분을 바꾸면 길이의 의미가 다시 달라진다.
- 근거 — `30+`의 학습 Scratch 비율 1.0은 필터를 거친 Scratch/none 중 **5/5**다. 같은 정의의 val에서는 **4/5**였다. 분석은 원본 한 변이 96 이하인 웨이퍼만 포함한다. [길이 측정 코드](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/y9.2_step1/line_length_rule.py:26>), [학습 집계](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/y9.2_step1/line_length_rule.txt:2>), [val 집계](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/y9.2_step1/line_length_rule.txt:28>)
- 제안 — 0단계 전에 `+1`, 단일점·무불량 처리, 길쭉함 필터, 주변 응답에 길이를 할당할 범위를 명시하라. 30을 고정할 경우 포화점으로 명시하고, **실제 채널에 쓰는 정의**로 학습 집계와 점수 분포를 다시 산출하라.

**C2. [심각도: 중요] 0단계가 끊긴 Scratch 보존을 요구하지 않는다**

- 문제 — 끊긴 20 die 선은 합성 목록에 있지만 합격 조건에 없다. 연속선 4개와 산발 불량 한 맵을 통과하고 Scratch/none 전체 AP가 조금 올라가도, 짧거나 끊긴 실제 Scratch를 크게 약화시키는 후보가 선택될 수 있다. `{V ≥ τ}` 후보의 `τ` 탐색 범위와 동률 처리도 미정이다.
- 근거 — y9.2 Frangi는 같은 두께의 선에서도 각도에 따라 응답이 달랐다. 길이 분석의 학습 Scratch **47장 중 11장**은 측정 길이가 10 die 미만이었다. 이 11장이 실제로 끊긴 Scratch인지는 해당 집계에서 확인할 수 없다. 기존 캐시와 동일하게 원본 한 변이 96을 넘는 사례도 다루려면 원본 pkl 경로가 필요하다. [Frangi 합성 검증](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/y9.2_step0/verify_frangi.txt:3>), [학습 길이 분포](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/y9.2_step1/line_length_rule.txt:2>), [원본 격자 캐시 경로](<C:/Users/annan/OneDrive/바탕 화면/umm/wafer/frangi.py:87>)
- 제안 — 각도·두께·웨이퍼 크기·불량 밀도를 바꾼 합성 사례에서 **96 리사이즈와 uint8 양자화 후** 점수를 확인하라. 끊긴 긴 선의 응답 보존 기준을 사전에 정하고, 실제 학습 세트에서는 짧은 Scratch의 점수 변화와 Scratch 대 전체 나머지 클래스 AP도 보라. 능선 후보를 비교한다면 `τ` 후보를 미리 고정하고, 후보 선택에 쓴 학습 AP는 탐색 결과로 기록하라.

**C3. [심각도: 중요] 관찰된 오탐 특성만으로 ‘짧은 직선 지름길’을 확인하기 어렵다**

- 문제 — `26 px 대 16 px`는 96×96에서 **면적이 가장 큰 성분**의 `4√λmax`다. Frangi 최댓값을 만든 성분의 원본 die 길이는 따로 측정하지 않았다. 높은 Scratch 확률도 모델이 그 성분을 근거로 삼았는지 알려주지 않는다. 새 채널의 이득은 길이에 따른 선택적 감쇠나 채널 전체의 진폭 변화에서 모두 생길 수 있다.
- 근거 — 기존 분석은 Frangi 최댓값과 성분 길이를 따로 집계한다. 세 seed 공통 오탐 19장 중 **17장**은 적어도 한 seed의 y9.2b 또는 ver8에서도 오탐이었다. [오탐 특징 계산](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/y9.2_step1/fp_none.py:46>), [공통 오탐 집계](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/y9.2_step1/fp_none.txt:23>)
- 제안 — 먼저 y9.2a 오탐에서 Frangi 피크에 실제로 할당되는 성분의 원본 길이를 집계하라. 길이 가중치 자체의 효과를 주장하려면 학습 세트에서 평균 진폭을 맞춘 `k·V`를 같은 조건으로 fine-tune해 비교하라.

**C4. [심각도: 중요] 현재 합격 기준은 미세한 변동과 Scratch 판정 이동도 통과시킨다**

- 문제 — macro가 세 seed에서 아주 조금씩 오르고 `none→Scratch`가 세 seed 합계로 한 건만 줄어도 두 조건을 통과한다. Scratch F1과 소형 Scratch 재현율은 허용 범위 안에서 하락할 수 있고, `Loc→Scratch`가 늘어도 오탐 조건은 통과한다. 또 y9.2a 대비 허용 하락폭은 프로젝트의 **WaFixMatch 재현값 대비** 클래스별 하한과 별개의 기준이다.
- 근거 — y9.2a 자체의 `none→Scratch`는 **56 / 33 / 67건**, 소형 Scratch 재현율은 **0.62 / 0.51 / 0.74**였다. 같은 val 웨이퍼를 세 모델이 평가한다. 소형 Scratch는 **61장**이라 한 장이 seed별 재현율 약 **0.016**에 해당한다. 저장된 `best.pt`는 매 epoch의 val macro로 고른다. [seed별 평가](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/y9.2_step1/eval_step1_s012.txt:2>), [소형 Scratch 분모](<C:/Users/annan/OneDrive/바탕 화면/umm/debate/y9.2/07_final.md:12>), [체크포인트 선택](<C:/Users/annan/OneDrive/바탕 화면/umm/wafer/engine.py:68>)
- 제안 — 실행 전에 의미 있는 최소 macro 변화와 오탐 감소 폭을 정하고, seed별 paired 차이와 오탐이 해소된 건수·새로 생긴 건수를 보고하라. Scratch AP와 **같은 Scratch 재현율에서의 none 오탐**을 함께 비교하고 `Loc→Scratch` 및 짧은 Scratch의 손실을 확인하라. 3 seed 판정은 후보 선별에 쓰고, 최종 판정에는 사전에 고정한 WaFixMatch 비교 기준을 적용하라.

**C5. [심각도: 중요] 기존 대조군 재사용에 필요한 실행 조건을 명시해야 한다**

- 문제 — 현재 로더는 `args.frangi == "frangi"`일 때만 캐시를 읽는다. 새 모드 문자열만 추가하면 0채널 경로로 들어갈 수 있다. 기존 캐시 작성기를 그대로 호출하면 y9.2 캐시 경로에 쓰며, 현재 인덱스 범위는 5% 라벨과 SSL 표본보다 넓다.
- 근거 — 학습 로더에서 `table=None`은 0채널이다. 캐시 작성기는 **라벨이 있는 전체 웨이퍼 ∪ SSL 표본**을 사용하고, `v3_nopl`에서도 SSL 표본과 전체 train의 임베딩을 군집 진단에 사용한다. y9.2의 실제 학습은 `ft30` **3라운드, 총 90 epoch**다. [채널 로더](<C:/Users/annan/OneDrive/바탕 화면/umm/wafer/selfsup.py:186>), [0채널 분기](<C:/Users/annan/OneDrive/바탕 화면/umm/wafer/models.py:138>), [캐시 경로와 인덱스](<C:/Users/annan/OneDrive/바탕 화면/umm/wafer/frangi.py:76>), [군집 진단 경로](<C:/Users/annan/OneDrive/바탕 화면/umm/wafer/selfsup.py:230>)
- 제안 — y9.3 전용 캐시를 쓰고 첫 학습 전에 채널 값과 인덱스 커버리지를 확인하라. 기존 루프를 유지한다면 `ds.train`, `ds.val`, 해당 seed의 SSL 표본을 포함하고 test 캐시는 결정 후 생성해도 된다. seed별 인코더·SSL 표본·라벨 인덱스와 **90 epoch**, val 체크포인트 선택, 사후 평가 정밀도를 맞춰라. y9.2와 동일한 test 평가 차단도 확인하라. [인코더·표본 선택 경로](<C:/Users/annan/OneDrive/바탕 화면/umm/wafixmatch.py:137>), [사후 평가 경로](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/y9.2_step1/eval_step1.py:31>)

**C6. [심각도: 중요] 절대 길이 결과는 크기 독립적인 라벨 규칙을 뒷받침하기에 부족하다**

- 문제 — 절대 길이가 상대 길이보다 유용하다는 결과는 후보 설계의 근거가 된다. 그룹 차이 `0.006`은 Scratch 판정이 갈리는 구간의 차이를 작게 보이게 한다.
- 근거 — 집계는 표본 수 20 이상인 셀만 비교하고 구간 표본 수로 가중한다. 실제 **10–<12 die**의 Scratch 비율은 크기군별 **2.4% / 8.9% / 10.3%**, **12–<15 die**는 **10.1% / 13.8% / 22.2%**다. 지름을 함께 넣으면 로그 손실은 **0.02738 → 0.02686**으로 내려간다. 이 분석은 5% 학습과 val을 합쳤으므로 현재 val은 가설 형성에 이미 쓰였다. [집계 및 회귀 코드](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/y9.2_step1/line_length_rel.py:40>), [크기군별 결과](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/y9.2_step1/line_length_rel.txt:6>)
- 제안 — 가설을 “절대 길이 가중치가 기존 Frangi의 오탐을 줄이는가”로 좁혀라. 길이·크기 구간별 분모와 재현율을 보고하고, 설계와 판정을 고정한 뒤 test에서 최종 비교하라.

## 빠진 대조군·검증

- **진폭을 맞춘 `k·V`**는 길이별 감쇠의 효과를 분리하는 대조군이다. 0단계에는 `max(V)`, `max(g(L))`, `max(V·g(L))`를 함께 두면 곱셈이 길이 점수 단독보다 추가 정보를 주는지 확인할 수 있다.
- **기존 Frangi의 같은 재현율에서의 오탐 수**를 비교하라. 1단계의 argmax 결과에 더해 판정 기준 이동만으로 얻을 수 있는 감소 폭을 확인할 수 있다.

## 더 싸거나 더 나은 대안

전체 캐시를 만들기 전에 기존 y9.2a 체크포인트에 0단계의 연속선·끊긴 선·산발 불량을 넣고, 원본 `V`와 길이 가중 `V·g(L)`에서 Scratch 점수가 얼마나 움직이는지 짝지어 볼 수 있다. 기존 평가 스크립트는 두 채널을 직접 받는 경로가 있다. 이 검사는 재학습 성능의 추정치로 쓰기보다, 감쇠가 짧은 오탐과 실제 Scratch에 비슷하게 작용하는 후보를 걸러내는 데 쓰면 된다. [두 채널 직접 입력 평가](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/y9.2_step1/eval_step1.py:49>)

코드, 로그, 저장된 지표를 읽어 검토했다. 파일 수정과 추가 학습은 없었다.