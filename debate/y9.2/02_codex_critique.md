## 판정: 수정 후 진행 — Frangi 채널의 가설은 검증할 가치가 있지만, 캐시 범위·Hessian 정의·마스킹 검증·val 기준을 고정한 뒤 50 epoch 사전학습에 들어가야 한다.

## 지적

**C1. [심각도: 치명] 캐시 범위와 2채널 입력 경로를 함께 확정해야 한다**

- 문제: ‘라벨 데이터’가 학습 라벨 6,919장을 뜻한다면, 현재 fine-tune은 첫 라운드부터 캐시에 없는 인덱스를 읽는다. ver7.3 분기를 그대로 사용하면 이미 만들어 둔 Frangi 채널에 팽창 경로까지 적용된다.
- 근거: `v3_nopl`이어도 클러스터 진단이 전체 `ds.train` 138,359장을 읽고, `_Tracker`는 매 epoch 전체 val을 평가한다. ver7.3은 사전학습 뷰에서 팽창 채널을 붙이고 분류 모델에는 `AddDilated`를 삽입한다. 공통 `to_input`은 모든 채널에 `×0.5`를 적용한다. [selfsup.py](/C:/Users/annan/OneDrive/바탕%20화면/umm/wafer/selfsup.py:220), [engine.py](/C:/Users/annan/OneDrive/바탕%20화면/umm/wafer/engine.py:42), [ver6.py](/C:/Users/annan/OneDrive/바탕%20화면/umm/wafer/ver6.py:128), [models.py](/C:/Users/annan/OneDrive/바탕%20화면/umm/wafer/models.py:122)
- 제안: 전체 train 진단을 유지하면 전체 라벨 인덱스와 두 seed의 SSL 표본 합집합을 캐시에 포함한다. 진단을 생략한다면 `lab_idx`, SSL, val, 최종 test의 실제 호출 인덱스를 명시한다. 팽창 플래그를 끄고 SSL 온라인·타깃 인코더와 fine-tune에 동일한 2채널 stem을 사용한다. 첫 채널은 기존 `[0,1]` 변환을 유지하고 Frangi는 uint8에서 복원할 때 한 번만 정규화한다. 학습 전 SSL 한 배치, fine-tune 한 배치, 체크포인트 로드, val 추론까지 확인한다.

**C2. [심각도: 중요] 제안한 Frangi 수식과 96 초과 처리의 의미가 달라진다**

- 문제: `skimage.filters.frangi` 호출과 제안서의 σ² 정규화 Hessian은 서로 다른 연산이다. 또한 캐시는 H와 W를 각각 96으로 리사이즈하므로, 비정방형 격자에서 `96/max(H,W)` 하나로 두 축의 die 척도를 맞출 수 없다.
- 근거: scikit-image v0.25.2 구현은 Hessian에 명시적으로 σ²를 곱하지 않고, `gamma=None`이면 첫 σ의 norm 최댓값으로 gamma를 정해 이후 σ에도 사용한다. 기존 캐시는 행·열을 독립적으로 nearest 샘플링한다. 실제 캐시를 집계한 결과 96 초과는 35,083장, 그중 비정방형은 34,908장이었다. 전체 라벨에서는 0.62%지만 Scratch는 60/1,193장(5.0%), Edge-Ring은 689/9,680장(7.1%), 두 SSL 표본에서는 각각 5.35%, 5.36%다. [scikit-image v0.25.2 소스](https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/filters/ridges.py), [data.py](/C:/Users/annan/OneDrive/바탕%20화면/umm/wafer/data.py:19), [크기·die 수 캐시](/C:/Users/annan/wafer_cache/wm811k_shapes_dies.npy)
- 제안: 원본 격자에서 계산한 뒤 96으로 리사이즈하는 경로를 96 초과 표본에도 우선 시험한다. 이 경로가 비용상 어렵다면 원본 격자 계산을 기준으로 축별 척도를 반영한 근사를 비교한다. `c` 산정 범위, σ별 응답의 결합 순서, 후처리 정규화, 양자화를 고정하고 uint8 복원값까지 검증한다.

**C3. [심각도: 중요] 같은 체커 마스크를 써도 가려진 die의 정보가 Frangi 채널에 남을 수 있다**

- 문제: Frangi를 전체 웨이퍼에서 먼저 계산하면, 보이는 위치의 응답에도 인접한 가림 영역의 불량 die가 반영된다. 같은 위치를 두 채널에서 0으로 만드는 것만으로는 이 경로를 차단할 수 없다.
- 근거: ver8은 회전 후 6픽셀 체커보드로 50%를 가린다. 캐시된 Frangi에는 마스크 적용 전의 이웃 정보가 들어가므로, ver8과 같은 마스크를 써도 SSL에서 볼 수 있는 정보가 달라진다. 실제로 모델이 이 경로를 얼마나 이용할지는 미확인이다. [wafixmatch.py](/C:/Users/annan/OneDrive/바탕%20화면/umm/wafixmatch.py:163), [ver6.py](/C:/Users/annan/OneDrive/바탕%20화면/umm/wafer/ver6.py:128), [augment.py](/C:/Users/annan/OneDrive/바탕%20화면/umm/wafer/augment.py:139)
- 제안: 같은 회전·마스크에서 가려진 die만 바꾸고, 보이는 Frangi 응답의 변화량을 측정한다. 변화가 크면 원래의 마스킹 과제를 보존하는 처리를 먼저 시험하거나, 결과의 해석을 ‘Frangi 입력과 달라진 SSL 관측 정보의 결합 효과’로 제한한다.

**C4. [심각도: 중요] test 공개 시점을 체크포인트 선택과 추가 seed 결정 이후로 미뤄야 한다**

- 문제: 현재 코드는 val 최고점이 갱신될 때마다 test를 실행한다. 기존 굵기 스크립트도 test의 none 웨이퍼에서 합성 배경을 뽑는다. 이 스크립트로 설정을 조정하거나 test를 본 뒤 seed 2~4를 추가하면 test가 결정에 관여한다.
- 근거: `_Tracker.step`은 val 개선 직후 test를 평가한다. `fewdie_test.py`의 조작 실험은 `te_none`에서 배경을 뽑는다. [engine.py](/C:/Users/annan/OneDrive/바탕%20화면/umm/wafer/engine.py:68), [fewdie_test.py](/C:/Users/annan/OneDrive/바탕%20화면/umm/runs/hough_eda/fewdie_test.py:98)
- 제안: ‘최종 체크포인트’를 **val macro F1 최고점의 `best.pt`**로 명시한다. 합성 배경은 train 또는 val에서 고르고, 실제 클래스별 응답 분포와 σ·정규화 선택에는 val을 사용한다. 추가 seed 여부도 사전에 정한 val 규칙으로 결정하고, 모든 결정을 고정한 뒤 test를 보고한다.

**C5. [심각도: 중요] WaFixMatch 목표 충족과 ver8 대비 개선을 각각 판정해야 한다**

- 문제: 목표선을 넘는 것과 ver8 대비 개선은 별개의 질문이다. 두 seed의 결과를 본 뒤 평균과 개별 seed 중 유리한 기준을 고르면 결론이 달라진다.
- 근거: 직접 읽은 test macro F1은 WaFixMatch `0.79497`, ver8 seed 0 `0.83137`, seed 1 `0.78347`이다. WaFixMatch 기준 소수 클래스 F1 하한은 Near-full `0.8367`, Donut `0.8019`, Random `0.7174`, Scratch `0.4346`, Loc `0.6309`다. ver8 seed 0은 다섯 하한을 모두 통과했고, seed 1은 macro 목표와 Donut 하한을 놓쳤다(Donut `0.7290`). ver7.3 seed 0도 macro `0.80294`에도 불구하고 Donut `0.7963`, Scratch `0.3493`으로 하한을 놓쳤다. [WaFixMatch metrics](/C:/Users/annan/OneDrive/바탕%20화면/umm/runs/resnet18/p0.05/seed0/wafixmatch/metrics.json:161), [ver8 seed 0 metrics](/C:/Users/annan/OneDrive/바탕%20화면/umm/runs/resnet18/p0.05/seed0/ver8_ssl50_ft30/metrics.json:291), [ver8 seed 1 metrics](/C:/Users/annan/OneDrive/바탕%20화면/umm/runs/resnet18/p0.05/seed1/ver8_ssl50_ft30/metrics.json:296), [ver7.3 metrics](/C:/Users/annan/OneDrive/바탕%20화면/umm/runs/resnet18/p0.05/seed0/ver7.3_ssl50_ft30/metrics.json:291)
- 제안: 0·1은 파일럿으로 두고 각 seed의 목표 통과 여부와 같은 seed의 `y9.2 − ver8` 차이를 모두 보고한다. 두 seed의 0.01~0.03 차이는 관측값으로 해석한다. 기준 결과를 재사용할 때는 같은 seed의 `ssl_subset_random.npy` 인덱스 해시, 라벨 인덱스, 실제 SSL 설정, `v3_nopl=True`, fine-tune 3×30 epoch를 대조한다. 두 seed의 기존 SSL 파일은 각각 100,000장이며 교집합은 15,632장으로 확인했다. 해당 파일이 과거 실행 이후 교체됐는지는 로그와 `metrics.json`으로 확인할 수 없다. [wafixmatch.py](/C:/Users/annan/OneDrive/바탕%20화면/umm/wafixmatch.py:137), [selfsup.py](/C:/Users/annan/OneDrive/바탕%20화면/umm/wafer/selfsup.py:196)

**C6. [심각도: 중요] 굵기 조작의 인용 조건을 바로잡고 비교를 짝지어야 한다**

- 문제: 제안서의 `0.90 → 0.11`은 같은 배경의 1→2 die 비교를 정확히 인용한 값이 아니다. 합성선 응답만으로 실제 저 die Scratch의 개선 여부를 판정하기도 어렵다.
- 근거: 큰 웨이퍼의 **원래 배경**에서 평균 Scratch 확률은 seed 0 `0.888 → 0.111`, seed 1 `0.847 → 0.027`이다. `0.910`은 seed 0의 **배경 불량을 지운 1 die** 조건이다. 같은 배경의 조작은 굵기 민감성을 뒷받침하지만, 기존 스크립트에는 작은 웨이퍼의 2 die 조건이 없고 1→2 die 조작은 선에 속한 불량 die 수도 바꾼다. [fewdie_test.txt](/C:/Users/annan/OneDrive/바탕%20화면/umm/runs/hough_eda/fewdie_test.txt:14), [fewdie_test.py](/C:/Users/annan/OneDrive/바탕%20화면/umm/runs/hough_eda/fewdie_test.py:127)
- 제안: 동일 배경·중심선에서 1/2/3 die를 렌더링하고 각 조건의 **조작된 원본**에서 Frangi를 재계산한다. die 수, 각도, 배경 불량률을 바꿔 선 중심부·배경 응답과 모델의 Scratch 확률 감소폭을 짝지어 비교한다. 실제 val에서는 저 die Scratch의 재현율과 none·Loc·Edge-Ring·Donut에서 Scratch로 가는 오탐을 같이 본다.

**C7. [심각도: 중요] y9.1 probe가 보여준 분리 가능성의 범위를 좁혀 해석해야 한다**

- 문제: ‘인코더에 Scratch 정보가 충분하고 라벨 부족이 병목’이라는 해석은 현재 진단이 직접 검증한 범위를 넘는다.
- 근거: seed 0 y9 50ep의 layer4 max probe AUC는 Scratch 대 none `0.971`, Scratch 대 Loc `0.938`이다. 저 die Scratch 대 Loc은 `0.922`, 모델 출력은 `0.837`이다. probe는 val 라벨로 교차검증한 이진 분류기다. 실제 5% train에는 Scratch가 48장이고, 이 라벨로 학습한 layer4 선형 헤드의 val Scratch F1은 `0.442`로 기존 헤드 `0.446`과 비슷했다. [layer_probe.py](/C:/Users/annan/OneDrive/바탕%20화면/umm/runs/y9.1_layer_probe/layer_probe.py:59), [layer_probe.txt](/C:/Users/annan/OneDrive/바탕%20화면/umm/runs/y9.1_layer_probe/layer_probe.txt:55), [head_probe.txt](/C:/Users/annan/OneDrive/바탕%20화면/umm/runs/y9.1_head_probe/head_probe.txt:19)
- 제안: 얼린 인코더 비교에는 동일한 6,919장으로 학습한 probe를 두고, val에서 Scratch PR-AUC와 저 die Scratch 대 Loc 분리를 비교한다. val 교차검증 AUC를 함께 쓰면 표현의 분리 가능성 진단으로 해석한다.

## 빠진 대조군·검증

- **Frangi 특이 효과를 주장하려면** 동일한 2채널 stem·초기화·캐시·회전·마스크를 쓰는 2채널 대조군이 필요하다. 예를 들어 두 번째 채널에 동일 원본 격자에서 만든 `불량 여부` 이진 맵을 넣는다. ver8 대 y9.2 A/B만 수행해도 전체 변경의 효과는 평가할 수 있다. ver7.3 기록은 회전 후 팽창을 쓴 seed 0 비교다. [ver6.py](/C:/Users/annan/OneDrive/바탕%20화면/umm/wafer/ver6.py:128), [ver7.3 metrics](/C:/Users/annan/OneDrive/바탕%20화면/umm/runs/resnet18/p0.05/seed0/ver7.3_ssl50_ft30/metrics.json:291)
- 학습 전 통과 기준을 수치로 정한다. 이미지 평균 응답은 선 길이와 불량 면적에도 좌우되므로, 선 중심부 응답 유지율과 배경·링·덩어리의 상위 응답을 함께 본다. PR-AUC와 재현율은 기존 `metrics.json`만으로 전부 복원할 수 없다. 재현율은 confusion에서 계산하고, PR-AUC는 고정한 두 모델의 `best.pt`에서 같은 방식으로 점수를 다시 추출한다. [engine.py](/C:/Users/annan/OneDrive/바탕%20화면/umm/wafer/engine.py:42)

## 더 싸거나 더 나은 대안

Frangi 검증을 통과하면 기존 ver8 SSL 인코더의 첫 채널 가중치를 복사하고, 새 채널 가중치는 0으로 초기화한 뒤 **fine-tune만 수행하는 두 조건**을 먼저 비교한다. 한쪽은 Frangi, 다른 쪽은 0 채널을 넣어 동일한 초기 모델과 학습 조건에서 시작한다. 이 비교는 기존 표현에 Frangi를 추가하는 효과를 싼 비용으로 확인한다. Frangi를 활용한 사전학습의 효과는 이후 SSL A/B에서 확인한다.

파일은 수정하지 않았다. Frangi 캐시 생성과 학습은 실행하지 않았으므로 실제 처리 시간과 응답 품질은 미검증이다.