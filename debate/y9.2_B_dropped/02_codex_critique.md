## 판정: 수정 후 진행

`split_seed=0`으로 캐시를 재계산한 결과, 3% 필터를 통과하는 사본은 두 학습 seed 모두 **3장**이고, 그중 **2장**만 `die<1200`입니다. 이 조건에서 결과가 좋아지면 후속 검증의 근거가 되지만, 효과가 없을 때 Scratch 라벨 다양성 가설을 기각하기에는 개입이 작습니다.

## 지적

**C1. [심각도: 치명] 실제 추가 표본과 목표 구간의 겹침이 계획보다 작음**

- **문제:** 두 실행은 같은 라벨 6,919장을 쓰므로 증강 대상도 같아야 합니다. 계획의 18장·17장과 Scratch 추출의 약 25%는 실제 제한판의 수치와 다릅니다.
- **근거:** `split_seed=0`의 라벨과 원본 격자를 캐시에서 복원해 계산했습니다. Scratch 48장 중 `die<1200`은 29장, 격자 크기와 `die≥1200` 조건을 통과한 것은 18장입니다. 8-이웃 최대 불량 성분을 빼고 배경 불량률 `≤3%`를 적용하면 **3장**이 남습니다. 사본 die 수는 **371, 371, 1,243**입니다. 앞의 두 장은 기존 저다이 train Scratch의 최솟값 516보다 작고, 마지막 장은 평가 기준인 1,200을 넘습니다. 균등 추출 시 사본 비중은 `3/51=5.9%`입니다. `fig_B.png`의 네 원본 중 필터를 통과하는 것은 한 장입니다. [라벨 선택 코드](<C:/Users/annan/OneDrive/바탕 화면/umm/wafixmatch.py:98>) · [원본 격자 복원과 미리보기 코드](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/y9.2_aug_preview/aug_preview.py:31>) · [미리보기 그림](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/y9.2_aug_preview/fig_B.png>)
- **제안:** 두 seed에 동일한 원본 ID 목록을 고정하고, 세 사본 전부를 원본과 나란히 검수하세요. 최대 연결 성분의 크기만으로 선형성을 보장할 수 없으므로 선 보존, 선 폭, 가장자리 불량 병합도 확인하세요. 3%를 유지한 채 실행한다면 결론의 범위를 **이 세 사본의 효과**로 제한하세요.

**C2. [심각도: 중요] 학습 중 test 평가는 이번부터 끄는 것이 맞음**

- **문제:** 체크포인트 선택 코드는 val macro만 사용합니다. 다만 val 최고점이 갱신될 때마다 test 점수가 로그와 `metrics.json`에 노출되어 후속 설정이나 후보 선택에 영향을 줄 수 있습니다.
- **근거:** `_Tracker.step()`은 val이 개선되면 `best.pt`를 저장하고 즉시 test를 계산합니다. y9 seed 0 로그에도 여러 중간 test 점수가 남아 있습니다. [트래커 코드](<C:/Users/annan/OneDrive/바탕 화면/umm/wafer/engine.py:68>) · [y9 학습 로그](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/log_y9_p0.05_s0.txt:134>)
- **제안:** 이번 대조군과 실험군 모두 학습 중 test 계산과 기록을 끄세요. val 기준으로 설정과 체크포인트를 확정한 뒤, 고정된 seed의 최종 체크포인트만 test에서 평가하세요.

**C3. [심각도: 중요] 클래스 비중을 맞춰도 원본별 가중치는 바뀜**

- **문제:** `sample_pow=0`이면 클래스별 기대 비중은 `1/9`입니다. Scratch 안에서는 선택된 원본 계열이 원본과 사본으로 각각 뽑혀, 사본을 받지 않은 원본보다 두 배 자주 등장합니다. 따라서 관측된 변화에는 2×2 변형 효과와 특정 원본의 노출 증가가 함께 들어갑니다.
- **근거:** `BalancedSampler`는 클래스 내 각 항목에 동일 확률을 주고 복원추출합니다. 현재 fine-tune은 `lab_idx` 길이로 epoch당 step 수도 정합니다. [샘플러](<C:/Users/annan/OneDrive/바탕 화면/umm/wafer/engine.py:31>) · [fine-tune step 계산](<C:/Users/annan/OneDrive/바탕 화면/umm/wafer/selfsup.py:196>)
- **제안:** 옵션을 끈 y9와 함께, 같은 세 원본을 **변형 없이 한 번씩 복제한 대조군**을 두세요. 복제군과 2×2군은 같은 표본 위치를 뽑도록 하고, 세 조건 모두 원래 6,919장 기준으로 **28 step × 90 epoch**를 고정하세요.

**C4. [심각도: 중요] y9.1 진단으로 라벨 다양성을 원인으로 특정하기 어려움**

- **문제:** layer probe의 이진 분리 성능과 실제 9클래스 분류 결과는 학습 데이터와 점수 산출 방식이 다릅니다. 그 차이만으로 인코더 표현과 판정층의 영향을 배제하기 어렵습니다.
- **근거:** layer probe는 val 안에서 5-fold로 이진 로지스틱 회귀를 적합합니다. y9의 `Scratch vs none` 평균 풀링 AUC는 layer3 **0.978**, layer4 **0.968**, 모델 출력 **0.957**로 감소 폭이 비슷합니다. `Scratch vs Loc`에서는 layer4가 저다이 **0.918**, 고다이 **0.948**로 차이가 남습니다. 또 y9 헤드 실험에서 Scratch F1은 기존 fc **0.446**, L3 **0.479**, 프로토타입 **0.321**로, 모든 헤드가 ±0.04 안에 든다는 요약은 맞지 않습니다. [probe 적합 코드](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/y9.1_layer_probe/layer_probe.py:59>) · [none 비교](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/y9.1_layer_probe/layer_probe.txt:21>) · [Loc 비교](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/y9.1_layer_probe/layer_probe.txt:55>) · [헤드 결과](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/y9.1_head_probe/head_probe.txt:19>)
- **제안:** 가설을 **“이 2×2 변형을 추가하면 실제 저다이 Scratch의 분리가 좋아지는가”**로 좁히세요. `none→Scratch` 오탐 수를 맞췄을 때의 저다이 재현율과 `Loc→Scratch` 오탐도 함께 비교하면 판정 기준 이동을 구별하는 데 도움이 됩니다.

**C5. [심각도: 중요] 음수 ID와 체크포인트 재사용은 현재 코드에서 명시적으로 처리해야 함**

- **문제:** 음수 ID가 기존 경로에 들어가면 `ds.labels[li]`와 `ds.fetch(li)`가 배열 뒤쪽의 실제 웨이퍼를 읽습니다. 또한 `masklat9` 접미사만 맞춰도 공유 캐시 경로가 다르면 사전학습으로 진입할 수 있습니다. 옵션 켬·끔이 같은 출력 경로를 쓰면 먼저 생성된 `done` 때문에 다음 실행이 건너뛰어질 수 있습니다.
- **근거:** 두 seed의 `wafer_cache/ver3/runs/resnet18/p0.05/seed{0,1}/masklat9_e50.pt`와 `ssl_subset_random.npy`가 실제로 존재합니다. 현재 CLI는 공유 캐시 분기와 접미사로 인코더 경로를 만들고, 파일이 없으면 사전학습을 호출합니다. [배치 조회](<C:/Users/annan/OneDrive/바탕 화면/umm/wafer/selfsup.py:248>) · [데이터 조회](<C:/Users/annan/OneDrive/바탕 화면/umm/wafer/data.py:120>) · [출력 경로와 캐시 분기](<C:/Users/annan/OneDrive/바탕 화면/umm/wafixmatch.py:132>) · [인코더 경로와 사전학습 분기](<C:/Users/annan/OneDrive/바탕 화면/umm/wafixmatch.py:151>)
- **제안:** 기존 `lab_idx`는 유지하고, 감독 배치의 샘플러 위치에서 원본과 사본을 나눠 조회한 뒤 한 배치로 합쳐 기존 `A.weak`를 한 번 적용하세요. 실행 초기에 실제 인코더 경로와 해시를 기록하고, 필요한 파일이 없으면 즉시 종료하세요. 옵션별 출력 경로를 구분하고 옵션을 껐을 때 같은 초기 배치 ID와 손실이 재현되는지 확인하세요.

**C6. [심각도: 중요] 진행 기준은 작은 변동과 프로젝트 목표를 구분해야 함**

- **문제:** 두 seed에서 val macro가 조금이라도 오르고 저다이 Scratch를 한 장씩 더 맞히면 현재 기준을 통과할 수 있습니다. `none→Scratch` 1.5배 상한은 seed별로 **43→64건**, **77→115건**까지 허용합니다. Scratch 정밀도나 전체 Scratch F1이 나빠져도 현재 기준을 통과할 여지가 있습니다.
- **근거:** 저다이 val Scratch는 두 seed에서 같은 **61장**이므로 한 장이 재현율 **1.64%p**입니다. 기존 y9의 val macro는 **0.81838 / 0.79451**, Scratch F1은 **0.4464 / 0.4291**입니다. 이 seed 간 차이에는 사전학습 인코더 차이도 포함됩니다. 최종 비교 기준인 WaFixMatch 재현 결과의 test macro는 **0.79497**이며, 소수 클래스별 기준도 해당 결과에 있습니다. [저다이 표본 수](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/y9.1_layer_probe/layer_probe.txt:1>) · [y9 seed 0 지표](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/resnet18/p0.05/seed0/y9_ssl50_ft30/metrics.json:175>) · [y9 seed 1 지표](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/resnet18/p0.05/seed1/y9_ssl50_ft30/metrics.json:180>) · [WaFixMatch 재현 지표](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/resnet18/p0.05/seed0/wafixmatch/metrics.json:161>)
- **제안:** 실행 전에 의미 있는 최소 변화량을 정하고, 저다이 Scratch에서 **새로 맞힌 수와 새로 놓친 수**, Scratch F1·정밀도, 전체 Scratch 오탐 수를 함께 보고하세요. 동일 val 웨이퍼의 짝지은 비교로 표본 불확실성을 보고, 결과가 경계선이면 동일 인코더에서 fine-tune seed를 추가하세요. y9 대비 val 기준은 후속 진행 판단에 쓰고, 프로젝트 목표 달성은 최종 test에서 WaFixMatch의 macro와 소수 5개 F1을 기준으로 판정하세요.

## 빠진 대조군·검증

- **변형 없는 복제 대조군**이 우선입니다. 옵션을 끈 y9, 동일 원본 복제, 2×2 변형을 같은 인코더와 업데이트 수로 비교해야 원본 노출 증가와 변형 효과를 구별할 수 있습니다.
- 보조 진단을 반복할 때 **실제 원본 저다이 val**을 주 평가로 두세요. 기존 noise 진단은 원본 격자 `≤96`인 Scratch **114장**만 사용하며 합성 선 진단도 포함합니다. 같은 진단을 재사용한다면 두 모델에 동일한 웨이퍼와 노이즈 실현값을 적용하고, 노이즈 곡선은 보조 결과로 보고하세요. [진단 대상 선택 코드](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/y9.1_noise_diag/noise_diag.py:76>) · [크기별 노이즈 결과](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/y9.1_noise_diag/noise_diag.txt:74>)

## 더 싸거나 더 나은 대안

세 사본의 육안 검수와 함께 기존 y9 `best.pt`로 원본·사본의 `Scratch−max(none, Loc)` 로짓 차이를 비교하세요. 사본의 선이 뭉개지거나 점수가 크게 꺾이는 경우를 학습 전에 확인할 수 있습니다. 점수는 진단용으로 쓰고, 포함 여부는 사전에 정한 형태 기준으로 결정하세요.

파일은 수정하지 않았습니다. 표본 수와 die 수는 캐시를 읽어 재계산했고, fine-tune은 실행하지 않았습니다.