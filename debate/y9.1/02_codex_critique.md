## 판정: 수정 후 진행

die 단위 노이즈는 검증할 가치가 있다. 불량을 추가하는 순서를 바로잡고, val에서 증강 강도와 소수 클래스 성능을 검증한 뒤 50 epoch 실험으로 넘어가자.

## 지적

**C1. [심각도: 치명] 회전 후에는 `(h, w)`만으로 원래 die 경계를 맞출 수 없다.**

- **문제** — 계획대로 회전한 뒤 96×96 영상에 축 정렬 블록을 적용하면 한 die의 일부만 바꾸거나 여러 die를 함께 바꿀 수 있다. 또한 픽셀 단위 대체 경로에 들어가는 비율은 계획의 0.6%보다 크다.
- **근거** — 캐시는 픽셀 중심으로 원래 die 좌표를 계산하고, `A.rotate`는 웨이퍼별 임의 각도로 nearest-neighbor 회전한다. `fewdie_test.py`에는 회전 전 캐시를 원래 격자로 복원한 뒤 같은 인덱스로 다시 그려 일치 여부를 확인하는 코드가 있다. 실제 `wm811k_shapes_dies.npy`와 두 `ssl_subset_random.npy`를 대조한 결과, 높이 또는 너비가 96을 넘는 샘플은 seed 0에서 **5,346/100,000**, seed 1에서 **5,364/100,000**이었다. [data.py](<C:/Users/annan/OneDrive/바탕 화면/umm/wafer/data.py:19>) · [augment.py](<C:/Users/annan/OneDrive/바탕 화면/umm/wafer/augment.py:24>) · [fewdie_test.py](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/hough_eda/fewdie_test.py:71>)
- **제안** — 배치의 전역 인덱스 `bi`로 `(h, w)`를 조회하고, 회전 전 정상 die마다 한 번 추첨해서 그 die에 해당하는 캐시 픽셀을 전부 `1→2`로 바꾼 뒤 뷰마다 독립 회전하라. 96 초과 샘플도 캐시에 보이는 픽셀을 동일 원본 die 좌표로 묶을 수 있다. 대응을 검사할 때 `p=0`의 입력 일치, `p=1`의 정상 die 전환, 기존 배경과 불량의 보존을 확인하라. [ver6.py](<C:/Users/annan/OneDrive/바탕 화면/umm/wafer/ver6.py:155>) · [data.py](<C:/Users/annan/OneDrive/바탕 화면/umm/wafer/data.py:120>)

**C2. [심각도: 중요] 진단 스크립트 재사용 경로에 test가 들어 있다.**

- **문제** — 보조 진단으로 `p`나 계속 진행할지를 판단하면 test가 모델 선택에 관여한다.
- **근거** — `verify_mask.py`의 [C]는 6,919장으로 로지스틱 회귀를 학습하고 `ds.test`에서 채점한다. [B]의 Scratch 후보 추출에는 split 제한이 없다. `fewdie_test.py`의 합성선은 test의 `none`에 그린다. 학습 코드의 체크포인트 갱신 조건은 val macro지만, val 최고점이 갱신될 때마다 test 점수도 출력한다. [verify_mask.py](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/hough_eda/verify_mask.py:35>) · [verify_mask.py의 probe](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/hough_eda/verify_mask.py:102>) · [fewdie_test.py](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/hough_eda/fewdie_test.py:101>) · [engine.py](<C:/Users/annan/OneDrive/바탕 화면/umm/wafer/engine.py:68>)
- **제안** — probe 학습은 고정 6,919장으로 하고 채점, 선 민감도, 합성선 진단은 모두 val로 옮겨라. 설정과 체크포인트를 확정한 후 test를 보고하라.

**C3. [심각도: 중요] seed 0의 미세한 상승과 macro 단독 기준으로는 프로젝트 목표를 판단하기 어렵다.**

- **근거** — y9의 val macro는 seed 0 **0.818379**, seed 1 **0.794510**이다. test는 각각 **0.831365**, **0.783472**이며, seed 1의 Donut은 **0.729**로 하한 **0.8019**에 미달한다. 계획의 하한은 WaFixMatch test F1에서 0.03을 뺀 값과 반올림 수준에서 일치한다. [y9 seed 0](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/resnet18/p0.05/seed0/y9_ssl50_ft30/metrics.json:175>) · [y9 seed 1 val](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/resnet18/p0.05/seed1/y9_ssl50_ft30/metrics.json:180>) · [y9 seed 1 test](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/resnet18/p0.05/seed1/y9_ssl50_ft30/metrics.json:296>) · [WaFixMatch test](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/resnet18/p0.05/seed0/wafixmatch/metrics.json:161>)
- **제안** — seed 0을 예산 선별에 쓴다면 진행 기준을 미리 정하라. 개선을 주장할 후보는 seed 0·1 각각에서 같은 seed의 y9 대비 val macro 차이, 소수 5개 클래스 F1 차이, Scratch 재현율과 `Scratch→none/Loc` 건수를 보고하라. 모델 선택에 쓸 클래스별 기준은 val에서 정하고, test에서는 macro 목표와 소수 클래스 하한을 모두 확인하라. 두 seed의 결과 방향이 다르면 seed 민감성까지 보고하라.

**C4. [심각도: 중요] 현재 근거는 배경 불량이 영향을 줄 가능성을 뒷받침한다.**

- **근거** — `0.078 vs 0.107`은 seed 0에서 **정분류한 Scratch와 모든 오분류 Scratch**를 비교한 96×96 전체 불량 픽셀 비율이다. 선 자체도 포함된다. 합성 조작은 기존 불량을 전부 정상화한 **뒤** 선을 그린다. 적은 die의 1-die 선에서 배경을 지우면 seed 0의 `none`은 `0.32→0.02`, `Loc`은 `0.11→0.61`, Scratch는 `0.55→0.37`이다. seed 1도 Scratch가 `0.57→0.18`로 줄었다. 별도로 마스킹 비교에서 `0.61`은 비율이 같다는 뜻이다. 선 삭제 시 특징 변화량은 SimCLR `0.0655`, ver8 `0.0042`였고, 두 학습 방식은 목적함수도 다르다. [ver8_eda.py](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/ver8_eda/ver8_eda.py:69>) · [fewdie_test.py](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/hough_eda/fewdie_test.py:130>) · [fewdie_test.txt](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/hough_eda/fewdie_test.txt:17>) · [verify_mask.txt](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/hough_eda/verify_mask.txt:11>)
- **제안** — 가설을 “선을 고정하고 흩어진 불량을 추가했을 때 y9.1의 Scratch 판정이 y9보다 덜 흔들리는가”로 좁혀라. 같은 val 웨이퍼와 같은 선에서 추가 강도만 바꾸고, Scratch 점수와 `none/Loc` 전이를 함께 비교하라.

**C5. [심각도: 중요] 네 강도를 한 번에 넣으면 강도별 학습 효과를 알기 어렵다.**

- **근거** — y9의 손실은 네 뷰의 predictor 출력을 평균한 뒤 원본과 cosine을 계산한다. 한 강도의 뷰가 원본에서 멀어져도 평균 손실은 낮을 수 있다. 또 정상 die의 16%를 불량으로 바꾸면 원래 불량률이 0.078인 웨이퍼의 기대 불량률은 `0.078 + 0.16 × (1 − 0.078) ≈ 0.226`이다. Random 판별에 필요한 밀도 정보가 약해지는지 확인할 이유가 있다. [ver6.py](<C:/Users/annan/OneDrive/바탕 화면/umm/wafer/ver6.py:72>) · [계획의 강도 설정](<C:/Users/annan/OneDrive/바탕 화면/umm/debate/y9.1/01_proposal.md:22>)
- **제안** — 혼합 강도 실험의 결론은 그 혼합 설정에 한정하라. 손실은 y9와 동일하게 두고 뷰별 cosine과 강도별 고정 특징 probe를 추가하라. 학습 강도 자체를 비교하려면 약한 고정 `p`와 혼합 `p`를 별도 실행으로 비교하라.

**C6. [심각도: 중요] `z_std≈0.02`와 낮은 손실만으로 붕괴를 판정하기 어렵다.**

- **근거** — 코드는 EMA 타깃의 정규화 특징에 대해 차원별 배치 표준편차를 평균한다. y9 seed 0도 첫 epoch에 `loss=0.0302`, `z_std=0.0228`이었고, 마지막에 `z_std=0.0253`이었다. ver6 로그의 첫 epoch에는 `0.0159 / 0.0220`이 기록돼 있다. 제안서의 **12스텝 시점 수치**는 확인한 로그에서 검증하지 못했다. ver6은 회전과 predictor도 y9와 다르다. [ver6.py](<C:/Users/annan/OneDrive/바탕 화면/umm/wafer/ver6.py:174>) · [y9 로그](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/log_y9_p0.05_s0.txt:5>) · [ver6 로그](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/log_ver6_p0.05_s0.txt:5>) · [ver6 설정 비교](<C:/Users/annan/OneDrive/바탕 화면/umm/results/ver6_experiments.md:49>)
- **제안** — 같은 epoch의 y9와 `z_std` 추이를 비교하고, 고정 val 표본의 온라인 인코더 특징 유효 랭크와 probe 성능도 확인하라. 여러 지표가 함께 악화될 때 조기 중단을 판단하라.

## 빠진 대조군·검증

- **회전만 적용한 `p=0` 대조군.** 기존 6px 마스킹 y9와 함께 비교하면 마스킹을 제거한 효과와 랜덤 불량을 추가한 효과를 구분할 수 있다.
- **실행 조건 일치 확인.** seed별 동일한 10만 장 인덱스와 고정 라벨 6,919장, `v3_nopl=true`, SSL 50 epoch, fine-tune `3×30` epoch를 맞춰라. y9.1에는 별도 인코더 파일명과 결과 태그를 쓰고 기존 인코더나 `done` 때문에 실행이 건너뛰어지지 않는지 확인하라. [wafixmatch.py의 캐시·실행 분기](<C:/Users/annan/OneDrive/바탕 화면/umm/wafixmatch.py:132>) · [selfsup.py의 fine-tune 설정](<C:/Users/annan/OneDrive/바탕 화면/umm/wafer/selfsup.py:196>)

## 더 싸거나 더 나은 대안

먼저 **기존 y9 체크포인트**로 val의 동일 웨이퍼에 같은 합성선을 그려 놓고, 불량만 단계적으로 추가하라. die 수와 선 두께를 고정하면 기존 조작 실험에서 함께 움직였던 요인을 분리할 수 있다. 그다음 짧은 동일 예산 파일럿으로 마스킹, 회전만, 노이즈를 비교하라. 기존 y9 로그의 SSL epoch 시간만 합해도 seed 0은 **4.12시간**, seed 1은 **6.50시간**이었다. fine-tune과 새 증강 비용을 더하기 전이므로 계획의 seed당 약 4시간은 파일럿 처리량으로 다시 산정하는 편이 좋다. [seed 0 SSL 로그](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/log_y9_p0.05_s0.txt:6>) · [seed 1 SSL 로그](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/log_y9_p0.05_s1.txt:43>)

코드, 로그, 지표 파일과 캐시를 읽어 확인했다. 파일 수정이나 학습 실행 없이 검토했다.