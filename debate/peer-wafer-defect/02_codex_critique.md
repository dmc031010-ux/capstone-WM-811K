## 판정: 수정 후 진행 — 기존 체크포인트로 이식은 가능하지만, val 채택 기준과 짝지은 대조군을 먼저 고정해야 한다.

## 지적

**C1. [심각도: 중요] test를 본 설계 선택과 val 체크포인트 선택을 구분해야 한다**

- **문제** 동료의 0.8527은 같은 test에서 여러 구조를 비교한 뒤 보고된 값이다. 동일한 test에서 로컬 이식 결과가 일치해도 독립적인 성능 확인으로 해석하기 어렵다. 제안의 “val과 test를 둘 다 보고”에는 후속 실험을 결정할 기준도 빠져 있다.
- **근거** 동료 `_Tracker`는 val macro가 갱신될 때 체크포인트를 저장한다. 실험 문서는 별도로 **Test Macro F1** 표에서 layer3를 확정하고 ver9.8을 기각한다. 실제로 val을 함께 검토했는지와 test 선택 편향의 크기는 원자료 없이 확인할 수 없다. [동료 engine.py 73줄](<C:/Users/annan/OneDrive/바탕 화면/umm/debate/peer-wafer-defect/peer_repo/wafer/engine.py:73>) · [동료 EXPERIMENTS.md 110줄](<C:/Users/annan/OneDrive/바탕 화면/umm/debate/peer-wafer-defect/peer_repo/EXPERIMENTS.md:110>)
- **제안** 모델과 후속 seed 실행 여부는 seed별 **val macro 차이**와 소수 클래스 기준으로 결정한다. 후보와 분석 계획을 고정한 뒤 기존 test를 보고하고, 이 test가 설계 탐색에 쓰였다는 이력을 명시한다.

**C2. [심각도: 중요] 샘플링 해석이 뒤집혀 있어 구조 효과와 레시피 효과가 섞인다**

- **문제** 문서의 “1:1 균형 샘플링에서 `0.35`로 전환”은 맞지만, 우리 `ver7.9`의 `sample_pow=1.0`을 1:1 실험으로 해석하면 방향이 뒤집힌다. 또 전체 이식만으로는 선 스트림과 레시피 중 어느 쪽이 Scratch를 개선했는지 알 수 없다.
- **근거** 양쪽 구현에서 샘플당 가중치는 `count ** (power - 1)`이다. 클래스별 총 추출 확률은 `count ** power`에 비례하므로 **0은 클래스 균등, 1은 원래 클래스 비율, 0.35는 중간**이다. 기본 ver7의 값은 0이다. `ver9.9B2`도 gradient clipping 1.0과 belief 보조 손실 0.3을 켠다. 동료 문서가 선 스트림 효과의 근거로 든 `ver9.9 → ver9.91`은 양쪽 모두 선 스트림을 쓰고 Scratch 손실 가중치가 1.35에서 1.0으로 바뀌는 비교다. [우리 engine.py 25줄](<C:/Users/annan/OneDrive/바탕 화면/umm/wafer/engine.py:25>) · [동료 engine.py 25줄](<C:/Users/annan/OneDrive/바탕 화면/umm/debate/peer-wafer-defect/peer_repo/wafer/engine.py:25>) · [동료 wafixmatch.py 345줄](<C:/Users/annan/OneDrive/바탕 화면/umm/debate/peer-wafer-defect/peer_repo/wafixmatch.py:345>) · [동료 wafixmatch.py 362줄](<C:/Users/annan/OneDrive/바탕 화면/umm/debate/peer-wafer-defect/peer_repo/wafixmatch.py:362>) · [동료 EXPERIMENTS.md 103줄](<C:/Users/annan/OneDrive/바탕 화면/umm/debate/peer-wafer-defect/peer_repo/EXPERIMENTS.md:103>)
- **제안** 전체 이식과 함께 기본 ResNet에 공통 적용 가능한 레시피만 붙인 대조군을 둔다. Scratch 개선을 선 스트림에 귀속하려면 전체 모델에서 선 스트림만 빼고 비교한다. belief 보조 손실은 belief 모듈과 함께 해석한다.

**C3. [심각도: 중요] 재현 성공 조건에 소수 클래스 하한이 빠졌다**

- **문제** macro와 적은 die 구간의 Scratch 재현율만으로는 프로젝트 목표를 판정할 수 없다. G2의 `+0.045 > ±0.024`도 개선의 재현성을 판단하는 기준으로 부적절하다. `±0.024`는 기준 모델 두 seed의 산포이며, 같은 seed에서 계산한 개선량의 산포는 별도로 봐야 한다.
- **근거** 로컬 ver9의 val/test macro는 seed 0에서 **0.8184/0.8314**, seed 1에서 **0.7945/0.7835**다. seed 1의 test Donut은 **0.7290**으로 WaFixMatch의 **0.8319**보다 **0.1029** 낮다. WaFixMatch test 기준으로 0.03 하락을 허용한 경계값은 Near-full **0.8367**, Donut **0.8019**, Random **0.7174**, Scratch **0.4346**, Loc **0.6309**다. [ver9 seed 0 지표](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/resnet18/p0.05/seed0/ver9_ssl50_ft30/metrics.json:174>) · [ver9 seed 1 지표](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/resnet18/p0.05/seed1/ver9_ssl50_ft30/metrics.json:296>) · [WaFixMatch 기준 지표](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/resnet18/p0.05/seed0/wafixmatch/metrics.json:161>)
- **제안** seed 0·1 각각에서 동일 체크포인트로 시작한 기준 모델 대비 val 차이를 보고한다. 채택에는 val만 사용하고, 최종 test 보고에는 macro 목표 **0.795 초과**와 다섯 클래스의 하한 충족 여부를 seed별로 명시한다. 추가 seed를 실행한다면 후보와 기준 모델을 함께 실행한다.

**C4. [심각도: 중요] 두 seed의 근접한 값은 원자료로 확인해야 한다**

- **문제** `164/242`는 보고된 Scratch F1과 일치하는 계산이지만, 반올림된 F1과 평균 TP만으로 개별 실행의 혼동행렬을 확정할 수 없다. 같은 TP·FP·FN이 나와도 어떤 웨이퍼를 맞혔는지는 달라질 수 있다. 같은 seed 재실행은 실행 반복성을 확인하는 데 쓰인다.
- **근거** 동료 코드는 출력과 캐시 경로에 `seed{args.seed}`를 포함하고 `ver9.91`에서 `masklat7`을 로드한다. 로컬 캐시를 SHA-256으로 확인한 결과, **각 seed 안에서** `masklat7/8/9_e50.pt`는 바이트 단위로 같고 **seed 0과 1 사이에서는** 체크포인트와 `ssl_subset_random.npy`의 해시가 각각 달랐다. 동료 사본에는 `metrics.json`과 실행 로그가 없어 동료가 실제로 로드한 파일은 확인할 수 없다. 캐시 혼용·중복 집계·문서 오기는 모두 추측이다. [동료 wafixmatch.py 99줄](<C:/Users/annan/OneDrive/바탕 화면/umm/debate/peer-wafer-defect/peer_repo/wafixmatch.py:99>) · [동료 wafixmatch.py 137줄](<C:/Users/annan/OneDrive/바탕 화면/umm/debate/peer-wafer-defect/peer_repo/wafixmatch.py:137>) · [동료 EXPERIMENTS.md 118줄](<C:/Users/annan/OneDrive/바탕 화면/umm/debate/peer-wafer-defect/peer_repo/EXPERIMENTS.md:118>)
- **제안** 동료에게 seed별 `metrics.json`, 실행 명령과 커밋, 로드한 인코더 및 SSL 부분집합 해시, 웨이퍼별 예측을 요청한다. 혼동행렬은 `metrics.json`에 이미 포함된다. 로컬 이식도 체크포인트 경로를 명시하고 파일이 없으면 즉시 실패하도록 설계한다. 반복 실행은 원자료 확인 후에도 비결정성이 쟁점일 때 진행한다. [동료 engine.py 52줄](<C:/Users/annan/OneDrive/바탕 화면/umm/debate/peer-wafer-defect/peer_repo/wafer/engine.py:52>)

**C5. [심각도: 중요] 평균 풀링과 적은 die 문제는 모듈 효과를 검증할 가설이다**

- **문제** #13의 로짓 분해와 특징 공간 이웃 분석은 설계 동기를 뒷받침한다. 풀링이 오분류의 원인인지, 선 스트림이 적은 die에서 특히 도움이 되는지는 별도의 비교가 필요하다.
- **근거** `decomp13.py`는 기존 layer4의 셀 기여를 합쳐 로짓을 복원한다. `numbers.txt`의 이웃 분석은 ver8 seed 0에서 Scratch를 none으로 놓친 41장의 이웃 중 none이 88%, Loc으로 놓친 19장의 이웃 중 Loc이 83%였다고 보고한다. 실제 선 스트림은 고정 필터 응답에 **원본 defect와 anomaly 채널**도 넣어 학습하므로 σ만으로 처리 가능한 선 폭을 단정하기 어렵다. 합성 실험에서는 같은 배경의 선을 1 die에서 2 die로 넓혔을 때 Scratch 판정률이 seed별 **0.79·0.84** 떨어졌다. 실제 Scratch 구간에서는 `600–1199 die`가 **42장, 재현율 0.23**으로 가장 취약했고, 기존 Wilson 구간은 같은 웨이퍼의 두 seed 예측을 `2n`개로 센다. [decomp13.py 19줄](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/hough_eda/decomp13.py:19>) · [numbers.txt 17줄](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/ver8_eda/numbers.txt:17>) · [동료 models.py 284줄](<C:/Users/annan/OneDrive/바탕 화면/umm/debate/peer-wafer-defect/peer_repo/wafer/models.py:284>) · [fewdie_test.txt 8줄](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/hough_eda/fewdie_test.txt:8>) · [fewdie_test.py 61줄](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/hough_eda/fewdie_test.py:61>)
- **제안** 기존 die 구간별로 표본 수, Scratch 재현율, `Scratch→none/Loc` 및 `none/Loc→Scratch`를 seed별로 보고한다. 구간 차이에 신뢰구간을 붙인다면 동일 웨이퍼의 모델·seed 예측을 함께 보존하는 짝지은 재표집을 쓴다. 선 폭이 원인인지 따로 확인하려면 동일 배경과 동일 선분에서 입력 픽셀 폭만 바꿔 전체 모델과 선 스트림 제거 모델을 비교한다.

**C6. [심각도: 중요] SimCLR 우위와 마스킹의 Scratch 우위는 seed 0 관찰이다**

- **문제** P5의 “마스킹 사전학습은 오히려 손해”는 결론이 앞선다. SimCLR 비교의 체크포인트 준비 비용도 별도로 잡아야 한다.
- **근거** 동료 문서의 동일 seed 비교는 `ver3.991` 대 `ver9.91`에서 macro **0.8674 대 0.8526**, Scratch **0.6585 대 0.6777**이다. 로컬 캐시에는 동료 기본 경로가 요구하는 seed 0 `simclr_random_cut_e50.pt`가 있지만 seed 1의 대응 파일은 확인되지 않았다. 동료 Phase 5 표에서 L1·L2는 완료, L3는 진행 중, L4는 예정으로 표시된다. [동료 EXPERIMENTS.md 124줄](<C:/Users/annan/OneDrive/바탕 화면/umm/debate/peer-wafer-defect/peer_repo/EXPERIMENTS.md:124>) · [동료 EXPERIMENTS.md 135줄](<C:/Users/annan/OneDrive/바탕 화면/umm/debate/peer-wafer-defect/peer_repo/EXPERIMENTS.md:135>) · [동료 wafixmatch.py 154줄](<C:/Users/annan/OneDrive/바탕 화면/umm/debate/peer-wafer-defect/peer_repo/wafixmatch.py:154>)
- **제안** 이식 검증 후 seed 0 캐시로 탐색 비교를 먼저 한다. 인코더 우열을 판정할 때는 같은 seed의 SSL 부분집합과 fine-tune 설정을 맞추고 양쪽 인코더를 짝지어 비교한다. 마스킹의 방법론적 기여가 과제 요구인 경우에만 별도의 Phase 5 채택 기준을 둔다.

## 빠진 대조군·검증

- **구조와 레시피** 동일 seed·라벨 인덱스·인코더에서 `기존 ver9 → 공통 레시피만 → 전체 이식`을 비교한다. 선 스트림 효과를 주장할 때는 전체 이식에서 선 스트림만 뺀 조건을 추가한다.
- **이식 일치성** 학습 전 실제 로드 경로, 누락·예상 밖 파라미터 키, 입력값 범위, 보조 손실 활성화, optimizer 파라미터 그룹을 확인한다. `ft30`은 동료 코드에서 **30 epoch × 3라운드**이며, 재현할 CLI는 pseudo-label을 끈다. [동료 selfsup.py 176줄](<C:/Users/annan/OneDrive/바탕 화면/umm/debate/peer-wafer-defect/peer_repo/wafer/selfsup.py:176>) · [동료 wafixmatch.py 321줄](<C:/Users/annan/OneDrive/바탕 화면/umm/debate/peer-wafer-defect/peer_repo/wafixmatch.py:321>)

## 더 싸거나 더 나은 대안

원자료 요청과 로컬 캐시 검증을 먼저 하고, seed 0에서 **공통 레시피만 적용한 대조군과 전체 이식**을 돌린다. 기존 ver9 지표는 입력·라벨·체크포인트·학습 설정이 일치하면 재사용할 수 있다. seed 0의 val 결과를 사전에 정한 기준으로 검토한 뒤 seed 1에서 같은 비교를 반복하는 편이, 같은 seed를 한 번 더 돌리는 것보다 채택 판단에 직접 도움이 된다.

## 확인 질문

- 이 과제에서 **마스킹 사전학습의 독창성 자체**가 필수인가? 이 요구에 따라 Phase 5를 별도 목표로 둘지, 인코더를 성능 비교의 한 조건으로 둘지가 달라진다.