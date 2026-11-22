## 판정: 수정 후 진행 — `yeeun`을 기준으로 통합하되, 캐시 선택과 학습 동작의 동등성까지 검증해야 기존 결과를 안전하게 이어 쓸 수 있다.

로컬 `yeeun`의 HEAD가 `b207fc9`임을 확인했다. 동료 사본에는 `.git`과 실행별 `metrics.json`, 체크포인트가 없어 `5446ff7` 출처와 동료 문서의 성능 수치는 독립적으로 확인하지 못했다. 검토 범위는 코드·로그·결과 파일 읽기와 파일 해시 비교까지다. 파일은 수정하지 않았다.

## 지적

**C1. [심각도: 중요] 캐시 파일명 매핑에 인코더 종류와 캐시가 없을 때의 동작이 빠져 있다.**

- **문제** `k9.91`은 기존 ver7 인코더를 요구하고, `k3.991`은 기존 SimCLR 인코더를 요구한다. 동료 CLI는 둘 다 캐시가 없으면 중단하지만 `ver3.991`에도 ver7 체크포인트가 필요하다는 오류를 출력한다. 이 분기를 이름 치환 중 놓치면 재학습 여부와 비교할 인코더가 달라진다.
- **근거** 동료 CLI는 `ver9.91`을 `masklat7_e50.pt`, `ver3.991`을 기본 옵션에서 `simclr_random_cut_e50.pt`로 해석한다. 이 PC에는 seed 0의 두 파일이 모두 있다. `masklat7`, `masklat8`, `masklat9`의 SHA-256은 seed별로 서로 같았지만, 동료가 사용한 파일의 해시는 확인할 수 없었다. [동료 CLI 캐시 선택](<C:/Users/annan/OneDrive/바탕 화면/umm/debate/peer-wafer-defect/peer_repo/wafixmatch.py:137>) · [캐시 누락 분기](<C:/Users/annan/OneDrive/바탕 화면/umm/debate/peer-wafer-defect/peer_repo/wafixmatch.py:151>)
- **제안** method별로 캐시 루트, 인코더 종류와 파일명, SSL subset, 누락 시 동작을 매핑표에 적는다. 두 method 모두 캐시가 있을 때 의도한 파일을 읽고, 없을 때 해당 인코더를 명시하며 중단하는지 검증한다. 동료 결과와 일치성을 확인할 때는 양쪽에 동일한 인코더와 subset 파일을 넣고 해시를 기록한다.

**C2. [심각도: 중요] 인자 합집합만으로는 사전학습 호출과 이어하기를 보존할 수 없다.**

- **문제** `consistency_loss`는 우리 쪽에서 두 값을, 동료 쪽에서 여섯 값을 반환한다. `MaskedLatentModel`의 세 번째 위치 인자도 각각 `size`와 `view_attention`으로 의미가 다르다. 양쪽에 이어하기 구현이 이미 있고, 동료 쪽은 view-weight 이력과 view-module 파일도 별도로 저장한다.
- **근거** 우리 `pretrain`은 `loss, z`로 결과를 받고 FiLM에 인덱스를 전달한다. 동료 구현은 진단값도 받아 기록하며, A1에서는 projector를 optimizer와 EMA 업데이트 양쪽에 넣는다. [우리 손실 함수](<C:/Users/annan/OneDrive/바탕 화면/umm/wafer/ver6.py:72>) · [우리 이어하기](<C:/Users/annan/OneDrive/바탕 화면/umm/wafer/ver6.py:139>) · [동료 손실 및 EMA](<C:/Users/annan/OneDrive/바탕 화면/umm/debate/peer-wafer-defect/peer_repo/wafer/ver6.py:102>) · [동료 이어하기](<C:/Users/annan/OneDrive/바탕 화면/umm/debate/peer-wafer-defect/peer_repo/wafer/ver6.py:271>) · [동료 산출물 저장](<C:/Users/annan/OneDrive/바탕 화면/umm/debate/peer-wafer-defect/peer_repo/wafer/ver6.py:384>)
- **제안** 통합 전 호출부별 인자, 반환값, 저장 파일을 확정한다. 기존 호출의 반환 형식과 FiLM·BN 인덱스 처리를 보존하고, 동료 옵션은 키워드로 전달한다. 양쪽 `.ckpt`에서 이어갈 때 모델·optimizer·scheduler·RNG 복원 후 같은 다음 배치의 손실과 업데이트가 일치하는지 확인한다.

**C3. [심각도: 중요] 계획 4(b)의 비교는 일부 핵심 분기와 실제 미세조정 동작을 놓친다.**

- **문제** `k9.91` 분류기 `forward`가 일치해도, 클러스터링에 쓰는 임베딩과 첫 optimizer 업데이트는 달라질 수 있다. 사전학습 비교 목록에는 `k9.2`의 inter-view attention, `k9.3`의 target-similarity 가중치, `k9.4`의 target-conditioned gate가 빠져 있다.
- **근거** 동료 `ver9.91`은 `forward_features`를 호출하는 래퍼로 임베딩을 만든다. 우리 기본 구현은 `model.children()`으로 특징 추출기를 조립한다. 동료 모델은 별도 분기를 결합하는 `forward_features`를 정의하고, 미세조정은 belief 보조 손실과 gradient clipping을 적용한다. [우리 특징 추출](<C:/Users/annan/OneDrive/바탕 화면/umm/wafer/selfsup.py:215>) · [동료 특징 추출](<C:/Users/annan/OneDrive/바탕 화면/umm/debate/peer-wafer-defect/peer_repo/wafer/selfsup.py:245>) · [동료 모델 분기](<C:/Users/annan/OneDrive/바탕 화면/umm/debate/peer-wafer-defect/peer_repo/wafer/models.py:414>) · [동료 CLI 사전학습 분기](<C:/Users/annan/OneDrive/바탕 화면/umm/debate/peer-wafer-defect/peer_repo/wafixmatch.py:179>)
- **제안** 모든 method의 CLI 옵션, 결과 경로, 캐시 선택을 원본과 대조한다. `k9.91`은 동일 인코더와 배치로 임베딩, 주·보조 손실, optimizer 그룹과 첫 업데이트까지 비교한다. `k9.2/.3/.4`는 손실과 view weight도 비교한다. 체크포인트 로딩에서는 예상된 `missing_keys`와 비어 있는 `unexpected_keys`를 각각 확인한다. 현재 양쪽 `strict=False` 로딩 검사는 `unexpected_keys`를 확인하지 않는다. [우리 로딩](<C:/Users/annan/OneDrive/바탕 화면/umm/wafer/selfsup.py:180>) · [동료 로딩](<C:/Users/annan/OneDrive/바탕 화면/umm/debate/peer-wafer-defect/peer_repo/wafer/selfsup.py:179>)

**C4. [심각도: 중요] 결과 이관은 실제 CLI 경로와 로컬 파일을 함께 다뤄야 한다.**

- **문제** 동료 기본 CLI가 생성하는 폴더는 예시의 `ver9.91_ssl50_ft30`에 `_cos`가 붙은 `ver9.91_ssl50_ft30_cos`다. 또한 Git에 기록되는 이름 변경만으로 동료 PC의 `best.pt`와 `done`은 이동하지 않는다.
- **근거** 동료 CLI가 `ver9.91`의 `args.cosine`을 켜고 `ssl_tag()`가 `_cos`를 붙인다. CLI는 `done`이 있으면 실행을 건너뛰며, `.gitignore`는 `*.pt`, `*.ckpt`, `runs/**/done`을 제외한다. 우리 `ver9`의 seed 0 로그 본문은 `ver6.2`, seed 1은 `ver7` 실행을 기록한다. 두 seed 모두 `ver9`와 `ver8`의 `best.pt`가 각각 바이트 단위로 같았다. [동료 태그 생성](<C:/Users/annan/OneDrive/바탕 화면/umm/debate/peer-wafer-defect/peer_repo/wafixmatch.py:61>) · [cosine 설정](<C:/Users/annan/OneDrive/바탕 화면/umm/debate/peer-wafer-defect/peer_repo/wafixmatch.py:366>) · [제외 규칙](<C:/Users/annan/OneDrive/바탕 화면/umm/.gitignore:7>) · [seed 0 로그](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/log_ver9_p0.05_s0.txt:56>) · [seed 1 로그](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/log_ver9_p0.05_s1.txt:93>)
- **제안** 이관표는 실제 태그에서 생성하고 seed 0·1, `metrics.json`, `best.pt`, `done`, 캐시·진단 파일을 함께 점검한다. 스크립트를 제공한다면 먼저 이동 목록과 충돌을 출력하고, 재실행해도 기존 파일이 덮어써지지 않게 한다. `y9`로 옮기는 로그는 본문을 보존하고 기존 실행 출처를 매핑표에 남긴다.

**C5. [심각도: 중요] 우리 쪽 회귀 기준에 `y9`와 클래스별 검증이 빠져 있다.**

- **문제** 전체 macro F1이 `1e-3` 안에 들어도 클래스별 변화가 상쇄될 수 있다. 현재 검증 대상에는 실제로 이름과 캐시 매핑을 변경하는 `y9`가 없다.
- **근거** seed 0의 test macro F1은 ver8·ver8.1·ver8.2·ver8.3 순서로 `0.831365 / 0.822447 / 0.802571 / 0.835845`다. ver8.2의 Scratch F1은 `0.3667`이고 저장 결과에 혼동행렬이 있다. ver9는 seed 0·1에서 각각 val/test `0.818379/0.831365`, `0.794510/0.783472`다. [ver8](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/resnet18/p0.05/seed0/ver8_ssl50_ft30/metrics.json:291>) · [ver8.1](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/resnet18/p0.05/seed0/ver8.1_ssl50_ft30/metrics.json:298>) · [ver8.2](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/resnet18/p0.05/seed0/ver8.2_ssl50_ft30/metrics.json:299>) · [ver8.3](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/resnet18/p0.05/seed0/ver8.3_ssl50_ft30/metrics.json:300>) · [ver9 seed 0](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/resnet18/p0.05/seed0/ver9_ssl50_ft30/metrics.json:174>) · [ver9 seed 1](<C:/Users/annan/OneDrive/바탕 화면/umm/runs/resnet18/p0.05/seed1/ver9_ssl50_ft30/metrics.json:179>)
- **제안** 기존 `best.pt`를 원본과 통합 코드 양쪽에서 같은 split, 크기 정보, 모델 래퍼, 정밀도로 평가한다. ver8 계열 seed 0과 `y9` seed 0·1의 val/test 예측 또는 혼동행렬, 클래스별 F1을 대조한다. test는 고정된 체크포인트의 회귀 확인에만 쓰고 모델·설정 선택은 val로 한다.

**C6. [심각도: 경미] 문서와 분석 스크립트를 옮긴 뒤 실행 경로가 깨진다.**

- **근거** 동료 README는 `results/` 아래 그림을 상대경로로 참조한다. `show_confusion.py`는 결과 경로를 `_ssl50_ft30`으로 고정하므로 `_cos`가 붙는 실행을 찾지 못한다. [동료 README](<C:/Users/annan/OneDrive/바탕 화면/umm/debate/peer-wafer-defect/peer_repo/README.md:28>) · [그림 스크립트](<C:/Users/annan/OneDrive/바탕 화면/umm/debate/peer-wafer-defect/peer_repo/analysis/show_confusion.py:3>)
- **제안** 문서 이동과 함께 그림 링크를 갱신하고, 스크립트는 실제 결과 디렉터리를 받게 한다. 동료 문서의 `0.8527`은 원본 저장소의 보고값으로 출처를 남긴다. [동료 실험 보고](<C:/Users/annan/OneDrive/바탕 화면/umm/debate/peer-wafer-defect/peer_repo/EXPERIMENTS.md:118>)

## 빠진 대조군·검증

- **동일 입력과 동일 가중치로 원본 대 통합 비교.** seed만 맞추는 경우와 달리 초기화 순서나 캐시 출처의 영향을 통제할 수 있다. 동료 사본은 통합 시작 전에 `5446ff7`의 트리와 대조해 기준을 확정한다.
- **캐시가 있는 경우, 없는 경우, 중단 체크포인트가 있는 경우.** 각 경로에서 올바른 인코더와 subset을 쓰는지 확인한다. 운영 결과의 `done` 때문에 실행이 건너뛰어지는 경우를 피하도록 스모크 출력 경로도 별도로 잡는다. [실행 건너뛰기 분기](<C:/Users/annan/OneDrive/바탕 화면/umm/wafixmatch.py:132>)

## 더 싸거나 더 나은 대안

4(d)는 먼저 **기존 SSL 캐시를 실제 CLI로 읽고, 한 라운드에서 optimizer step 한 번과 저장·재로딩까지** 확인하는 편이 싸다. 현재 `--ft_epochs 1`은 라운드당 1 epoch이므로 총 3 epoch이다. 각 라운드는 `v3_nopl`에서도 임베딩과 클러스터링을 수행하며, `k9.91`의 warmup은 5 epoch이라 이 실행만으로 cosine 구간을 검증할 수 없다. LR 값은 warmup 시작·경계·종료 지점에서 별도로 대조하면 된다. [동료 미세조정 루프](<C:/Users/annan/OneDrive/바탕 화면/umm/debate/peer-wafer-defect/peer_repo/wafer/selfsup.py:245>) · [스케줄러](<C:/Users/annan/OneDrive/바탕 화면/umm/debate/peer-wafer-defect/peer_repo/wafer/selfsup.py:224>)