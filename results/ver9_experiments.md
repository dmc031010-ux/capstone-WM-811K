# `ver9` 아키텍처 & 실험 가이드 (`ver9.1` ~ `ver9.8`)

이 문서는 `main` 브랜치에 병합된 `ver9` 계열 실험(`ResNet-18`, WM-811K `96×96`, 5% 라벨 6,919장 + Unlabeled 100,000장, `split_seed=0`)의 **도입 목적 · 동작 원리 · 코드 위치 · 실측 수치 · 실행 CLI**를 정리한 기술 명세서입니다. (사람 및 AI 코딩 에이전트 모두 이 문서를 기준으로 코드 구조와 실험 이력을 파악할 수 있습니다.)

---

## 1. 전체 요약표 (`seed=0` 기준)

| `--method` | 분류 / 출처 | SSL 인코더 캐시 | `USE_SA` / 핵심 설정 | Val Macro F1 | **Test Macro F1** | **Scratch F1 (`TP/120`, `n→S`)** | 결과 로그 (`metrics.json`) |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| `ver3` | SimCLR 기준선 | `simclr_random_cut_e50.pt` | `False` | - | `0.8391` | `0.6161` (`69/120`, `19`) | - |
| `ver7` | MaskLat 기준선 (`ver8` 동일) | `masklat7_e50.pt` | `False` | - | `0.8314` | `0.4766` (`56/120`, `44`) | - |
| `ver9.1` | Frangi 2채널 입력 (`yeeun`) | `masklat8_e50.pt` | `FrangiInput`, `no_test=True` | `0.832` | *(미평가)* | *(val 0.561)* | `runs/.../seed0/ver9.1_ssl50_ft30/metrics.json` |
| `ver9.2A` | Spatial SA (`layer4` 3×3) | `masklat7_e50.pt` | `True`, `use_gate=False` | - | `0.8253` | `0.4715` | - |
| **`ver9.2B`** | Spatial SA (`layer3` 6×6) | `masklat7_e50.pt` | `"layer3"`, `use_gate=True` | - | `0.8286` | `0.4956` | - |
| `ver9.2C` | Spatial SA (`layer2` 12×12) | `masklat7_e50.pt` | `"layer2"`, `use_gate=True` | - | `0.8178` | `0.4740` | - |
| `ver9.3A` | SSL 4뷰 Inter-View SA | `masklat93a_e50.pt` | `view_attention=True` | - | `0.8296` | `0.4150` | - |
| `ver9.3B` | SSL 4뷰 Target Sim Gate | `masklat93b_e50.pt` | `view_gate_mode="target_similarity"` | - | `0.8148` | - | - |
| `ver9.3C` | SSL 4뷰 Cond Gate | `masklat93c_e50.pt` | `view_gate_mode="target_conditioned"` | - | `0.8165` | - | - |
| **`ver9.4`** | **`Line` 기본형** | `masklat7_e50.pt` | `"line_base"` | - | `0.8230` | `0.5024` (`52/120`, `30`) | `runs/.../seed0/ver9.4_ssl50_ft30/metrics.json` |
| **`ver9.5`** | **`Belief` 기본형** | `masklat7_e50.pt` | `"belief_base"`, `aux_belief_w=0.3` | - | `0.8288` | `0.4074` (`44/120`, `49`) | `runs/.../seed0/ver9.5_ssl50_ft30/metrics.json` |
| **`ver9.6`** | **`Pool` 기본형** | `masklat7_e50.pt` | `"pool_base"` | `0.8147` | `0.8275` | `0.4396` (`60/120`, `82`) | `runs/.../seed0/ver9.6_ssl50_ft30/metrics.json` |
| **`ver9.7A`** | **`ver7` + 3모듈 통합** | `masklat7_e50.pt` | `"three_base"`, `aux_belief_w=0.3` | `0.8272` | `0.8297` | `0.5900` (`59/120`, `17`) | `runs/.../seed0/ver9.7A_ssl50_ft30/metrics.json` |
| **`ver9.7B`** ⭐ | **`SimCLR` + 3모듈 통합** | `simclr_random_cut_e50.pt` | `"three_base"`, `aux_belief_w=0.3` | `0.8251` | **`0.8640`** | `0.6204` (`67/120`, `24`) | `runs/.../seed0/ver9.7B_ssl50_ft30/metrics.json` |
| **`ver9.8A`** ⭐ | **`ver9.7A` + 안정화 레시피** | `masklat7_e50.pt` | `"three_base"` + 5종 레시피 (`feat/hanbandoche ver9.4`) | - | **`0.8527 ± 0.0001`** *(2-seed)* | **`0.6777 ± 0.0000`** | - |
| **`ver9.8B`** ⭐ | **`ver9.7B` + 안정화 레시피** | `simclr_random_cut_e50.pt` | `"three_base"` + 5종 레시피 (`feat/hanbandoche ver9.4_simclr`) | - | **`0.8674`** | `0.6585` | - |

---

## 2. 왜 3대 모듈(`ver9.4` + `ver9.5` + `ver9.6` → `ver9.7`)을 도입했는가?

WM-811K에서 `Scratch`가 가장 자주 `none`으로 오분류되는 원인을 분해한 결과, 표준 `ResNet-18` 구조에 3가지 병목이 존재함을 확인했습니다.

1. **초반 입력 병목 (`Line` → `ver9.4`)**: `conv1(stride 2) + maxpool(stride 2)`를 지나며 `96×96 → 24×24`로 축소될 때, 두께 1~2px짜리 얇은 `Scratch` 선이 배경(`0`)과 평균 내어지며 소실됨.
2. **중단 문맥 병목 (`Belief` → `ver9.5`)**: CNN `3×3` 국소 커널만으로는 끊어진 선 조각을 하나의 `Scratch`로 이어 붙이지 못하고, 단순 어텐션(`ver9.2x`)은 클래스 관점의 신념 추론이 부족함.
3. **말단 출력 병목 (`Pool` → `ver9.6`)**: 마지막 `AdaptiveAvgPool2d((1, 1))`이 `3×3` 9칸을 동일하게 `1/9` 평균 내면서, 1~2칸에만 걸친 국소 `Scratch` 신호가 정상 배경 7~8칸에 희석됨.

---

## 3. 버전별 상세 설명 및 코드 위치

### `ver9.1`: Frangi 선 강조 2채널 입력 (`yeeun` 브랜치 `y9.2a`)
- **파일**: `wafer/frangi.py` (`vesselness`), `wafer/models.py` (`FrangiInput`, `FrangiFeatures`), `wafer/selfsup.py`
- **도입 목적**: 소형 웨이퍼 리사이즈 전 **원본 die 격자** 위에서 2차 미분 헤시안 고유값 비율로 관/선형 구조를 추출해 2번째 입력 채널로 제공.
- **동작 원리**: `conv1`을 2채널로 확장하되 기존 채널 0 가중치는 보존하고, 새 Frangi 채널 1 가중치는 `0`에서 시작.

### `ver9.2` (`ver9.2A / ver9.2B / ver9.2C`): Spatial Self-Attention 위치 탐색
- **파일**: `wafer/models.py` (`PositionalEncoding2D`, `SpatialSelfAttention`, `build_model` 내 `USE_SA` 분기), `wafer/selfsup.py` (`sa_prefixes` 누락 키 허용)
- **도입 목적**: CNN이 못 보는 멀리 떨어진 결함 조각 간 전역 문맥 연결 및 **ResNet 내 최적 삽입 위치 탐색**.
- **동작 원리**:
  - `(B, C, H, W)` 특징 맵에 학습 가능한 `PositionalEncoding2D`를 더하고 `H*W` 토큰 간 4-head Self-Attention + FFN 수행.
  - `ver9.2A` (`layer4` 3×3, 9토큰) · **`ver9.2B` (`layer3` 6×6, 36토큰, `use_gate=True`, 최적 위치 확정)** · `ver9.2C` (`layer2` 12×12, 144토큰, 과적합).
  - `use_gate=True`는 `attn_scale`/`ffn_scale`을 `0`으로 초기화해 사전학습 특징을 보호하며 출발.

### `ver9.3` (`ver9.3A / ver9.3B / ver9.3C`): SSL 4뷰 어텐션 · 게이트 결합 실험
- **파일**: `wafer/ver6.py` (`InterViewSelfAttention`, `TargetConditionedViewGate`, `consistency_loss`, `pretrain`)
- **도입 목적**: 바둑판 마스킹 사전학습(`ver7`) 시 4개 뷰 예측 벡터를 단순 `1/4` 산술 평균하는 대신, 뷰 간 어텐션 또는 타깃 유사도 기반 가중 결합을 쓰면 표현력이 오를지 검증.
- **실험 결과 및 교훈 (기각)**:
  - `ver9.3A` (`InterViewSelfAttention`, `0.8296`) · `ver9.3B` (`target_similarity`, `0.8148`) · `ver9.3C` (`TargetConditionedViewGate`, `0.8165`).
  - 뷰 가중치를 열어주면 결함이 덜 가려진 쉬운 뷰 1장에만 가중치가 쏠리는 **편식(winner-take-all)**이 발생해 단순 평균(`ver7` `0.8314`)보다 하락. → 어텐션은 지도 신호가 있는 파인튜닝에 배치.

### `ver9.4`: `OrientedLineBranch` 기본형 (`USE_SA="line_base"`)
- **파일**: `wafer/models.py` (`OrientedLineBranch`), `wafer/selfsup.py` (`_ForwardFeatures`)
- **도입 목적**: `conv1 + maxpool` 초반 병목에서 1~2px `Scratch` 선이 소실되는 문제를 우회하기 위해, **원본 `96×96` 입력에서 얇은 선만 직접 추출해 `layer3` 뒤(`6×6`)에 합류**시키는 보조 스트림.
- **동작 원리**:
  1. **평균 덜어내기 (`_anom`)**: `anom = (defect - mean_d) * wafer`로 웨이퍼 전체 배경 노이즈 바닥과 외곽 경계 가짜 반응을 먼저 제거.
  2. **8방향 zero-mean 리지 필터 (`k9`, `k15`)**: `9×9`, `15×15` 격자에 8각도(`22.5°` 간격) 직선 커널(중앙 `+`, 양옆 `-0.5` 억제 날개)을 적용해 뭉친 결함(`Loc`)과 등방성 노이즈(`none`)를 상쇄.
  3. **방향성 이방성 (`_resp`)**: 8방향 응답에서 최댓값(`max`)과 `max - mean`을 추출.
  4. **피크 보존 축소 (`net`)**: 6채널(`defect, anom, max9, aniso9, max15, aniso15`)을 경량 Conv + **stride-2 대신 `MaxPool2d(2)` 4회**(`96→48→24→12→6`)로 축소해 `line_scale`(초기값 `0.05`)로 `layer3` 출력에 합산.

### `ver9.5`: `IterativeSpatialBeliefAttention` 기본형 (`USE_SA="belief_base"`)
- **파일**: `wafer/models.py` (`IterativeSpatialBeliefAttention`), `wafer/selfsup.py` (`aux_belief_w=0.3`, `grad_clip=1.0`)
- **도입 목적**: `ver9.2B`(`6×6` 기본 어텐션)를 고도화하여, 각 공간 칸이 결함 클래스 관점의 임시 신념(Belief)을 만들고 반복적으로 주변 문맥과 상호 보완하며 끊어진 선 조각을 잇도록 설계.
- **동작 원리 (3-step 가중치 공유 반복)**:
  1. **칸별 임시 신념(Belief) 점등**: `belief_head`로 36칸 각각의 9-class 확률을 뽑아 `belief_proj`로 토큰에 주입.
  2. **2D 상대 위치 어텐션 (`_rel_mha`)**: 칸 간 상대 좌표 `(Δy, Δx)` 바이어스 테이블(`rel_pos_bias`)을 더해 거리·방향 기반 문맥 연결.
  3. **연속성 FFN (`_conv_ffn`)**: `3×3 Depthwise Conv`가 포함된 FFN으로 상하좌우/대각 인접 연속성을 다듬고 `0.05` 초기화 LayerScale(`attn_scales`, `ffn_scales`)로 안정적 잔차 연결.
  4. **보조 감독 (`last_step_logits`)**: 매 스텝 끝 칸별 로짓을 `cell_gate` 가중합해 `aux_belief_w=0.3` 보조 CE 손실 적용.

### `ver9.6`: `SpatialEvidencePooling` 기본형 (`USE_SA="pool_base"`)
- **파일**: `wafer/models.py` (`SpatialEvidencePooling`)
- **도입 목적**: `layer4` 뒤 `3×3` 9칸을 균등 `1/9`로 평균 내는 기존 `AdaptiveAvgPool2d`가 국소 `Scratch` 신호를 희석하는 말단 병목 해소.
- **동작 원리**: `LayerNorm → Linear(128) → GELU → Linear(1)` 게이트의 마지막 층을 **`0`으로 초기화**해 학습 시작 시 기존 평균 풀링(`1/9`)과 동일하게 출발하고, 학습되면서 결함 신호가 켜진 공간 칸으로 풀링 가중치를 집중.

### `ver9.7` (`ver9.7A` · `ver9.7B`): 3모듈 통합 기본형 (`USE_SA="three_base"`)
- **파일**: `wafer/models.py` (`USE_SA == "three_base"`), `wafixmatch.py` (`--method ver9.7A / ver9.7B`)
- **도입 목적**: 단독 적용 시 각각 앞(`ver9.4`), 중간(`ver9.5`), 끝(`ver9.6`)에 머물렀던 3개 모듈을 하나의 ResNet-18 파이프라인으로 통합해 3대 병목을 동시에 해소하고, **SSL 백본에 무관한 범용 구조**임을 검증.
- **구성 및 성과**:
  - **`ver9.7A`** (`ver7` MaskLat 백본 + 3모듈): `seed=0` Test `0.8297`, Scratch `0.5900` (`seed=1`: Test `0.8457`, Scratch `0.6696`).
  - **`ver9.7B`** (`SimCLR` `ver3` 백본 + 3모듈): `seed=0` **Test `0.8640`** (`SimCLR` 단독 `0.8391` 대비 **`+2.49%p`**), Scratch `0.6204` (`TP=67/120`, `n→S=24`).

### `ver9.8` (`ver9.8A` · `ver9.8B`): 3모듈 + 5종 파인튜닝 안정화 레시피 (`feat/hanbandoche` `ver9.4` · `ver9.4_simclr`)
- **파일**: `wafixmatch.py` (`--method ver9.8A / ver9.8B`), `wafer/selfsup.py` (`attn_wd`, `warmup_epochs`+`cosine`, `label_smoothing`, `use_hflip`), `wafer/models.py` (`USE_SA="three_base"`)
- **도입 목적**: `ver9.7`(순정 고정 LR + `1:1` 균형 샘플링)의 **과도한 `none→defect` 오탐**과 **실행·seed 간 변동성**을 없애고, 전 클래스 성능과 다중 seed 일관성을 극대화하기 위해 도입.
- **5종 안정화 레시피**:
  1. `sample_pow = 0.35`: 완전 `1:1` 균형 대신 장수 비례 완화로 `none→defect` 오탐 절반 이상 억제
  2. `label_smoothing = 0.05`: 5% 소량 라벨 과적합/과신 방지 (`Ls`, `La` 동시 적용)
  3. `attn_wd = 0.02`: 새로 추가된 `Line/Belief/Pool/fc` 가중치에만 강한 Weight Decay 분리 적용
  4. `warmup_epochs = 5 + cosine` (`0.1→1.0→0.05`): 초반 사전학습 백본 보호 후 부드러운 코사인 수렴
  5. `use_hflip = True`: 약증강에 회전 외에 좌우 반전(`A.hflip`) 추가 (`+ aux_belief_w=0.3`, `grad_clip=1.0`)
- **검증 결과**:
  - **`ver9.8A`** (`ver7` 인코더): 2-seed Test **`0.8527 ± 0.0001`**, Scratch **`0.6777 ± 0.0000`** (seed 편차 사실상 0 달성)
  - **`ver9.8B`** (`SimCLR` 인코더): `seed=0` Test **`0.8674`**, Scratch **`0.6585`** (`Scratch`는 바둑판 마스킹 `ver9.8A`가 `+1.92%p` 우위)

---

## 4. 실행 CLI 예시

```bash
# ver9.2 (Spatial Self-Attention 위치 탐색)
python wafixmatch.py run --frac 0.05 --seed 0 --method ver9.2B

# ver9.3 (SSL 4뷰 결합 탐색)
python wafixmatch.py run --frac 0.05 --seed 0 --method ver9.3A

# ver9.4 / ver9.5 / ver9.6 (Line / Belief / Pool 기본형 단독)
python wafixmatch.py run --frac 0.05 --seed 0 --method ver9.4
python wafixmatch.py run --frac 0.05 --seed 0 --method ver9.5
python wafixmatch.py run --frac 0.05 --seed 0 --method ver9.6

# ver9.7A / ver9.7B (3모듈 통합 기본형: ver7 / SimCLR 백본)
python wafixmatch.py run --frac 0.05 --seed 0 --method ver9.7A
python wafixmatch.py run --frac 0.05 --seed 0 --method ver9.7B

# ver9.8A / ver9.8B (3모듈 + 5종 안정화 레시피: ver7 / SimCLR 백본)
python wafixmatch.py run --frac 0.05 --seed 0 --method ver9.8A
python wafixmatch.py run --frac 0.05 --seed 0 --method ver9.8B
```
