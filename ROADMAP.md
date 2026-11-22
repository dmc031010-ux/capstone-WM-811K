# Project Roadmap — WM-811K 웨이퍼빈맵 결함 탐지

> **최종 목표**: 5% 소량 라벨 환경에서 **우리의 잠재 다중 마스킹 사전학습 + 웨이퍼 특화 파인튜닝(`ver9.91`)** 조합으로 `SimCLR` 기준선(`ver3.991: 0.8674`)을 **전체 Macro F1과 최약체 `Scratch` 모두에서 확실하게 상회**하고, 다중 시드·다양한 라벨 비율에서 재현성을 입증하는 것.

---

## 전체 로드맵 흐름

```mermaid
flowchart LR
    P1["Phase 1 ✅<br/>준지도 한계 진단<br/>→ SSL 전환"]
    --> P2["Phase 2 ✅<br/>잠재 다중 마스킹<br/>레퍼런스(ver7) 확립"]
    --> P3["Phase 3 ✅<br/>Scratch 약점<br/>5대 가설 검증"]
    --> P4["Phase 4 ✅<br/>파인튜닝 구조 혁신<br/>ver9.91 (0.8527)"]
    --> P5["Phase 5 🔄<br/>사전학습 표현력 고도화<br/>SimCLR(0.8674) 돌파"]
    --> P6["Phase 6 📋<br/>Multi-seed &<br/>라벨 비율 확장 검증"]
```

---

## 마일스톤별 진행 현황

| Phase | 목표 | 핵심 지표 / 산출물 | 상태 |
|---|---|---|---|
| **Phase 1** | 준지도(FixMatch) 한계 규명 및 SSL 전환 | `WaFixMatch 0.795` → `ver3 SimCLR 0.8391` | ✅ 완료 |
| **Phase 2** | 디코더 없는 잠재 다중 마스킹(`ver7`) 설계 | `ver6 0.786` → `ver7 0.8314` (6px 4뷰 바둑판) | ✅ 완료 |
| **Phase 3** | `Scratch` 취약 원인 및 seed 불안정성 진단 | `diesize`, `probe`, `ver7.91(0.8371)` | ✅ 완료 |
| **Phase 4** | 웨이퍼 특화 파인튜닝 아키텍처 완성 | **`ver9.91` 2-seed `0.8527 ± 0.0001` / Scratch `0.6777`** | ✅ 완료 |
| **Phase 5** | 사전학습 반발력·Projector 도입으로 SimCLR 돌파 | `ver9L1~L4`, `ver9A1` → 최종 결합 체크포인트 (`> 0.8674`) | 🔄 진행 중 |
| **Phase 6** | 최종 Ablation, Multi-seed, 라벨 비율 확장 | `Line+Pool` 검증, `seed 0/1/2`, `frac 1%/5%/10%` | 📋 예정 |

---

## 상세 체크리스트

### Phase 1. 베이스라인 구축 및 준지도 한계 진단 ✅
- [x] `Supervised(5%)`, `FixMatch`, `WaFixMatch` 재현 및 오분류 분석
- [x] 손실 감쇠(`ver1`, `ver1.1`), 패턴 사전(`ver2`) 검증 → pseudo-label 86~88% `none` 쏠림 확인
- [x] 10만 장 자기지도 학습(`ver3` SimCLR: `0.8391`)으로 방향 전환 및 군집 PL 역효과 확인(`--v3_nopl`)

### Phase 2. 독창적 잠재 다중 마스킹 사전학습(`ver7`) 확립 ✅
- [x] `ver6` (`0.7859`) 초기 붕괴 진단 → `ver6.1` → **`ver6.2/ver7` (`0.8314`)** 설계
- [x] 핵심 요소 Ablation 완료: 칸 크기(`12/6/4px`), 뷰 수(`3/4`), 비율(`30%/50%`), MLP Predictor, 백본(`R18/R34`)
- [x] 형제 뷰 간 정보 누수(치팅) 없음 검증 (`analysis/leak.py`)

### Phase 3. `Scratch` 취약 원인 정밀 진단 ✅
- [x] 96×96 리사이즈 손실 가설 검증 → 기각 (`analysis/scratch_resize.py`, `ver7.3`)
- [x] die 수 구간별 재현율 분석 → 소형 웨이퍼(`<1200 die`) 약점 특정 (`analysis/diesize.py`, `diecell.py`)
- [x] 뷰 게이트/어텐션(`ver9.2~9.4`), 손실 가중치(`ver7.4~7.83`) 검증 → **부분 균형 샘플링(`ver7.91: 0.8371`)** 유효 확인
- [x] Frozen Probe(`analysis/probe.py`) 및 `seed 1` 복제로 사전학습 분리력·고정 LR 불안정성(`±0.0239`) 확인

### Phase 4. 파인튜닝 아키텍처 혁신 (`ver9.1` → `ver9.91`) ✅
- [x] 어텐션 최적 위치 탐색 (`layer4/3/2` → `layer3 6×6` 확정)
- [x] **`ver9.7`** (`IterativeSpatialBeliefAttention` 3-step + `SpatialEvidencePooling` + 안정화 레시피): 2-seed `0.8395 ± 0.0016` + 6종 leave-one-out ablation (`a0~a5`)
- [x] **`ver9.91`** (`+ OrientedLineBranch` + `hflip`): 2-seed **`0.8527 ± 0.0001`**, Scratch **`0.6777 ± 0.0000`**, 전 9개 클래스 동반 상승
- [x] Forward Ablation (`ver9.9A`, `ver9.9B~B4`): `ver9.9B2` (`0.8479`, 순정 학습 100%)로 구조 시너지 입증

### Phase 5. 사전학습 표현력 고도화 — SimCLR(`0.8674`) 돌파 🔄
- [x] 공정 대조 실험 (`ver3.991` SimCLR `0.8674` vs `ver9.91` `0.8526`): **`Scratch` 우리 우위(`+1.92%p`)**, `Donut/Loc/Center` 열세 원인(Projector·반발력 부재) 특정
- [x] 단일 변경 통제 실험 1차 완료: `ver999` (`hflip`: `0.8099`), `ver9L1` (Masked InfoNCE: `0.8168`), `ver9L2` (Uniformity: `0.8240`)
- [ ] 단일 변경 통제 실험 2차 완료 *(실행 중)*: `ver9L3` (VICReg), `ver9L4` (Debiased InfoNCE), `ver9A1` (경량 `Projector + Predictor`, `231K`)
- [ ] 최적 사전학습 결합 모델 설계 및 50 epoch 체크포인트 생성 *(후보: `A1 경량 Projector` + 최적 반발력 + 다중 스케일 바둑판 `4/6/12px` + EMA Cosine/Warmup)*
- [ ] 신규 사전학습 체크포인트 + `ver9.91` 파인튜닝으로 **`ver3.991` (`0.8674`) 정면 돌파 확인**

### Phase 6. 최종 검증 및 리포트 완성 📋
- [ ] 남은 구조 ablation 1종 완료: **`Line + Pool`** (`ver9.9B2`에서 `Belief`만 제거) → 3각 시너지 최종 확정
- [ ] 최종 승자 모델 **Multi-seed (`seed 0, 1, 2`)** 평균·표준편차 보고
- [ ] **라벨 비율 확장 실험 (`frac = 1%, 5%, 10%`)**: 극소량 라벨 환경에서의 강건성 입증
- [ ] 최종 혼동행렬·클래스별 비교표·아키텍처 도식화 완성 및 최종 리포트 정리
