# 01. 동료 코드 통합 계획 (capstone-WM-811K ← Hanbandoche/wafer-defect) — Claude

## 배경과 사용자 결정 (2026-11-22)
- 메인 저장소 = capstone-WM-811K (사용자). 동료는 Collaborator 로 초대됨. 사용자 브랜치 = `yeeun` (b207fc9, push 완료).
- 실험 이름: 사용자 새 실험 `y9.x`, 동료 실험 `k9.x` (예: 동료 ver9.91 → `k9.91`).
- 두 저장소는 같은 코드에서 갈라졌지만 git 이력이 이어져 있지 않다 (각자 첫 커밋). → GitHub PR 로 바로 못 합침. 한 번은 손으로 통합해야 한다.
- 동료 코드 사본: `debate/peer-wafer-defect/peer_repo/` (동료 커밋 5446ff7).

## 차이 (우리 yeeun vs 동료)
| 파일 | 동료에만 | 우리에만 | 내용 |
|---|---|---|---|
| wafer/models.py | +389 | −90 | 동료: SpatialSelfAttention, IterativeSpatialBeliefAttention, OrientedLineBranch, SpatialEvidencePooling, `USE_SA` 분기. 우리: IMAGENET, SizeHead, FiLMResNet/FiLMFeatures, ClassPool, load_size |
| wafer/ver6.py | +260 | −50 | 동료: InterViewSelfAttention, TargetConditionedViewGate, projector, loss_mode (InfoNCE/Uniformity/VICReg/Debiased), flip. 우리: `_predict(bn_groups, idx)`, size(FiLM), 체크포인트 이어하기(확인 필요). **같은 함수 인자를 서로 다르게 바꿈 → 충돌** |
| wafer/selfsup.py | +90 | −34 | 동료: attn_wd 파라미터 그룹, warmup+cosine, label smoothing, belief 보조 손실, hflip, grad clip. 우리: film / size_head / class_pool, embed 의 needs_idx |
| wafixmatch.py | +79 | −32 | 양쪽 method 목록·분기 |
| wafer/engine.py | +12 | −5 | evaluate / _Tracker 일부 |
| data.py, augment.py, patterns.py | 0 | 0 | 동일 |
| 동료에만 있는 파일 | | | README.md, EXPERIMENTS.md, ROADMAP.md, requirements.txt, analysis/*.py (6개) |

## 계획
1. **기준 = 우리 yeeun 코드.** 동료 변경을 함수 단위로 옮긴다 (git merge 가 아니라 수동 3-way: 공통 조상이 없으므로 "동료에만 있는 변경" 과 "우리에만 있는 변경" 을 각각 보존).
   - 충돌 함수 (ver6.py `MaskedLatentModel.__init__`, `consistency_loss`, `_views`, `pretrain`, selfsup.py `finetune`, `embed`, engine.py `evaluate`) 는 **양쪽 인자를 모두 기본값과 함께 받는 합집합** 으로 만들고, 기본값일 때 각자의 원래 동작과 같게 한다.
2. **이름 변경.**
   - 동료 29개: `ver` → `k` (ver9.1…ver9.91, ver9.9A/B/B1~B4, ver9.7a0~a5, ver9.8, ver9L1~L4, ver9A1, ver999 → k999, ver3.991 → k3.991).
   - 우리 `ver9` → `y9` (runs 폴더 `ver9_ssl50_ft30` → `y9_ssl50_ft30`, 로그, metrics.json 의 method, 인코더 캐시 `masklat9_e50.pt` 는 파일명 유지하고 매핑만).
   - 공통 조상 (ver1 ~ ver7.X) 과 우리 ver8.x 는 그대로.
   - `startswith(("ver3","ver6","ver7","ver8"))` 같은 접두어 검사에 `y`, `k` 추가. 인코더 캐시 이름 매핑 (예: k9.91 → masklat7, k999 → 동료 원래 이름) 을 명시.
3. **동료 문서·분석 스크립트**: `docs/k/` (README, EXPERIMENTS, ROADMAP — 버전 이름을 k 로 치환) 와 `analysis/` 로 넣는다. requirements.txt 는 루트에.
4. **검증 (학습 없이 / 짧게).**
   - (a) 우리 쪽 회귀: 저장된 best.pt 로 ver8, ver8.1, ver8.2, ver8.3 (seed 0) 을 다시 평가해 metrics.json 의 test macro 와 1e-3 이내 일치.
   - (b) 동료 쪽 일치성: 같은 seed·같은 입력으로 **동료 원본 models.py 와 통합 models.py** 에서 k9.91 모델을 만들어 파라미터 키·개수·forward 출력이 같은지 비교. ver6 의 k9L1~L4, k9A1, k999 도 손실값 1배치 비교.
   - (c) 모든 method 이름에 대해 CLI 인자 처리 → 모델 생성 → 1 배치 forward/backward 가 오류 없이 도는지 (dry-run 스크립트).
   - (d) k9.91 fine-tune 1 epoch 실제 실행 (seed 0, 수 분) — 경로·로딩 확인.
5. **전달.** yeeun 에 커밋 → push → `yeeun → main` PR 생성 (설명에 이름 매핑표, 동료가 할 일). 동료에게: 로컬 결과 폴더·캐시 이름을 새 이름으로 바꾸는 매핑표 (또는 스크립트) 제공. 앞으로 동료는 자기 브랜치에서 `k` 이름으로 작업.

## 위험
- 수동 통합 중 동료 기능을 빠뜨리거나 동작을 바꿀 위험 → 4(b) 로 확인.
- 동료가 로컬에서 쓰던 이름 (ver9.91_ssl50_ft30 등) 과 캐시가 새 이름과 안 맞아 동료가 다시 학습하게 될 위험 → 매핑표.
- 동료 저장소의 최신 커밋이 5446ff7 이후 바뀌었으면 통합 기준이 어긋남 → 시작 직전에 다시 fetch.
비용: 제 작업 약 1~2시간, GPU 는 4(d) 수 분.
