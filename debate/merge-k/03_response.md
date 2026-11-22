# 03. Codex 비판에 대한 응답 — Claude

| 지적 | 응답 | 반영 |
|---|---|---|
| C1 캐시 매핑에 인코더 종류·누락 동작 없음 | 수용 | method 별 매핑표 (캐시 루트, 인코더 종류·파일명, SSL subset, 누락 시 동작). k9.91 → masklat7, k3.991 → simclr_random_cut. 누락 시 해당 인코더 이름을 말하며 중단. 동료 원본의 k3.991 오류 문구 버그도 수정 |
| C2 인자 합집합만으로 부족 (반환값·위치 인자·이어하기) | 수용 | 호출부별 인자·반환·저장 파일 표를 먼저 확정. `MaskedLatentModel` 은 동료 옵션을 **키워드 전용** 으로 받고 우리 `size` 위치 유지. `consistency_loss` 는 기본 호출 시 우리 형식 (loss, z), 동료 진단이 필요한 호출만 확장 반환. 이어하기는 동료 쪽 저장 항목 (view weight 이력, view module, projector) 을 합쳐 하나로, 복원 후 다음 배치 손실 일치 확인 |
| C3 forward 비교만으로 부족 | 수용 | k9.91: 같은 인코더·배치로 임베딩 (forward_features), 주·보조 손실, optimizer 그룹, 첫 업데이트 후 파라미터까지 원본과 비교. k9.2/9.3/9.4 손실·view weight 비교 추가. 로딩 시 missing / unexpected key 둘 다 검사 |
| C4 결과 이관 (_cos 태그, best.pt·done 은 git 밖) | 수용 | 동료 이관표는 실제 `ssl_tag()` 로 생성 (`ver9.91_ssl50_ft30_cos` → `k9.91_ssl50_ft30_cos`). 이관 스크립트는 먼저 목록·충돌만 출력 (dry-run), 실행 시 덮어쓰지 않음. y9 로그 본문은 그대로 두고 출처 (ver6.2 / ver7 실행 기록의 복사본) 를 매핑표에 남김 |
| C5 회귀에 y9·클래스별 빠짐 | 수용 | ver8 / 8.1 / 8.2 / 8.3 seed 0 + y9 seed 0·1 을 원본 코드와 통합 코드에서 같은 조건으로 평가해 **예측 단위** 로 비교 (혼동행렬·클래스별 F1 동일) |
| C6 문서 링크·스크립트 경로 | 수용 | docs/k/ 로 옮기며 그림 링크 수정, show_confusion.py 는 결과 폴더를 인자로. 0.8527 은 "동료 원본 저장소 보고값" 으로 출처 표기 |
| 대안 (4d 를 1 step + 저장·재로딩으로) | 수용 | 3 epoch 실행 대신: 실제 CLI 로 캐시 읽기 → optimizer 1 step → 저장·재로딩. LR 은 warmup 시작·경계·종료 값을 계산으로 대조. 스모크 출력은 별도 runs 경로 (done 건너뛰기 방지) |

## 수정한 계획
0. 시작 직전 동료 저장소 다시 fetch, 사본이 5446ff7 트리와 같은지 확인 (바뀌었으면 사용자에게 보고).
1. 호출부·반환값·저장 파일 표 + method 별 캐시 매핑표 작성 (debate/merge-k/mapping.md).
2. 수동 통합 (기준 yeeun), 이름 변경 (동료 29개 → k, ver9 → y9).
3. 검증: C1 (캐시 있음/없음), C2 (이어하기), C3 (k9.91 1 step 동등성, k9.2~9.4 손실), C5 (회귀 예측 단위), 모든 method dry-run, CLI 1 step 스모크.
4. docs/k/, analysis/ 이관 + 동료용 이관 스크립트 (dry-run 기본).
5. yeeun 커밋·push → `yeeun → main` PR (매핑표·동료 할 일 포함).
비용: 제 작업 약 2~3시간, GPU 는 검증용 수 분.
