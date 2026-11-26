# capstone-WM-811K

WM-811K 웨이퍼 빈맵(WBM, `96×96`, 9-class, 5% 라벨 6,919장 + Unlabeled 100,000장, `ResNet-18` from scratch) 불량 분류 프로젝트입니다.

## 주요 파일 구성
- `wafixmatch.py`: 통합 실험 진입점 (`prepare`, `run`, `summary`)
- `wafer/models.py`: ResNet 백본 및 `ver8`~`ver9` 아키텍처 모듈 (`FrangiInput`, `SpatialSelfAttention`, `OrientedLineBranch`, `IterativeSpatialBeliefAttention`, `SpatialEvidencePooling`)
- `wafer/ver6.py`: Latent Multi-Masking Ensemble Consistency 사전학습 (`ver6`~`ver8`, `ver9.3A/B/C`)
- `wafer/selfsup.py`: SimCLR 사전학습(`ver3`) 및 라벨 미세조정 파이프라인
- `wafer/frangi.py`: `ver9.1` Frangi vesselness 2D 선 강조 필터
- [`results/ver6_experiments.md`](./results/ver6_experiments.md): `ver3` ~ `ver7` 실험 기록
- **[`results/ver9_experiments.md`](./results/ver9_experiments.md)**: **`ver9.1` ~ `ver9.7` 도입 목적 · 동작 원리 · 코드 위치 · 실측 수치 · CLI 가이드**

## `ver9` 버전 인덱스 (`main` 브랜치)
| 버전 (`--method`) | 요약 | 핵심 모듈 / 위치 |
| --- | --- | --- |
| `ver9.1` | Frangi 2채널 입력 (`yeeun` 브랜치) | `wafer/frangi.py`, `FrangiInput` |
| `ver9.2A / 9.2B / 9.2C` | Spatial Self-Attention 위치 탐색 (`layer4` / `layer3` / `layer2`) | `SpatialSelfAttention` (`wafer/models.py`) |
| `ver9.3A / 9.3B / 9.3C` | 사전학습(SSL) 4뷰 어텐션 · 타깃 게이트 결합 | `InterViewSelfAttention`, `TargetConditionedViewGate` (`wafer/ver6.py`) |
| `ver9.4` | `Line` 기본형: 평균 제거 + 8방향 리지 + `MaxPool` 보조 스트림 | `OrientedLineBranch` (`wafer/models.py`) |
| `ver9.5` | `Belief` 기본형: `layer3` 6×6 3-step 반복 공간 신념 추론 | `IterativeSpatialBeliefAttention` (`wafer/models.py`) |
| `ver9.6` | `Pool` 기본형: 0-초기화 공간 증거 게이트 풀링 | `SpatialEvidencePooling` (`wafer/models.py`) |
| `ver9.7A / 9.7B` | **3모듈(`Line+Belief+Pool`) 통합 기본형** (`ver7` / `SimCLR` 백본) | `USE_SA="three_base"` (`wafer/models.py`) |

자세한 동작 원리와 수치는 [`results/ver9_experiments.md`](./results/ver9_experiments.md)를 참고하세요.
