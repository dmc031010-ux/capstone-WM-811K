# 04. 사용자 결정 (2026-11-23)

- **진행.** 03_response.md 의 수정 계획대로.
- **projector 추가** (사용자 제안): 인코더 512 → projector (512→512→128, BN+ReLU) → predictor (128→512→128). 타깃은 인코더+projector 를 EMA 로. 사전학습 후 projector·predictor 는 버리고 인코더만 fine-tune.
- 원인 분리를 위해 파일럿 조건 변경 (seed 0, SSL 10 epoch 씩):
  ① 마스킹, projector 없음 (= y9 방식, 기준) ② 마스킹 + projector ③ 회전만 + projector ④ 노이즈 혼합 (2/4/8/16%) + projector.
  참고선: 기존 ver3 SimCLR seed 0 인코더 (학습 없음). "노이즈 고정 4%" 조건은 제외.
- **0단계부터** (기존 y9 모델로 잡음 민감도 점검, val 만 사용). 0단계 결과를 보고 1단계 진행 여부를 다시 확인.
