# 실험 토론 구조 (Claude 실행 ↔ Codex 비판 ↔ 사용자 결정)

실험 하나당 폴더 하나: `debate/<버전>/` (예: `debate/ver9.1/`). 토론은 1라운드, 결과도 검토.

| 순서 | 파일 | 작성 | 내용 |
|---|---|---|---|
| 1 | `01_proposal.md` | Claude | 목표, 가설, 근거, 설계 (바꿀 코드), 대조군, 판단 기준, 비용 |
| 2 | `02_codex_critique.md` | Codex | 판정 + 지적 C1, C2 … (AGENTS.md 의 형식) |
| 3 | `03_response.md` | Claude | 지적마다 수용 / 일부 수용 / 반박 (근거) + 수정한 최종 계획 |
| 4 | `04_decision.md` | 사용자 | 진행 / 수정 / 중단. 남은 쟁점에 대한 결정 |
| 5 | (실행) | Claude | 결정대로 코드 수정·학습 |
| 6 | `05_result.md` | Claude | 결과 수치, 해석, 계획 대비 달라진 점 |
| 7 | `06_codex_result_review.md` | Codex | 해석 판정 + 지적 R1, R2 … |
| 8 | `07_final.md` | Claude | 지적별 응답 + 최종 결론 (미팅 자료로 쓸 문장) |

## Codex 실행
```
python debate/ask_codex.py <버전> plan     # 01_proposal.md → 02_codex_critique.md
python debate/ask_codex.py <버전> result   # 05_result.md (+ 앞 문서들) → 06_codex_result_review.md
```
- `codex exec` 를 **읽기 전용 샌드박스** 로 실행한다. Codex 는 저장소를 읽을 수 있지만 고치지 못한다.
- 역할과 출력 형식은 저장소 루트의 `AGENTS.md` 에 있다 (Codex 가 자동으로 읽음).
- 실행하면 사용자의 ChatGPT / Codex 사용량이 소모된다.
