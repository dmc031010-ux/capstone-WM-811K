"""Codex 에게 계획 / 결과 비판을 받는다 (읽기 전용).

python debate/ask_codex.py <버전> plan     01_proposal.md → 02_codex_critique.md
python debate/ask_codex.py <버전> result   05_result.md (+ 01~04) → 06_codex_result_review.md
python debate/ask_codex.py _smoke test     동작 확인용 (짧은 질문)
"""
import glob, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
topic, mode = sys.argv[1], sys.argv[2]
d = os.path.join(ROOT, "debate", topic)


def codex_bin():
    c = sorted(glob.glob(os.path.expanduser(r"~\.vscode\extensions\openai.chatgpt-*\bin\windows-x86_64\codex.exe")), key=os.path.getmtime)
    if not c:
        sys.exit("codex.exe 를 찾지 못함 (VS Code 의 OpenAI ChatGPT/Codex 확장 필요)")
    return c[-1]


def read(name):
    p = os.path.join(d, name)
    return f"\n\n===== {name} =====\n" + open(p, encoding="utf-8").read() if os.path.exists(p) else ""


if mode == "plan":
    src, out = read("01_proposal.md"), os.path.join(d, "02_codex_critique.md")
    assert src, f"{d}\\01_proposal.md 가 없음"
    task = ("AGENTS.md 의 '계획 검토 (plan)' 역할과 출력 형식에 따라, 아래 실험 계획을 비판적으로 검토하라. "
            "계획이 말하는 코드·결과 파일을 직접 열어 근거를 확인하라. 파일은 수정하지 마라.")
elif mode == "result":
    src = "".join(read(f) for f in ("01_proposal.md", "02_codex_critique.md", "03_response.md", "04_decision.md", "05_result.md"))
    out = os.path.join(d, "06_codex_result_review.md")
    assert "05_result.md" in src, f"{d}\\05_result.md 가 없음"
    task = ("AGENTS.md 의 '결과 검토 (result)' 역할과 출력 형식에 따라, 아래 05_result.md 의 결과와 해석을 비판적으로 검토하라. "
            "앞 문서들은 맥락이다. metrics.json, 로그 등 결과 파일을 직접 열어 수치를 확인하라. 파일은 수정하지 마라.")
elif mode == "test":
    src, out = "", os.path.join(ROOT, "debate", "_smoke_test.md")
    task = "AGENTS.md 를 읽고, 이 저장소에서 너의 역할을 한국어 두 문장으로 요약하라. 파일은 수정하지 마라."
else:
    sys.exit("mode 는 plan / result / test")

os.makedirs(os.path.dirname(out), exist_ok=True)
prompt = task + src
cmd = [codex_bin(), "exec", "-s", "read-only", "-C", ROOT, "-o", out, "-"]
t = time.time()
r = subprocess.run(cmd, input=prompt.encode("utf-8"), capture_output=True)
log = os.path.splitext(out)[0] + ".log"
open(log, "wb").write(r.stdout + b"\n---- stderr ----\n" + r.stderr)
if r.returncode != 0 or not os.path.exists(out):
    sys.exit(f"codex 실패 (코드 {r.returncode}). 로그: {log}")
print(f"완료 {time.time() - t:.0f}초 → {out}")
