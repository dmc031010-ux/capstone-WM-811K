"""BN 형제 뷰 치팅 검증 미팅 그림 (기록된 결과 파일에서 숫자를 읽음)."""
import os, re, sys
import numpy as np
sp, ver7log, outd = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(outd, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
CT, CS, CO = "#eb6834", "#2a78d6", "#9a9992"     # together / split / 원래 ver7
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.facecolor": BG, "savefig.facecolor": BG,
                     "axes.facecolor": BG, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "font.size": 10.5})
read = lambda f: open(os.path.join(sp, f), encoding="utf-8").read()
loss = lambda t: [float(v) for v in re.findall(r"ssl epoch=\d+ loss=([\d.]+)", t)]
L_t, L_s = loss(read("cheat_train_together.txt")), loss(read("cheat_train_split.txt"))
L_o = loss(open(ver7log, encoding="utf-8", errors="ignore").read())[:5]
g = re.search(r"BN train: 자기 입력 ([\d.e+-]+) \| 형제 뷰 3장 평균 ([\d.e+-]+) \| 상관없는 웨이퍼 평균 ([\d.e+-]+)", read("cheat_grad.txt"))
own, sib, oth = map(float, g.groups())
ev = {m.group(1): (float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)))
      for m in re.finditer(r"(\w+)\s*로 학습한 모델: 같은 반 채점 ([\d.]+) \| 다른 반 채점 ([\d.]+) \| 차이\(다른-같은\) ([+-][\d.]+) ± ([\d.]+)", read("cheat_eval.txt"))}

# ---- 그림 1: 실험 설계
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#008300", "#e34948"]
for ax, mode, ttl in [(axes[0], "together", "together: 형제 4장이 같은 BN 묶음 (통로 열림)"),
                      (axes[1], "split", "split: 형제 4장이 서로 다른 BN 묶음 (통로 닫힘)")]:
    ax.set_xlim(-0.5, 9.5); ax.set_ylim(-0.6, 4.9); ax.axis("off")
    for gi in range(4):                                 # BN 묶음 4개, 묶음마다 8칸 (설명용 축소: 실제는 256장)
        ax.add_patch(Rectangle((-0.3, 3.6 - gi * 1.1 - 0.42), 9.4, 0.84, fc="none", ec=INK2, lw=1, ls=(0, (3, 2))))
        ax.text(-0.45, 3.6 - gi * 1.1, f"묶음 {gi + 1}", ha="right", va="center", fontsize=9, color=INK2)
        for c in range(8):
            if mode == "together":
                w, v = gi * 2 + c // 4, c % 4          # 묶음마다 웨이퍼 2장 × 마스킹본 4장
            else:
                w, v = c, gi                           # 묶음 = 모든 웨이퍼의 같은 번호 마스킹본
            ax.add_patch(Rectangle((c + 0.05, 3.6 - gi * 1.1 - 0.33), 0.9, 0.66, fc=pal[w], ec=BG, lw=1.5))
            ax.text(c + 0.5, 3.6 - gi * 1.1, f"W{w + 1}\n#{v + 1}", ha="center", va="center", fontsize=7.5, color="white")
    ax.set_title(ttl, loc="left", fontsize=11.5, color=CT if mode == "together" else CS, fontweight="bold")
fig.text(0.01, 0.01, "색 = 웨이퍼 (W1~W8), #1~#4 = 같은 웨이퍼의 마스킹본 번호. 설명을 위해 축소: 실제는 묶음 4개 × 256장 (together = 웨이퍼 64장 × 4장, split = 웨이퍼 256장의 같은 번호 마스킹본).\n"
         "두 조건 모두 묶음당 이미지 수 (256) 와 데이터·seed·마스크가 같고, 다른 것은 '형제가 같은 묶음에 있느냐' 하나뿐. 원래 ver7 은 1,024장 전체가 한 묶음 (형제 포함).",
         fontsize=9, color=INK2)
fig.suptitle("실험 설계: BatchNorm 통계를 함께 계산하는 묶음만 다르게", x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0.09, 1, 0.95)); fig.savefig(os.path.join(outd, "1_design.png"), dpi=150); plt.close(fig)

# ---- 그림 2: 기울기 검사
fig, ax = plt.subplots(figsize=(8, 4))
vals, names = [own, sib, oth], ["자기 입력", "형제 마스킹본\n(같은 웨이퍼)", "무관한 웨이퍼"]
bars = ax.bar(range(3), vals, color=[INK2, CT, CO], width=0.6, zorder=2)
for b_, v in zip(bars, vals):
    ax.text(b_.get_x() + b_.get_width() / 2, v * 1.15, f"{v:.2f}", ha="center", va="bottom", fontsize=10.5, color=INK)
ax.set_yscale("log"); ax.set_ylim(0.1, 400); ax.set_xticks(range(3), names)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_ylabel("출력의 입력 기울기 크기 (로그 축)"); ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
ax.set_title(f"① 통로가 있나: 형제 기울기 = 무관한 웨이퍼의 {sib / oth:.2f}배, 자기 입력의 {sib / own:.1%}\nBN 을 평가 모드로 바꾸면 형제·무관 모두 정확히 0 → 통로는 BN 하나뿐",
             loc="left", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(outd, "2_gradient.png"), dpi=150); plt.close(fig)

# ---- 그림 3: 학습 손실 5 epoch
fig, ax = plt.subplots(figsize=(8, 4.2))
e = np.arange(1, 6)
ax.plot(e, L_t, "-o", color=CT, lw=2, label="together (형제 같은 묶음, 256장)")
ax.plot(e, L_s, "-o", color=CS, lw=2, label="split (형제 다른 묶음, 256장)")
ax.plot(e, L_o, "--o", color=CO, lw=1.5, label="참고: 원래 ver7 (1,024장 한 묶음)")
for L, c in ((L_t, CT), (L_s, CS), (L_o, CO)):
    ax.text(5.08, L[-1], f"{L[-1]:.4f}", va="center", fontsize=9.5, color=c)
ax.set_xticks(e); ax.set_xlim(0.8, 5.6); ax.set_xlabel("사전학습 epoch (seed 0)"); ax.set_ylabel("일관성 손실 (낮을수록 잘 맞춤)")
ax.grid(color=GRID, lw=0.8); ax.legend(frameon=False)
ax.set_title("② 통로를 쓰나: 치팅이 가능하면 together 가 더 낮아야 하는데 → 오히려 더 높음", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(outd, "3_train_loss.png"), dpi=150); plt.close(fig)

# ---- 그림 4: 교차 평가
fig, ax = plt.subplots(figsize=(8, 4.2))
for j, (m, lab) in enumerate((("together", "together 로 학습한 모델"), ("split", "split 로 학습한 모델"))):
    t, s, d, se = ev[m]
    for k, (v, c, nm) in enumerate(((t, CT, "together 방식으로 채점 (통로 열림)"), (s, CS, "split 방식으로 채점 (통로 닫힘)"))):
        b_ = ax.bar(j + (k - 0.5) * 0.36, v, 0.34, color=c, label=nm if j == 0 else None, zorder=2)
        ax.text(b_[0].get_x() + 0.17, v + 0.0006, f"{v:.4f}", ha="center", va="bottom", fontsize=9.5)
ax.set_xticks([0, 1], ["together 로 학습한 모델", "split 로 학습한 모델"]); ax.set_ylabel("일관성 손실 (고정 배치 40개 평균)")
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.legend(frameon=False, loc="upper left")
t, s, d, se = ev["together"]
ax.set_title(f"③ 학습된 모델이 통로에 기대나: together 모델의 통로를 닫았더니 손실이 오히려 {abs(d):.4f}±{se:.4f} 감소\n→ 형제 통계는 도움이 아니라 잡음", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(outd, "4_cross_eval.png"), dpi=150); plt.close(fig)

print("손실 together", L_t, "\nsplit", L_s, "\nver7", L_o, "\n기울기", own, sib, oth, "\n교차", ev)
print("저장:", outd)
