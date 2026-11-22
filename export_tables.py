"""미팅 자료용 결과표 추출: runs/ 의 metrics.json → results/*.csv (엑셀용 UTF-8 BOM).

  python export_tables.py
"""
import csv
import json
import os

import numpy as np

BASE = os.path.join("runs", "resnet18", "p0.05", "seed0")
MINOR = ["Near-full", "Donut", "Random", "Scratch", "Loc"]
CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Random", "Scratch", "Near-full", "none"]
# (표시 이름, 결과 폴더, 총 학습 시간[h] — 노트북 실측, 로그 기준)
ROWS = [("Supervised (우리)", "supervised", 0.3),
        ("WaFixMatch (우리 재현)", "wafixmatch", 3.3),
        ("ver1", "ver1_m0-0.5", 2.0),
        ("ver1.1", "ver1.1_m0-0.5_L0-0.05", 2.2),
        ("ver2", "ver2", 2.5),
        ("ver3.2", "ver3.2_ssl50_ft30", 2.4)]
PAPER_WAFIXMATCH = 0.841
REF = "wafixmatch"


def load(run):
    with open(os.path.join(BASE, run, "metrics.json"), encoding="utf-8") as f:
        return json.load(f)["test"]


def write(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("saved", path)


def main():
    os.makedirs("results", exist_ok=True)
    T = {run: load(run) for _, run, _ in ROWS}
    ref = T[REF]

    main_rows = [["WaFixMatch (논문 보고값)", f"{PAPER_WAFIXMATCH:.3f}", "", "", "", ""]]
    for name, run, hours in ROWS:
        t = T[run]
        minor = np.mean([t["per_class_f1"][c] for c in MINOR])
        if run in ("supervised", REF):
            verdict = "기준" if run == REF else "참고"
        else:
            drops = {c: t["per_class_f1"][c] - ref["per_class_f1"][c] for c in MINOR}
            bad = {c: d for c, d in drops.items() if d < -0.03}
            ok = t["macro_f1"] > ref["macro_f1"] and not bad
            worst = min(drops, key=drops.get)
            verdict = "통과" if ok else "탈락 (" + ", ".join(f"{c} {d:+.3f}" for c, d in sorted(bad.items(), key=lambda x: x[1])) + \
                      (f"macro {t['macro_f1'] - ref['macro_f1']:+.3f}" if t["macro_f1"] <= ref["macro_f1"] and not bad else "") + ")"
            if ok:
                verdict += f" (최대 하락 {worst} {drops[worst]:+.3f})"
        main_rows.append([name, f"{t['macro_f1']:.3f}", f"{minor:.3f}", f"{t['acc']:.3f}", verdict, f"약 {hours}"])
    write(os.path.join("results", "table1_main.csv"),
          ["Method", "macro-F1 (5% 라벨)", "소수 클래스 평균 F1", "Accuracy", "성공 기준 (WaFixMatch 재현 대비)", "총 학습 시간 (h)"],
          main_rows)

    write(os.path.join("results", "table2_per_class_f1.csv"),
          ["클래스"] + [name for name, _, _ in ROWS],
          [[c] + [f"{T[run]['per_class_f1'][c]:.3f}" for _, run, _ in ROWS] for c in CLASSES])

    notes = [["조건", "WM-811K, ResNet-18 (96x96), 5% 라벨 6,919장 + unlabeled 638,507장, seed 0"],
             ["학습", "AdamW lr 0.01 고정 (ver3.2 미세조정 lr 1e-3 고정, ver3.2 자기지도 단계만 cosine)"],
             ["증강", "ver1/ver1.1/ver2/ver3.2 는 cutout 없음, WaFixMatch 는 saliency 기반 cutout"],
             ["평가", "val macro-F1 최고 epoch 의 test 값"],
             ["성공 기준", "macro-F1 > WaFixMatch 재현, 소수 클래스(Near-full/Donut/Random/Scratch/Loc) 각각 -0.03 이내"],
             ["학습 시간", "RTX 3050 노트북 4GB 실측, 필요한 사전 단계 포함, 발열에 따라 약 ±15%"],
             ["주의", "단일 seed 결과. 학습률 고정 시 val F1 의 epoch 간 흔들림이 큼 (±0.13~0.15)"]]
    write(os.path.join("results", "table_notes.csv"), ["항목", "내용"], notes)


if __name__ == "__main__":
    main()
