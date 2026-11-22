"""#13 (idx 642918) 등: ResNet 끝 = 평균 풀링 + 선형 → 로짓을 3x3 칸별 기여 + bias 로 '정확히' 분해 (CAM, 음수 포함).
logit_k = b_k + (1/9) Σ_칸 w_k · a_칸"""
import os, sys
import numpy as np, torch
proj = sys.argv[1]; idxs = [int(v) for v in sys.argv[2].split(",")]
os.chdir(proj); sys.path.insert(0, proj)
from wafer.data import WM811K, CLASSES, NUM_CLASSES
from wafer.models import build_model, to_input

ds = WM811K(os.path.expanduser("~/wafer_cache"))
m = build_model("resnet18", NUM_CLASSES)
m.load_state_dict(torch.load("runs/resnet18/p0.05/seed0/ver8_ssl50_ft30/best.pt", map_location="cpu", weights_only=True))
m.eval()
feat = {}
m.layer4.register_forward_hook(lambda mod, i, o: feat.__setitem__("a", o))
SC, NO = CLASSES.index("Scratch"), CLASSES.index("none")
W, b = m.fc.weight.detach(), m.fc.bias.detach()
print("분류기 bias:", {c: round(float(v), 2) for c, v in zip(CLASSES, b)})
for i in idxs:
    x = ds.fetch([i], "cpu")
    with torch.no_grad():
        logit = m(to_input(x))[0]
    a = feat["a"][0]                                   # (512, 3, 3)
    contrib = torch.einsum("kc,chw->khw", W, a) / (a.shape[1] * a.shape[2])   # (9, 3, 3)
    assert torch.allclose(contrib.sum((1, 2)) + b, logit, atol=1e-3)
    print(f"\n=== idx {i} 정답 {CLASSES[ds.labels[i]]} · 예측 {CLASSES[logit.argmax()]} · 확률 none {torch.softmax(logit, 0)[NO]:.3f} Scratch {torch.softmax(logit, 0)[SC]:.4f}")
    for k in (NO, SC):
        c = contrib[k]
        print(f"[{CLASSES[k]}] 로짓 {logit[k]:.2f} = bias {b[k]:.2f} + 양(+) 기여 {c.clamp(min=0).sum():.2f} + 음(-) 기여 {c.clamp(max=0).sum():.2f}")
        print("   칸별 기여 (3x3, 위→아래):\n" + "\n".join("     " + "  ".join(f"{v:+6.2f}" for v in row) for row in c.numpy()))
