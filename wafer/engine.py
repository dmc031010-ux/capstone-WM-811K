"""학습 루프 (supervised = saliency 모델 = Supervised 베이스라인, WaFixMatch SSL), 평가, saliency 계산."""
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score

from . import augment as A
from .data import CLASSES, NUM_CLASSES

NONE = CLASSES.index("none")
from .models import SizeHead, build_model, to_input


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BalancedSampler:
    """WaPIRL 식 batch-level resampling: 클래스 빈도의 역수로 가중 복원추출.

    power: 클래스별 뽑힐 비율 ∝ 장수^power (0 = 완전 균형, 1 = 원래 비율 (ver7.9), 0.5 = 제곱근 비례 (ver7.91))
    """

    def __init__(self, labels, seed, power=0.0):
        cnt = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
        w = cnt[labels] ** (power - 1.0)
        self.p = w / w.sum()
        self.rng = np.random.default_rng(seed)

    def __call__(self, n):
        return self.rng.choice(len(self.p), size=n, replace=True, p=self.p)


@torch.no_grad()
def evaluate(model, ds, idx, device, bs=1024):
    model.eval()
    preds = []
    for i in range(0, len(idx), bs):
        x = to_input(ds.fetch(idx[i:i + bs], device))
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x, torch.as_tensor(idx[i:i + bs], device=device)) if getattr(model, "needs_idx", False) else model(x)
            preds.append(out.argmax(1).cpu())
    y_pred = torch.cat(preds).numpy()
    y_true = ds.labels[idx]
    labels = list(range(NUM_CLASSES))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp = np.diag(cm)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(NUM_CLASSES, dtype=float), where=cm.sum(0) != 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(NUM_CLASSES, dtype=float), where=cm.sum(1) != 0)
    return dict(
        macro_f1=float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        acc=float((y_true == y_pred).mean()),
        per_class_f1=dict(zip(CLASSES, f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0).round(4).tolist())),
        per_class_precision=dict(zip(CLASSES, precision.round(4).tolist())),
        per_class_recall=dict(zip(CLASSES, recall.round(4).tolist())),
        confusion=cm.tolist(),
    )


class _Tracker:
    """매 epoch val macro-F1 평가, 최고 시점의 test 성능 기록 + 체크포인트 저장."""

    def __init__(self, out_dir, model, ds, device, args, extra):
        self.out_dir, self.model, self.ds, self.device, self.args = out_dir, model, ds, device, args
        self.best, self.history, self.extra = -1.0, [], extra

    def step(self, epoch, train_log):
        val = evaluate(self.model, self.ds, self.ds.val, self.device)
        rec = dict(epoch=epoch, val_macro_f1=val["macro_f1"],
                   val_scratch_precision=val["per_class_precision"]["Scratch"],
                   val_scratch_recall=val["per_class_recall"]["Scratch"], **train_log)
        if val["macro_f1"] > self.best:
            self.best = val["macro_f1"]
            torch.save(self.model.state_dict(), os.path.join(self.out_dir, "best.pt"))
            test = evaluate(self.model, self.ds, self.ds.test, self.device)
            self.result = dict(best_epoch=epoch, val=val, test=test)
            rec["test_macro_f1"] = test["macro_f1"]
        self.history.append(rec)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items())
              + ("  *" if "test_macro_f1" in rec else ""), flush=True)
        with open(os.path.join(self.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(dict(args=vars(self.args), **self.extra, **self.result, history=self.history),
                      f, indent=2, ensure_ascii=False)


def _setup(ds, lab_idx, args, device, epochs):
    set_seed(args.seed)
    model = build_model(args.arch, NUM_CLASSES).to(device)  # channels_last 는 1채널 입력에서 ~11배 느려 사용 안 함
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = math.ceil(len(lab_idx) / args.batch_size)  # 1 epoch = 라벨 데이터 한 바퀴
    sched = None
    if args.cosine:  # FixMatch 원 논문의 lr * cos(7*pi*k / (16*K))
        total = steps * epochs
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: math.cos(7 * math.pi * k / (16 * total)))
    ema = None
    if args.ema:  # FixMatch 원 논문: 가중치(+BN 통계) 이동평균 모델로 평가
        ema = torch.optim.swa_utils.AveragedModel(
            model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema), use_buffers=True)
    return model, opt, torch.amp.GradScaler(), BalancedSampler(ds.labels[lab_idx], args.seed), steps, sched, ema


def _update(opt, scaler, sched, ema, model, loss):
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    if sched:
        sched.step()
    if ema:
        ema.update_parameters(model)


def train_supervised(ds, lab_idx, args, out_dir, device):
    """Stage 1 (식 1): 라벨 데이터 + weak aug(회전) 로 f_theta1 학습. 이 결과가 Supervised 베이스라인."""
    os.makedirs(out_dir, exist_ok=True)
    model, opt, scaler, sampler, steps, sched, ema = _setup(ds, lab_idx, args, device, args.pretrain_epochs)
    tracker = _Tracker(out_dir, ema.module if ema else model, ds, device, args,
                       dict(n_labeled=len(lab_idx), stage="supervised"))
    for ep in range(1, args.pretrain_epochs + 1):
        model.train()                                                                           
        t, tot = time.time(), 0.0
        for _ in range(steps):
            idx = lab_idx[sampler(args.batch_size)]
            y = torch.from_numpy(ds.labels[idx]).to(device)
            with torch.no_grad():
                x = to_input(A.weak(ds.fetch(idx, device), "rotate"))
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(model(x), y)
            _update(opt, scaler, sched, ema, model, loss)
            tot += loss.item()
        tracker.step(ep, dict(loss=tot / steps, sec=round(time.time() - t, 1)))
    return tracker.result


def compute_saliency(model, ds, idx, device, bs=512, down=4):
    """|d l_y(x) / dx| (Simonyan et al.), y = 예측 클래스 (unlabeled 이므로).

    단일 채널이라 채널 max 불필요. 저장 용량 때문에 down x down 평균 풀링 후
    이미지별 max 정규화 → uint8. (96 → 24x24, 63만장 ≈ 370MB)
    """
    model.eval()
    s = ds.size // down
    out = np.empty((len(idx), s, s), np.uint8)
    t = time.time()
    for i in range(0, len(idx), bs):
        x = to_input(ds.fetch(idx[i:i + bs], device)).requires_grad_(True)
        logits = model(x)  # fp32: 입력 그래디언트 underflow 방지
        g, = torch.autograd.grad(logits.max(1).values.sum(), x)
        g = F.avg_pool2d(g.abs(), down)
        g = g / g.amax((2, 3), keepdim=True).clamp_min(1e-12)
        out[i:i + bs] = (g[:, 0] * 255).round().byte().cpu().numpy()
        if (i // bs) % 200 == 0:
            print(f"  saliency {i + len(x)}/{len(idx)}  {time.time() - t:.0f}s", flush=True)
    return out


def lambda_s(mask_prev, m0):
    """ver1: 직전 epoch 평균 채택률 m 이 m0 를 넘으면 Ls 가중치를 (1-m)/(1-m0) 로 선형 감쇠."""
    if m0 is None or mask_prev < m0:
        return 1.0
    return (1.0 - mask_prev) / (1.0 - m0)


def train_wafixmatch(ds, lab_idx, sal, args, out_dir, device, pat=None):
    """Stage 2: L = lambda_s * Ls + lambda_u * Lu (식 2~4, lambda_s 는 --switch_mask 없으면 1).

    sal: ds.unlabeled 순서의 saliency (N_u, h', w') uint8. args.keep=False 면 None.
    pat: ver2 패턴 통로 dict(q_arg, q_max: ds.unlabeled 순서, tau_p). 결함 클래스만,
         확신도 < tau_s 라도 패턴 1등 == 분류기 예측 & q_max >= tau_p 면 추가 채택.
    strong: wafixmatch = 웨이퍼 5종(KeepCutout), fixmatch = 기하 RandAugment, ver1/ver1.1/ver2 = 웨이퍼 5종(일반 cutout, --no_cutout 이면 4종).
    """
    os.makedirs(out_dir, exist_ok=True)
    model, opt, scaler, sampler, steps, sched, ema = _setup(ds, lab_idx, args, device, args.epochs)
    B, uB = args.batch_size, args.mu * args.batch_size
    urng = np.random.default_rng(args.seed + 1)
    n_u = len(ds.unlabeled)
    # pseudo-label 은 원 FixMatch 처럼 학습 중인 model 로 만들고, 평가/저장만 EMA 모델
    tracker = _Tracker(out_dir, ema.module if ema else model, ds, device, args,
                       dict(n_labeled=len(lab_idx), n_unlabeled=n_u, stage="wafixmatch", aug_cfg=A.AUG_CFG))
    mask_prev, ls_prev, lam_s, started = 0.0, float("inf"), 1.0, False
    for ep in range(1, args.epochs + 1):
        model.train()
        t = time.time()
        if args.method == "ver1.1":
            # ver1.1: labeled 를 충분히 배운 뒤(Ls <= L0)에만 감쇠 시작, 이후 lambda_s 는 단조 감소
            started = started or (ls_prev <= args.ls_start and mask_prev >= args.switch_mask)
            if started:
                lam_s = min(lam_s, lambda_s(mask_prev, args.switch_mask))
        else:
            lam_s = lambda_s(mask_prev, args.switch_mask)
        acc = dict(loss=0.0, Ls=0.0, Lu=0.0, mask=0.0)
        for _ in range(steps):
            li = lab_idx[sampler(B)]
            ui = urng.integers(0, n_u, uB)
            yl = torch.from_numpy(ds.labels[li]).to(device)
            with torch.no_grad():
                xl = A.weak(ds.fetch(li, device), args.weak)
                xu = ds.fetch(ds.unlabeled[ui], device)
                su = torch.from_numpy(sal[ui]).to(device).unsqueeze(1).float() if args.keep else None
                xw = A.weak(xu, args.weak)
                if args.method == "fixmatch":
                    xs = A.randaugment_geo(xu)
                elif args.method in ("ver1", "ver1.1", "ver2"):
                    # 웨이퍼 5종 (cutout 은 saliency 없이 무작위), --no_cutout 이면 cutout 제외 4종
                    xs = A.strong(xu, None, args.tau_c, keep=False,
                                  ops=A.STRONG_OPS_NO_CUTOUT if args.no_cutout else A.STRONG_OPS)
                else:
                    xs = A.strong(xu, su, args.tau_c, args.keep)
                # 4GB VRAM: pseudo-label 용 weak 분기는 gradient 가 없으므로 별도 no_grad forward
                # (원 FixMatch 는 한 번에 concat forward → 차이는 BN 배치 통계뿐)
                with torch.autocast("cuda", dtype=torch.float16):
                    lw = model(to_input(xw)).float()
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(to_input(torch.cat([xl, xs]))).float()
            ll, ls = logits[:B], logits[B:]
            Ls = F.cross_entropy(ll, yl)
            conf, pl = torch.softmax(lw, 1).max(1)
            mask = conf >= args.tau_s
            if pat is not None:
                qa = torch.from_numpy(pat["q_arg"][ui]).to(device).long()
                qm = torch.from_numpy(pat["q_max"][ui]).to(device).float()
                route = (~mask) & (qa == pl) & (qa != NONE) & (qm >= pat["tau_p"][qa])
                acc["mask_pat"] = acc.get("mask_pat", 0.0) + route.float().mean().item()
                mask = mask | route
            mask = mask.float()
            Lu = (F.cross_entropy(ls, pl, reduction="none") * mask).mean()
            loss = lam_s * Ls + args.lambda_u * Lu
            _update(opt, scaler, sched, ema, model, loss)
            acc["loss"] += loss.item(); acc["Ls"] += Ls.item(); acc["Lu"] += Lu.item(); acc["mask"] += mask.mean().item()
        log = {k: v / steps for k, v in acc.items()}
        mask_prev, ls_prev = log["mask"], log["Ls"]
        log["lambda_s"] = lam_s
        log["sec"] = round(time.time() - t, 1)
        tracker.step(ep, log)
    return tracker.result
