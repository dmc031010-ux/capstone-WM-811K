"""WaFixMatch (Jo & Kim, IEEE Access 2025) 재현 - WM-811K.

  python wafixmatch.py prepare                       # LSWMD.pkl → 캐시 (1회)
  python wafixmatch.py run --frac 0.05 --seed 0      # stage1(supervised) + saliency + stage2(WaFixMatch)
  python wafixmatch.py run --frac 0.05 0.1 0.25 0.5 1.0 --seed 0 1 2 3 4
  python wafixmatch.py run --frac 0.05 --no_keep               # Table 4: w/o KeepCutout
  python wafixmatch.py run --frac 0.05 --weak flipshift        # Table 4: w/o Rotation
  python wafixmatch.py run --frac 0.05 --arch densenet121      # Table 5 (resnet50: Table 6)
  python wafixmatch.py run --frac 0.05 --method fixmatch                     # 표준 FixMatch 베이스라인
  python wafixmatch.py run --frac 0.05 --method ver1                         # ver1 기반 (감쇠 없음)
  python wafixmatch.py run --frac 0.05 --method ver1 --switch_mask 0.5       # 제안 ver1 (Ls 감쇠)
  python wafixmatch.py run --frac 0.05 --method ver1.1 --switch_mask 0.5     # ver1.1 (Ls<=0.05 후 시작, 단조 감소)
  python wafixmatch.py run --frac 0.05 --method ver2 --cosine                # ver2 (패턴 사전 + 패턴 통로)
  python wafixmatch.py run --frac 0.05 --method ver3                         # ver3 기준선 (일반 자기지도 학습)
  python wafixmatch.py run --frac 0.05 --method ver3.1 --v3_weight purity    # ver3.1 (필터 + 군집 + cutout + 발전 옵션)
  python wafixmatch.py run --frac 0.05 --method ver3.2                       # ver3.2 (SimCLR → 군집 → 미세조정 처음 버전, cutout 없음)
  python wafixmatch.py run --frac 0.05 --method ver6                         # ver6 (잠재 벡터 멀티 마스킹 consistency + ver3 미세조정)
  python wafixmatch.py run --frac 0.05 --method ver6.1                       # ver6.1 (바둑판 50% + 서로 다른 회전 + MLP Predictor)
  python wafixmatch.py run --frac 0.05 --method ver6.2                       # ver6.2 (ver6.1 에서 바둑판 칸 12 → 6px)
  python wafixmatch.py run --frac 0.05 --method ver6.3                       # ver6.3 (ver6.2 에서 마스킹본 4 → 3 장)
  python wafixmatch.py run --frac 0.05 --method ver6.4                       # ver6.4 (ver6.2 에서 가리는 비율 50% → 30%)
  python wafixmatch.py run --frac 0.05 --method ver6.5                       # ver6.5 (ver6.2 에서 바둑판 칸 6 → 4px)
  python wafixmatch.py run --frac 0.05 --method ver6.6                       # ver6.6 (ver6.2 에서 Predictor MLP → 선형)
  python wafixmatch.py run --frac 0.05 --method ver7                         # ver7 (= ver6.2, 7주차 출발점)
  python wafixmatch.py run --frac 0.05 --method ver7.1                       # ver7.1 (ver7 에서 백본 ResNet-34)
  python wafixmatch.py run --frac 0.05 --method ver7.2                       # ver7.2 (ver7 에서 손실을 마스킹본별 cos 손실 평균으로)
  python wafixmatch.py run --frac 0.05 --method ver7.3                       # ver7.3 (ver7 에서 입력 2채널: 원본 + 불량 3x3 팽창)
  python wafixmatch.py run --frac 0.05 --method ver7.4                       # ver7.4 (ver7 인코더 + 미세조정 Triplet)
  python wafixmatch.py run --frac 0.05 --method ver7.5                       # ver7.5 (ver7 인코더 + 미세조정 Focal)
  python wafixmatch.py run --frac 0.05 --method ver7.6                       # ver7.6 (ver7.4 + ver7.5: Triplet + Focal)
  python wafixmatch.py run --frac 0.05 --method ver7.7                       # ver7.7 (ver7 에서 마스킹 = 바둑판 2 + 세로줄 1 + 가로줄 1)
  python wafixmatch.py run --frac 0.05 --method ver7.8                       # ver7.8 (ver7 인코더 + CE 의 none 가중치 0.5)
  python wafixmatch.py run --frac 0.05 --method ver7.81                      # ver7.81 (ver7 인코더 + CE 의 none 가중치 0.1)
  python wafixmatch.py run --frac 0.05 --method ver7.82                      # ver7.82 (ver7.8 + Scratch 가중치 2.0)
  python wafixmatch.py run --frac 0.05 --method ver7.83                      # ver7.83 (ver7.8 + Scratch 가중치 5.0)
  python wafixmatch.py run --frac 0.05 --method ver7.9                       # ver7.9 (ver7 인코더 + 균형 샘플링 끔)
  python wafixmatch.py run --frac 0.05 --method ver7.91                      # ver7.91 (ver7 인코더 + 장수 제곱근 비례 샘플링)
  python wafixmatch.py run --frac 0.05 --method ver7.92                      # ver7.92 (ver7.91 + ver7.7 줄무늬 인코더)
  python wafixmatch.py run --frac 0.05 --method ver7.93                      # ver7.93 (ver7.91 + Triplet)
  python wafixmatch.py run --frac 0.05 --method ver7.X1                      # ver7.X1 (사전학습: 줄무늬 25 → 바둑판 25)
  python wafixmatch.py run --frac 0.05 --method ver7.95                      # ver7.95 (사전학습: 바둑판 25 → 줄무늬 25)
  python wafixmatch.py run --frac 0.05 --method ver7.X2                      # ver7.X2 (사전학습 + 라벨 Triplet 0.1)
  python wafixmatch.py run --frac 0.05 --method ver7.X                       # ver7.X (마스킹본만 무작위 크기 자르기 후 마스킹)
  python wafixmatch.py summary                      # runs/ 전체 test macro-F1 표 (mean(std))
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from wafer import models as wmodels, patterns, selfsup, ver6
from wafer.data import NUM_CLASSES, WM811K, prepare
from wafer.engine import compute_saliency, train_supervised, train_wafixmatch
from wafer.models import DEFAULT_BATCH, build_model


def stab_tag(args):
    """학습 안정화 설정 접미사 (없으면 빈 문자열 → 기존 결과 폴더 그대로)."""
    return ("_cos" if args.cosine else "") + (f"_ema{args.ema:g}" if args.ema else "") + ("_imagenet" if args.imagenet else "")


def cut_tag(args):
    """ver 계열: 기본(일반 cutout 포함)은 '_cut', --no_cutout 은 접미사 없음 (cutout 없이 돌린 기존 결과 폴더와 동일)."""
    return "" if args.no_cutout else "_cut"


def ssl_tag(args):
    if args.method == "ver1":
        tag = f"ver1{cut_tag(args)}_base" if args.switch_mask is None else f"ver1{cut_tag(args)}_m0-{args.switch_mask:g}"
    elif args.method == "ver1.1":
        tag = f"ver1.1{cut_tag(args)}_m0-{args.switch_mask:g}_L0-{args.ls_start:g}"
    elif args.method == "fixmatch":
        tag = "fixmatch"
    elif args.method == "ver2":
        tag = f"ver2{cut_tag(args)}"
    elif args.method == "ver3":
        tag = f"ver3_ssl{args.ssl_epochs}_ft{args.ft_epochs}"
    elif args.method in ("ver3.2", "ver6", "ver6.1", "ver6.2", "ver6.3", "ver6.4", "ver6.5", "ver6.6", "ver7", "ver7.1", "ver7.2", "ver7.3", "ver7.4", "ver7.5", "ver7.6", "ver7.7", "ver7.8", "ver7.81", "ver7.82", "ver7.83", "ver7.9", "ver7.91", "ver7.92", "ver7.93", "ver7.X1", "ver7.95", "ver7.X2", "ver7.X", "ver8", "ver8.1", "ver8.2", "ver8.3", "y9", "y9.1a", "y9.1b", "y9.1c", "y9.1d"):
        tag = f"{args.method}_ssl{args.ssl_epochs}_ft{args.ft_epochs}"
    elif args.method == "ver3.1":
        # ver3.1 → ver3_cut_..._pw 등 (기존 결과 폴더 이름 유지)
        tag = f"ver3{cut_tag(args)}{'_mc' if args.v3_multicrop else ''}_ssl{args.ssl_epochs}_ft{args.ft_epochs}"
        tag += ("_pw" if args.v3_weight == "purity" else "") + ("_nopl" if args.v3_nopl else "") + \
               (f"_k{args.v3_clusters}" if args.v3_clusters != 100 else "")
    else:
        tag = "wafixmatch"
        if args.weak != "rotate":
            tag += "_woRotation"
        if not args.keep:
            tag += "_woKeepCutout"
    return tag + stab_tag(args)


def run_one(args, ds, device):
    base = os.path.join(args.runs, args.arch, f"p{args.frac:g}", f"seed{args.seed}")
    lab_idx = ds.labeled_subset(args.frac, seed=args.split_seed)
    print(f"\n===== {args.arch} frac={args.frac:g} seed={args.seed}  labeled={len(lab_idx)} unlabeled={len(ds.unlabeled)} =====")

    # Stage 1: 라벨 데이터만으로 supervised (= saliency 모델 f_theta1 = Supervised 베이스라인)
    sup_dir = os.path.join(base, "supervised" + stab_tag(args))
    sup_ckpt = os.path.join(sup_dir, "best.pt")
    if os.path.exists(os.path.join(sup_dir, "done")):
        print(f"--- stage 1: skip (done {sup_dir})")
    elif args.method.startswith(("ver3", "ver6", "ver7", "ver8", "y9")):
        print("--- stage 1: skip (자기지도 계열은 supervised 모델을 쓰지 않음)")
    else:
        print("--- stage 1: supervised (saliency model) ---")
        train_supervised(ds, lab_idx, args, sup_dir, device)
        open(os.path.join(sup_dir, "done"), "w").close()
        torch.cuda.empty_cache()  # stage1 캐시 조각이 남으면 stage2 가 4GB 를 넘어 공유 메모리로 넘침
    if args.stage1_only:
        return

    # Saliency map acquisition: 고정된 f_theta1 로 unlabeled 전체 (캐시 폴더에 저장)
    sal = None
    if args.keep:
        sal_path = os.path.join(args.cache, "saliency", os.path.relpath(base), f"saliency_unlabeled{stab_tag(args)}.npy")
        if not os.path.exists(sal_path):
            print("--- saliency map acquisition ---")
            os.makedirs(os.path.dirname(sal_path), exist_ok=True)
            m = build_model(args.arch, NUM_CLASSES).to(device)
            m.load_state_dict(torch.load(sup_ckpt, map_location=device, weights_only=True))
            np.save(sal_path, compute_saliency(m, ds, ds.unlabeled, device))
            del m
            torch.cuda.empty_cache()
        sal = np.load(sal_path)

    out = os.path.join(base, ssl_tag(args))
    if os.path.exists(os.path.join(out, "done")):
        print(f"--- stage 2: skip (done {out})")
        return
    # ver3.1 / ver3.2 / ver6 는 ver3 의 캐시 폴더를 공유 (파일 이름으로 구분)
    shared = ("ver3.1", "ver3.2", "ver6", "ver6.1", "ver6.2", "ver6.3", "ver6.4", "ver6.5", "ver6.6", "ver7", "ver7.1", "ver7.2", "ver7.3", "ver7.4", "ver7.5", "ver7.6", "ver7.7", "ver7.8", "ver7.81", "ver7.82", "ver7.83", "ver7.9", "ver7.91", "ver7.92", "ver7.93", "ver7.X1", "ver7.95", "ver7.X2", "ver7.X", "ver8", "ver8.1", "ver8.2", "ver8.3", "y9", "y9.1a", "y9.1b", "y9.1c", "y9.1d")
    cache_base = os.path.join(args.cache, "ver3" if args.method in shared else args.method, os.path.relpath(base))
    os.makedirs(cache_base, exist_ok=True)

    if args.method in ("ver3", "ver3.1", "ver3.2", "ver6", "ver6.1", "ver6.2", "ver6.3", "ver6.4", "ver6.5", "ver6.6", "ver7", "ver7.1", "ver7.2", "ver7.3", "ver7.4", "ver7.5", "ver7.6", "ver7.7", "ver7.8", "ver7.81", "ver7.82", "ver7.83", "ver7.9", "ver7.91", "ver7.92", "ver7.93", "ver7.X1", "ver7.95", "ver7.X2", "ver7.X", "ver8", "ver8.1", "ver8.2", "ver8.3", "y9", "y9.1a", "y9.1b", "y9.1c", "y9.1d"):
        # ver3 = 기준선, ver6/6.1 = ver3 와 같은 데이터: 불량 비율 필터 없이 unlabeled 전체에서 무작위 (r0 = 0)
        random_sub = args.method in ("ver3", "ver6", "ver6.1", "ver6.2", "ver6.3", "ver6.4", "ver6.5", "ver6.6", "ver7", "ver7.1", "ver7.2", "ver7.3", "ver7.4", "ver7.5", "ver7.6", "ver7.7", "ver7.8", "ver7.81", "ver7.82", "ver7.83", "ver7.9", "ver7.91", "ver7.92", "ver7.93", "ver7.X1", "ver7.95", "ver7.X2", "ver7.X", "ver8", "ver8.1", "ver8.2", "ver8.3", "y9", "y9.1a", "y9.1b", "y9.1c", "y9.1d")
        r0 = 0.0 if random_sub else selfsup.V3["r0"]
        sub_path = os.path.join(cache_base, "ssl_subset_random.npy" if random_sub else "ssl_subset.npy")
        if not os.path.exists(sub_path):
            sub, n_cand = selfsup.select_subset(ds, r0, selfsup.V3["n_sub"], args.seed)
            print(f"--- ver3 subset: {len(sub)} of {n_cand} unlabeled with fail ratio >= {r0}")
            np.save(sub_path, sub)
        sub = np.load(sub_path)
        if args.method in ("ver6", "ver6.1", "ver6.2", "ver6.3", "ver6.4", "ver6.5", "ver6.6", "ver7", "ver7.1", "ver7.2", "ver7.3", "ver7.4", "ver7.5", "ver7.6", "ver7.7", "ver7.8", "ver7.81", "ver7.82", "ver7.83", "ver7.9", "ver7.91", "ver7.92", "ver7.93", "ver7.X1", "ver7.95", "ver7.X2", "ver7.X", "ver8", "ver8.1", "ver8.2", "ver8.3", "y9", "y9.1a", "y9.1b", "y9.1c", "y9.1d"):
            suffix = {"ver6": "", "ver6.1": "61", "ver6.2": "62", "ver6.3": "63", "ver6.4": "64", "ver6.5": "65", "ver6.6": "66", "ver7": "7", "ver7.1": "71", "ver7.2": "72", "ver7.3": "73", "ver7.4": "7", "ver7.5": "7", "ver7.6": "7", "ver7.7": "77", "ver7.8": "7", "ver7.81": "7", "ver7.82": "7", "ver7.83": "7", "ver7.9": "7", "ver7.91": "7", "ver7.92": "77", "ver7.93": "7", "ver7.X1": "7X1", "ver7.95": "795", "ver7.X2": "7X2", "ver7.X": "7X", "ver8": "8", "ver8.1": "8", "ver8.2": "82", "ver8.3": "8", "y9": "9", "y9.1a": "91a", "y9.1b": "91b", "y9.1c": "91c", "y9.1d": "91d"}[args.method]
            enc_path = os.path.join(cache_base, f"masklat{suffix}_e{args.ssl_epochs}.pt")
        else:
            enc_name = "simclr_random" if args.method == "ver3" else "simclr"
            enc_path = os.path.join(cache_base, f"{enc_name}{cut_tag(args)}{'_mc' if args.v3_multicrop else ''}_e{args.ssl_epochs}.pt")
        if not os.path.exists(enc_path):
            print(f"--- {args.method} self-supervised pretraining ({args.ssl_epochs} epochs) ---")
            if args.method in ("ver6", "ver6.1", "ver6.2", "ver6.3", "ver6.4", "ver6.5", "ver6.6", "ver7", "ver7.1", "ver7.2", "ver7.3", "ver7.4", "ver7.5", "ver7.6", "ver7.7", "ver7.8", "ver7.81", "ver7.82", "ver7.83", "ver7.9", "ver7.91", "ver7.92", "ver7.93", "ver7.X1", "ver7.95", "ver7.X2", "ver7.X", "ver8", "ver8.1", "ver8.2", "ver8.3", "y9", "y9.1a", "y9.1b", "y9.1c", "y9.1d"):
                # ver6.2 = ver6.1 에서 바둑판 칸만 12 → 6 픽셀, ver6.3 = ver6.2 에서 마스킹본만 4 → 3 장,
                # ver6.4 = ver6.2 에서 가리는 비율만 50% → 30%, ver6.5 = ver6.2 에서 칸만 6 → 4 픽셀
                # ver6.6 = ver6.2 에서 Predictor 만 2층 MLP → 선형, ver7 = ver6.2 와 동일 (7주차 출발점)
                ver6.pretrain(ds, sub, args, device, enc_path, v61=args.method != "ver6",
                              cell={"ver6.2": 6, "ver6.3": 6, "ver6.4": 6, "ver6.5": 4, "ver6.6": 6, "ver7": 6, "ver7.1": 6, "ver7.2": 6, "ver7.3": 6, "ver7.4": 6, "ver7.5": 6, "ver7.6": 6, "ver7.7": 6, "ver7.8": 6, "ver7.81": 6, "ver7.82": 6, "ver7.83": 6, "ver7.9": 6, "ver7.91": 6, "ver7.92": 6, "ver7.93": 6, "ver7.X1": 6, "ver7.95": 6, "ver7.X2": 6, "ver7.X": 6, "ver8": 6, "ver8.1": 6, "ver8.2": 6, "ver8.3": 6, "y9": 6, "y9.1a": 6, "y9.1b": 6, "y9.1c": 6, "y9.1d": 6}.get(args.method, ver6.V61["cell"]),
                              n_masks=3 if args.method == "ver6.3" else ver6.V61["n_masks"],
                              ratio=0.3 if args.method == "ver6.4" else 0.5,
                              mlp=False if args.method == "ver6.6" else None,
                              per_view=args.method == "ver7.2",
                              dilate=wmodels.INPUT_DILATE,
                              stripes=args.method in ("ver7.7", "ver7.92"),
                              curriculum={"ver7.X1": "stripe_first", "ver7.95": "checker_first"}.get(args.method),
                              lab_idx=lab_idx, triplet_w=0.1 if args.method == "ver7.X2" else 0.0,
                              crop=args.method == "ver7.X",
                              size=wmodels.load_size(args.cache) if args.film else None,
                              corrupt=args.corrupt, noise_p=args.noise_p, proj=args.proj,
                              hw_table=np.load(os.path.join(args.cache, "wm811k_shapes_dies.npy")) if args.corrupt == "noise" else None)
            else:
                selfsup.pretrain(ds, sub, args, device, enc_path)
            torch.cuda.empty_cache()
        print(f"--- ver3 cluster + finetune: {ssl_tag(args)} ---")
        selfsup.finetune(ds, lab_idx, sub, enc_path, args, out, device)
        open(os.path.join(out, "done"), "w").close()
        return

    pat = None
    if args.method == "ver2":
        # 패턴 사전 + unlabeled 패턴 점수 (학습 전 한 번, 고정된 supervised 모델)
        pat_path = os.path.join(cache_base, f"patterns{stab_tag(args)}.npz")
        if not os.path.exists(pat_path):
            print("--- ver2 pattern dictionary ---")
            m = build_model(args.arch, NUM_CLASSES).to(device)
            m.load_state_dict(torch.load(sup_ckpt, map_location=device, weights_only=True))
            d = patterns.build(m, ds, lab_idx, device)
            q_arg, q_max = patterns.score(m, ds, ds.unlabeled, d, device)
            np.savez(pat_path, q_arg=q_arg, q_max=q_max, tau_p=d["tau_p"], report=json.dumps(d["report"], ensure_ascii=False))
            del m
            torch.cuda.empty_cache()
        z = np.load(pat_path)
        pat = dict(q_arg=z["q_arg"], q_max=z["q_max"], tau_p=torch.from_numpy(z["tau_p"]).float().to(device))
        print("--- ver2 pattern report: " + str(z["report"]))

    # Stage 2: SSL (랜덤 초기화된 f_theta2)
    print(f"--- stage 2: {ssl_tag(args)} ---")
    train_wafixmatch(ds, lab_idx, sal, args, out, device, pat)
    open(os.path.join(out, "done"), "w").close()


def summary(args):
    rows = {}
    for f in glob.glob(os.path.join(args.runs, "*", "p*", "seed*", "*", "metrics.json")):
        parts = os.path.normpath(f).split(os.sep)
        arch, frac, method = parts[-5], parts[-4][1:], parts[-2]
        with open(f, encoding="utf-8") as fh:
            m = json.load(fh)
        rows.setdefault((arch, method), {}).setdefault(frac, []).append(m["test"]["macro_f1"])
    fracs = sorted({fr for v in rows.values() for fr in v}, key=float)
    print(f"{'arch':12s} {'method':32s}" + "".join(f"{float(fr) * 100:>18g}%" for fr in fracs))
    for (arch, method), v in sorted(rows.items()):
        cells = [f"{np.mean(v[fr]):.3f}({np.std(v[fr]):.3f}) n={len(v[fr])}" if fr in v else "-" for fr in fracs]
        print(f"{arch:12s} {method:32s}" + "".join(f"{c:>19s}" for c in cells))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["prepare", "run", "summary"])
    p.add_argument("--pkl", default="LSWMD.pkl")
    p.add_argument("--cache", default=os.path.join(os.path.expanduser("~"), "wafer_cache"),
                   help="전처리/saliency 캐시 (OneDrive 밖)")
    p.add_argument("--runs", default="runs")
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--split_seed", type=int, default=0, help="train/val/test 분할과 라벨 부분집합 고정용")
    p.add_argument("--frac", type=float, nargs="+", default=[0.05])
    p.add_argument("--seed", type=int, nargs="+", default=[0])
    p.add_argument("--arch", default="resnet18", choices=list(DEFAULT_BATCH))
    p.add_argument("--batch_size", type=int, default=None, help="기본: resnet18 256 / resnet50 128 / densenet121 64")
    p.add_argument("--mu", type=int, default=4)
    p.add_argument("--lambda_u", type=float, default=0.217)
    p.add_argument("--tau_s", type=float, default=0.893, help="pseudo-label confidence threshold")
    p.add_argument("--tau_c", type=float, default=0.493, help="KeepCutout 영역 중요도 threshold")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--wd", type=float, default=3e-4)
    p.add_argument("--pretrain_epochs", type=int, default=100)
    p.add_argument("--epochs", type=int, default=180)
    p.add_argument("--weak", default="rotate", choices=["rotate", "flipshift"])
    p.add_argument("--no_keep", dest="keep", action="store_false", help="saliency 없이 일반 cutout")
    p.add_argument("--method", default="wafixmatch",
                   choices=["wafixmatch", "fixmatch", "ver1", "ver1.1", "ver2", "ver3", "ver3.1", "ver3.2", "ver6", "ver6.1", "ver6.2", "ver6.3", "ver6.4", "ver6.5", "ver6.6", "ver7", "ver7.1", "ver7.2", "ver7.3", "ver7.4", "ver7.5", "ver7.6", "ver7.7", "ver7.8", "ver7.81", "ver7.82", "ver7.83", "ver7.9", "ver7.91", "ver7.92", "ver7.93", "ver7.X1", "ver7.95", "ver7.X2", "ver7.X", "ver8", "ver8.1", "ver8.2", "ver8.3", "y9", "y9.1a", "y9.1b", "y9.1c", "y9.1d"],
                   help="fixmatch: 표준 FixMatch 증강 (weak=flip+shift, strong=기하 RandAugment), saliency 없음. "
                        "ver1: weak=회전, strong=웨이퍼 5종(일반 cutout) + 채택률 기반 Ls 감쇠 (--switch_mask 없으면 감쇠 없는 기반). "
                        "ver1.1: ver1 + Ls <= --ls_start 일 때만 감쇠 시작, lambda_s 단조 감소 (--switch_mask 필수). "
                        "ver3: 기준선 = 일반 자기지도 학습 (무작위 unlabeled 10만 장 SimCLR + labeled 미세조정, 우리 아이디어 없음). "
                        "ver3.1: 불량 비율 필터 + 군집 pseudo-label + 일반 cutout + --v3_* 발전 옵션. "
                        "ver3.2: 자기지도 → 군집 → 미세조정의 처음 버전 (cutout 없음). "
                        "ver6: ver3 의 SimCLR 대신 Latent Multi-Masking Ensemble Consistency (마스킹 4장 평균 vs EMA 원본, 디코더 없음) + ver3 미세조정. "
                        "ver6.1: ver6 + 바둑판 50%% 마스킹(12px 칸) + 원본/마스킹본 서로 다른 회전 + 2층 MLP Predictor. "
                        "ver6.2: ver6.1 에서 바둑판 칸만 6px. "
                        "ver6.3: ver6.2 에서 마스킹본만 4 → 3 장. "
                        "ver6.4: ver6.2 에서 가리는 비율만 50%% → 30%% (바둑판 칸 중 60%%만 가림). "
                        "ver6.5: ver6.2 에서 바둑판 칸만 4px. "
                        "ver6.6: ver6.2 에서 Predictor 만 선형 (512→512). "
                        "ver7: ver6.2 와 동일 (7주차 출발점). "
                        "ver7.1: ver7 에서 백본만 ResNet-34. "
                        "ver7.2: ver7 에서 손실만 mean_i(1 - cos(p_i, z)) (마스킹본별 손실 평균). "
                        "ver7.3: ver7 에서 입력만 2채널 (원본 + 불량 3x3 팽창본, 팽창 후 두 채널 같이 마스킹). "
                        "ver7.4: ver7 인코더 그대로 + 미세조정 손실에 batch-hard Triplet 추가. "
                        "ver7.5: ver7 인코더 그대로 + 미세조정 CE 를 Focal (gamma 2) 로. "
                        "ver7.6: ver7.4 + ver7.5 (Triplet + Focal). "
                        "ver7.7: ver7 에서 마스킹본 4장 = 바둑판 2 + 세로줄 1 + 가로줄 1 (6px, 50%%). "
                        "ver7.8 / ver7.81: ver7 인코더 그대로 + 미세조정 CE 의 none 가중치 0.5 / 0.1 (나머지 1.0). "
                        "ver7.82 / ver7.83: ver7.8 (none 0.5) + Scratch 가중치 2.0 / 5.0. "
                        "ver7.9 / ver7.91: ver7 인코더 그대로 + 미세조정 샘플링을 원래 비율 / 장수 제곱근 비례로. "
                        "ver7.92: ver7.91 샘플링 + ver7.7 (줄무늬) 인코더. ver7.93: ver7.91 + Triplet (ver7.4). "
                        "ver7.X1 / ver7.95: 사전학습 마스킹 단계 전환 줄무늬 25 → 바둑판 25 / 반대. "
                        "ver7.X2: ver7 사전학습 + 라벨 데이터 batch-hard Triplet (가중치 0.1). "
                        "ver7.X: ver7 에서 마스킹본만 회전 → 무작위 크기 자르기 (면적 50~100%%) → 마스킹. "
                        "ver8: ver7 과 동일 (8주차 출발점). "
                        "ver8.1: ver8 인코더 그대로 + 분류기 직전에 웨이퍼 크기 (log die 수) 임베딩을 이어 붙임. "
                        "ver8.2: ver8 + FiLM (크기 → γ, β → ResNet 각 stage 특징맵에 곱하고 더함), 사전학습부터. "
                        "ver8.3: ver8 인코더 그대로 + 평균 풀링 대신 클래스별 학습 풀링 (평균↔최대↔최소). "
                        "y9: ver8 과 동일 (9주차 출발점, 사용자 브랜치 이름; main 에는 ver9 로)")
    p.add_argument("--switch_mask", type=float, default=None,
                   help="ver1/ver1.1 의 m0: 채택률 m >= m0 이면 Ls 가중치를 (1-m)/(1-m0) 로 감쇠")
    p.add_argument("--cosine", action="store_true", help="학습률 lr*cos(7*pi*k/16K) 감쇠 (원 FixMatch)")
    p.add_argument("--ema", type=float, default=0.0, help="평가/저장에 쓸 EMA 모델 decay (원 FixMatch 0.999, 0=끔)")
    p.add_argument("--stage1_only", action="store_true", help="supervised(stage 1)만 학습하고 종료")
    p.add_argument("--imagenet", action="store_true", help="ResNet 을 ImageNet 사전학습 가중치로 시작 (결과 폴더에 _imagenet)")
    p.add_argument("--ls_start", type=float, default=0.05, help="ver1.1 의 L0: 직전 epoch 평균 Ls <= L0 이어야 감쇠 시작 (0.05 ≈ supervised 최고점 epoch)")
    p.add_argument("--no_cutout", action="store_true", help="ver 계열에서 strong 증강의 cutout 제외 (cutout 유무 비교용)")
    p.add_argument("--v3_weight", default="none", choices=["none", "purity"],
                   help="ver3.1 군집 pseudo-label 가중치: purity = 군집 순도 가중치 + 순도 20%% 미만 군집 제외")
    p.add_argument("--v3_nopl", action="store_true", help="ver3.1 ablation: 군집 pseudo-label 끔 (자기지도 + labeled 미세조정)")
    p.add_argument("--v3_multicrop", action="store_true", help="ver3.1 자기지도에 국소 crop view(48x48) 2장 추가")
    p.add_argument("--v3_clusters", type=int, default=100, help="ver3.1 군집 수")
    p.add_argument("--ssl_epochs", type=int, default=50, help="ver3 자기지도 epoch (10만 장, epoch당 ~137초 → 약 2시간)")
    p.add_argument("--ft_epochs", type=int, default=30, help="ver3 미세조정 라운드당 epoch (3 라운드)")
    args = p.parse_args()
    if args.switch_mask is not None and args.method not in ("ver1", "ver1.1"):
        p.error("--switch_mask 는 --method ver1 / ver1.1 에서만 사용")
    if args.no_cutout and args.method not in ("ver1", "ver1.1", "ver2"):
        p.error("--no_cutout 은 ver1 / ver1.1 / ver2 에서만 사용 (ver3.2 는 항상 cutout 없음, ver3 / ver3.1 은 항상 cutout 포함)")
    v3_opts = args.v3_weight != "none" or args.v3_nopl or args.v3_multicrop or args.v3_clusters != 100
    if v3_opts and args.method != "ver3.1":
        p.error("--v3_weight / --v3_nopl / --v3_multicrop / --v3_clusters 는 --method ver3.1 에서만 사용")
    if args.method == "ver3.2":
        args.no_cutout = True       # ver3.2 = 처음 버전 (cutout 없음)
    if args.method in ("ver3", "ver6", "ver6.1", "ver6.2", "ver6.3", "ver6.4", "ver6.5", "ver6.6", "ver7", "ver7.1", "ver7.2", "ver7.3", "ver7.4", "ver7.5", "ver7.6", "ver7.7", "ver7.8", "ver7.81", "ver7.82", "ver7.83", "ver7.9", "ver7.91", "ver7.92", "ver7.93", "ver7.X1", "ver7.95", "ver7.X2", "ver7.X", "ver8", "ver8.1", "ver8.2", "ver8.3", "y9", "y9.1a", "y9.1b", "y9.1c", "y9.1d"):
        args.v3_nopl = True         # 군집 pseudo-label 없음 (labeled 로만 미세조정)
    if args.method == "ver1.1" and args.switch_mask is None:
        p.error("--method ver1.1 은 --switch_mask 필수")
    if args.method == "fixmatch":
        args.weak, args.keep = "flipshift", False
    if args.method in ("ver1", "ver1.1", "ver2", "ver3", "ver3.1", "ver3.2", "ver6", "ver6.1", "ver6.2", "ver6.3", "ver6.4", "ver6.5", "ver6.6", "ver7", "ver7.1", "ver7.2", "ver7.3", "ver7.4", "ver7.5", "ver7.6", "ver7.7", "ver7.8", "ver7.81", "ver7.82", "ver7.83", "ver7.9", "ver7.91", "ver7.92", "ver7.93", "ver7.X1", "ver7.95", "ver7.X2", "ver7.X", "ver8", "ver8.1", "ver8.2", "ver8.3", "y9", "y9.1a", "y9.1b", "y9.1c", "y9.1d"):
        args.weak, args.keep = "rotate", False
    if args.method == "ver7.1":
        args.arch = "resnet34"      # ver7.1 = ver7 에서 백본만 ResNet-18 → 34
    wmodels.IMAGENET = args.imagenet
    if args.method == "ver7.3":
        wmodels.INPUT_DILATE = 3    # ver7.3 = ver7 에서 입력만 2채널 (원본 + 불량 3x3 팽창본)
    # ver7.4 = ver7 인코더 그대로, 미세조정에 Triplet (margin 0.3, 가중치 1.0)
    args.triplet = 1.0 if args.method in ("ver7.4", "ver7.6", "ver7.93") else 0.0
    # ver7.5 = ver7 인코더 그대로, 미세조정 CE → Focal (gamma 2, alpha 없음: 배치가 이미 클래스 균형)
    # ver7.6 = ver7.4 + ver7.5 (Triplet + Focal, 같은 값)
    args.focal = 2.0 if args.method in ("ver7.5", "ver7.6") else 0.0
    # ver7.8 / ver7.81 = ver7 인코더 그대로, 미세조정 CE 에서 none 가중치만 0.5 / 0.1 (균형 샘플러는 유지)
    args.none_w = {"ver7.8": 0.5, "ver7.81": 0.1, "ver7.82": 0.5, "ver7.83": 0.5}.get(args.method, 1.0)
    # ver7.82 / ver7.83 = ver7.8 에서 Scratch 가중치만 2.0 / 5.0
    args.scratch_w = {"ver7.82": 2.0, "ver7.83": 5.0}.get(args.method, 1.0)
    # ver7.9 / ver7.91 = ver7 에서 라벨 샘플링만: 균형 끔 (원래 비율) / 부분 균형 (장수 제곱근 비례)
    args.sample_pow = {"ver7.9": 1.0, "ver7.91": 0.5, "ver7.92": 0.5, "ver7.93": 0.5}.get(args.method, 0.0)
    # ver8.1 = ver8 인코더 그대로, 분류기 입력 = 백본 특징 512 + 크기 임베딩 16
    args.size_head = args.method == "ver8.1"
    # ver8.2 = ver8 + FiLM: 사전학습 / 미세조정 모두 인코더가 크기 조건을 받음 (layer1~4 출력 h*(1+γ)+β)
    args.film = args.method == "ver8.2"
    # ver8.3 = ver8 인코더 그대로, 마지막 평균 풀링 → 클래스별 학습 풀링 (Scratch 처럼 일부에만 있는 불량을 위한)
    args.class_pool = args.method == "ver8.3"
    # y9.1 파일럿 (교수님 제안: 마스킹 대신 노이즈 SSL, debate/y9.1/): a = 마스킹 (projector 없음, y9 와 같은 방식)
    # b = 마스킹 + projector, c = 회전만 + projector, d = die 단위 랜덤 불량 (뷰별 2/4/8/16%) + projector
    args.corrupt = {"y9.1c": "none", "y9.1d": "noise"}.get(args.method, "checker")
    args.noise_p = (0.02, 0.04, 0.08, 0.16) if args.method == "y9.1d" else None
    args.proj = args.method in ("y9.1b", "y9.1c", "y9.1d")
    # ver7.92 = ver7.91 샘플링 + ver7.7 (줄무늬) 인코더, ver7.93 = ver7.91 샘플링 + Triplet (위 args.triplet)

    if args.cmd == "prepare":
        prepare(args.pkl, args.cache, args.size)
        return
    if args.cmd == "summary":
        summary(args)
        return

    args.batch_size = args.batch_size or DEFAULT_BATCH[args.arch]
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    t = time.time()
    ds = WM811K(args.cache, args.size, args.split_seed)
    print(f"data loaded {time.time() - t:.1f}s  train={len(ds.train)} val={len(ds.val)} test={len(ds.test)} unlabeled={len(ds.unlabeled)}")
    fracs, seeds = args.frac, args.seed
    for args.frac in fracs:
        for args.seed in seeds:
            run_one(args, ds, device)


if __name__ == "__main__":
    main()
