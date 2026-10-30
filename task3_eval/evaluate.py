"""
Task 3 - Evaluation of the classical method against the SAM pseudo-GT.

Treats Task-1 (Grounded-SAM) masks as ground truth and Task-2 (classical)
masks as predictions. Reports, for the dev and eval splits:

  Pixel level : IoU, Dice, Precision, Recall (mean over images + aggregate)
  Boundary    : Boundary-F1 @2px
  Instance    : match pred<->GT eggs (IoU>=0.5, Hungarian) -> detection P/R/F1
                and count MAE
  Qualitative : IoU histogram, IoU-vs-(egg size) scatter, worst-K overlays

Pseudo-GT is imperfect (SAM over-segments eggs sitting in water), so metrics
are also reported EXCLUDING images whose GT covers >55% of the frame.

Run:
  python task3_eval/evaluate.py --gt outputs/task1 --pred outputs/task2 \
        --split outputs/split.json --out outputs/task3
"""
import argparse, glob, json, os, csv
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def find_img(data, stem):
    for e in ("jpg", "jpeg", "png"):
        g = glob.glob(os.path.join(data, "**", stem + "." + e), recursive=True)
        if g:
            return g[0]
    return None

def iou_dice_pr(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = inter / union if union else 1.0
    dice = 2 * inter / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) else 1.0
    prec = inter / pred.sum() if pred.sum() else (1.0 if gt.sum() == 0 else 0.0)
    rec = inter / gt.sum() if gt.sum() else 1.0
    return iou, dice, prec, rec

def boundary_f1(pred, gt, tol=2):
    def edge(m):
        m = m.astype(np.uint8)
        er = cv2.erode(m, np.ones((3, 3), np.uint8))
        return (m - er) > 0
    ep, eg = edge(pred), edge(gt)
    if ep.sum() == 0 and eg.sum() == 0:
        return 1.0
    if ep.sum() == 0 or eg.sum() == 0:
        return 0.0
    k = 2 * tol + 1
    dil = lambda e: cv2.dilate(e.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    prec = (ep & dil(eg)).sum() / ep.sum()
    rec = (eg & dil(ep)).sum() / eg.sum()
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

def instances(path):
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        return []
    return [(m == v) for v in np.unique(m) if v != 0]

def match_instances(pred_i, gt_i, thr=0.5):
    if not pred_i and not gt_i:
        return 0, 0, 0
    if not pred_i:
        return 0, 0, len(gt_i)
    if not gt_i:
        return 0, len(pred_i), 0
    M = np.zeros((len(pred_i), len(gt_i)))
    for a, p in enumerate(pred_i):
        for b, g in enumerate(gt_i):
            M[a, b] = iou_dice_pr(p, g)[0]
    r, c = linear_sum_assignment(-M)
    tp = sum(M[i, j] >= thr for i, j in zip(r, c))
    return tp, len(pred_i) - tp, len(gt_i) - tp

def evaluate_split(stems, gt_dir, pred_dir, data):
    rows = []
    for s in stems:
        g = cv2.imread(os.path.join(gt_dir, "binary", s + ".png"), 0)
        p = cv2.imread(os.path.join(pred_dir, "binary", s + ".png"), 0)
        if g is None:
            continue
        gt = g > 127
        pred = (p > 127) if p is not None else np.zeros_like(gt)
        iou, dice, prec, rec = iou_dice_pr(pred, gt)
        bf1 = boundary_f1(pred, gt)
        tp, fp, fn = match_instances(
            instances(os.path.join(pred_dir, "instance", s + ".png")),
            instances(os.path.join(gt_dir, "instance", s + ".png")))
        rows.append(dict(stem=s, iou=iou, dice=dice, prec=prec, rec=rec, bf1=bf1,
                         tp=tp, fp=fp, fn=fn,
                         gt_area=float(gt.mean()), bad_gt=gt.mean() > 0.55))
    return rows

def summarize(rows, title):
    def agg(rs):
        if not rs:
            return {}
        f = lambda k: float(np.mean([r[k] for r in rs]))
        tp = sum(r["tp"] for r in rs); fp = sum(r["fp"] for r in rs); fn = sum(r["fn"] for r in rs)
        det_p = tp / (tp + fp) if tp + fp else 0.0
        det_r = tp / (tp + fn) if tp + fn else 0.0
        return dict(n=len(rs), IoU=f("iou"), Dice=f("dice"), Prec=f("prec"),
                    Rec=f("rec"), BoundaryF1=f("bf1"),
                    det_P=det_p, det_R=det_r,
                    det_F1=2*det_p*det_r/(det_p+det_r) if det_p+det_r else 0.0)
    out = {"all": agg(rows), "good_gt": agg([r for r in rows if not r["bad_gt"]])}
    print(f"\n=== {title} ===")
    for k, a in out.items():
        if a:
            print(f" [{k:7s}] n={a['n']:3d}  IoU={a['IoU']:.3f} Dice={a['Dice']:.3f} "
                  f"P={a['Prec']:.3f} R={a['Rec']:.3f} BF1={a['BoundaryF1']:.3f} | "
                  f"det P/R/F1={a['det_P']:.2f}/{a['det_R']:.2f}/{a['det_F1']:.2f}")
    return out

def worst_overlays(rows, data, gt_dir, pred_dir, out_png, k=9):
    rs = sorted([r for r in rows if not r["bad_gt"]], key=lambda r: r["iou"])[:k]
    tiles = []
    for r in rs:
        s = r["stem"]; f = find_img(data, s)
        im = cv2.resize(cv2.imread(f), (256, 256))
        gt = cv2.resize(cv2.imread(os.path.join(gt_dir, "binary", s+".png"), 0), (256, 256)) > 127
        pp = cv2.imread(os.path.join(pred_dir, "binary", s+".png"), 0)
        pr = cv2.resize(pp, (256, 256)) > 127 if pp is not None else np.zeros((256, 256), bool)
        ov = im.copy()
        ov[gt & ~pr] = (0, 0, 255)      # missed (FN) = red
        ov[pr & ~gt] = (255, 0, 0)      # extra  (FP) = blue
        ov[gt & pr] = (0, 255, 0)       # hit    (TP) = green
        ov = cv2.addWeighted(im, 0.5, ov, 0.5, 0)
        cv2.putText(ov, f"IoU{r['iou']:.2f}", (5, 24), 0, 0.7, (0, 255, 255), 2)
        tiles.append(ov)
    while len(tiles) % 3:
        tiles.append(np.zeros((256, 256, 3), np.uint8))
    rows_img = [np.hstack(tiles[i:i+3]) for i in range(0, len(tiles), 3)]
    cv2.imwrite(out_png, np.vstack(rows_img))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="outputs/task1")
    ap.add_argument("--pred", default="outputs/task2")
    ap.add_argument("--split", default="outputs/split.json")
    ap.add_argument("--data", default="data/egg_v1")
    ap.add_argument("--out", default="outputs/task3")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sp = json.load(open(args.split))

    report = {}
    all_rows = {}
    for name in ("dev", "eval"):
        rows = evaluate_split(sp[name], args.gt, args.pred, args.data)
        all_rows[name] = rows
        report[name] = summarize(rows, name)
        with open(os.path.join(args.out, f"per_image_{name}.csv"), "w", newline="") as fp:
            wr = csv.DictWriter(fp, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)

    json.dump(report, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)

    # plots on eval
    ev = [r for r in all_rows["eval"] if not r["bad_gt"]]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist([r["iou"] for r in ev], bins=np.linspace(0, 1, 11), color="#4C78A8", edgecolor="w")
    ax[0].set_title("Eval IoU distribution"); ax[0].set_xlabel("IoU"); ax[0].set_ylabel("images")
    ax[1].scatter([r["gt_area"] for r in ev], [r["iou"] for r in ev], c="#E45756", s=18)
    ax[1].set_title("IoU vs egg size"); ax[1].set_xlabel("GT area fraction"); ax[1].set_ylabel("IoU")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "plots.png"), dpi=120)

    worst_overlays(all_rows["eval"] + all_rows["dev"], args.data, args.gt, args.pred,
                   os.path.join(args.out, "worst_cases.png"))
    print(f"\n[done] wrote metrics.json, per_image_*.csv, plots.png, worst_cases.png to {args.out}")
    print("overlay legend: green=correct, red=missed(FN), blue=extra(FP)")

if __name__ == "__main__":
    main()
