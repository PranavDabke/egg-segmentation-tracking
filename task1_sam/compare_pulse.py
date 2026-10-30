"""
Comparison pulse: per-image mask agreement of SAM-huge and FastSAM vs SAM-base
(our ground-truth model), across N images. Shows where the models agree and
where they diverge. Saves a clean white pulse graph + a CSV.

Run:  python task1_sam/compare_pulse.py --n 12
      (add --no-huge to skip the slow CPU SAM-huge and run in ~15s)
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse, glob, json
import numpy as np
import torch
from transformers import SamModel, SamProcessor
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def list_images(data):
    fs = []
    for e in ("*.jpg", "*.jpeg", "*.png"):
        fs += glob.glob(os.path.join(data, "**", e), recursive=True)
    return {os.path.splitext(os.path.basename(f))[0]: f for f in sorted(set(fs))}

class Sam:
    def __init__(self, mid, dev):
        self.proc = SamProcessor.from_pretrained(mid)
        self.model = SamModel.from_pretrained(mid).to(dev).eval(); self.dev = dev
    def masks(self, bgr, boxes):
        from PIL import Image
        inp = self.proc(Image.fromarray(bgr[:, :, ::-1]), input_boxes=[boxes], return_tensors="pt").to(self.dev)
        with torch.no_grad():
            out = self.model(**inp)
        m = self.proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())[0]
        sc = out.iou_scores.cpu().numpy()[0]
        u = np.zeros(bgr.shape[:2], bool)
        for j in range(m.shape[0]):
            u |= m[j, sc[j].argmax()].numpy().astype(bool)
        return u

def iou(a, b):
    un = (a | b).sum()
    return (a & b).sum() / un if un else 1.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/egg_v1")
    ap.add_argument("--meta", default="outputs/task1/meta.json")
    ap.add_argument("--out", default="outputs/task1_compare")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--no-huge", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    gpu = "cuda" if torch.cuda.is_available() else "cpu"

    imgs = list_images(args.data); meta = json.load(open(args.meta))
    stems = [s for s in meta if meta[s]["boxes"]]
    stems = stems[:: max(1, len(stems) // args.n)][:args.n]

    base = Sam("facebook/sam-vit-base", gpu)
    huge = None if args.no_huge else Sam("facebook/sam-vit-huge", "cpu")
    from ultralytics import FastSAM
    fast = FastSAM("FastSAM-s.pt")

    rows = []
    for i, s in enumerate(stems, 1):
        bgr = cv2.imread(imgs[s]); boxes = meta[s]["boxes"]
        mb = base.masks(bgr, boxes)
        r = fast(bgr[:, :, ::-1], bboxes=[list(map(int, b)) for b in boxes],
                 device=gpu, verbose=False, retina_masks=True)
        mf = np.zeros(bgr.shape[:2], bool)
        if r and r[0].masks is not None:
            for m in r[0].masks.data.cpu().numpy():
                mf |= cv2.resize(m, (bgr.shape[1], bgr.shape[0])) > 0.5
        row = {"img": s.split("_jpg")[0].replace("not_", ""), "FastSAM": iou(mf, mb)}
        if huge:
            row["SAMhuge"] = iou(huge.masks(bgr, boxes), mb)
        rows.append(row); print(f"[{i}/{len(stems)}] {row}")

    import csv
    with open(os.path.join(args.out, "pulse_agreement.csv"), "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(12, 4.2))
    if huge:
        ax.plot(x, [r["SAMhuge"] for r in rows], "-o", color="#e41a1c", lw=1.8, ms=5, label="SAM-huge vs base")
    ax.plot(x, [r["FastSAM"] for r in rows], "-o", color="#2b8cbe", lw=1.8, ms=5, label="FastSAM vs base")
    ax.axhline(1.0, ls=":", color="#888", lw=1, label="perfect match")
    ax.set_xticks(x); ax.set_xticklabels([r["img"] for r in rows], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("agreement with SAM-base (IoU)"); ax.set_ylim(0, 1.05)
    ax.set_title("Model comparison pulse - agreement with SAM-base, per image", fontweight="bold")
    ax.legend(loc="lower center", ncol=3, frameon=True); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "comparison_pulse.png"), dpi=130, facecolor="white")
    print(f"[done] wrote {args.out}/comparison_pulse.png + pulse_agreement.csv")

if __name__ == "__main__":
    main()
