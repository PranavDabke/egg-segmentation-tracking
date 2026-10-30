"""
Task 2 - Classical (non-deep-learning) egg segmentation.  [v2]

Why v2: v1 thresholded a "whiteness" map, but the egg is often NOT the
brightest thing (white bags, sinks, sparkly ash are brighter), so it missed
eggs entirely. v2 extracts the foreground OBJECT with GrabCut seeded from the
image border (brightness-independent), then keeps only EGG-SHAPED components
(convex / elliptical / right size). A whiteness pass is added as a second
candidate source for low-contrast cases, and duplicates are merged by IoU.

The experiment log that led here is in task2_classical/experiments.md.

Run:
  python task2_classical/segment.py --data data/egg_v1 --out outputs/task2
  (add --fast to skip the whiteness augmentation / speed up)
Outputs per image:  binary/<stem>.png, instance/<stem>.png, overlay/<stem>.jpg
"""
import argparse, glob, os
import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes

P = dict(amin=500, amax_frac=0.80,
         solidity=0.85, ellfit=0.72, aspect=0.38,
         open_k=7, gc_iter=5, inset=0.06, augment=True)

def whiteness(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    return (hsv[:, :, 2] / 255.0 * (1 - hsv[:, :, 1] / 255.0) * 255).astype(np.uint8)

def ccs(mask, amin):
    mask = mask.astype(np.uint8)
    n, lab, _, _ = cv2.connectedComponentsWithStats(mask, 8)
    return [binary_fill_holes(lab == i) for i in range(1, n) if (lab == i).sum() >= amin]

def grabcut_border(bgr, inset, iters):
    H, W = bgr.shape[:2]
    m = np.zeros((H, W), np.uint8)
    bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    rect = (int(inset * W), int(inset * H), int((1 - 2 * inset) * W), int((1 - 2 * inset) * H))
    try:
        cv2.grabCut(bgr, m, rect, bg, fg, iters, cv2.GC_INIT_WITH_RECT)
    except Exception:
        return np.zeros((H, W), bool)
    return (m == 1) | (m == 3)

def egg_metrics(blob):
    cnts, _ = cv2.findContours(blob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    hull = cv2.contourArea(cv2.convexHull(cnt))
    sol = area / hull if hull > 0 else 0.0
    ell = asp = 0.0
    if len(cnt) >= 5:
        (_, _), (MA, ma), _ = cv2.fitEllipse(cnt)
        e = np.pi * MA * ma / 4
        ell = min(area, e) / max(area, e) if e > 0 else 0.0
        asp = min(MA, ma) / max(MA, ma) if max(MA, ma) > 0 else 0.0
    return area, sol, ell, asp

def segment(bgr, p=P):
    H, W = bgr.shape[:2]
    K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["open_k"],) * 2)

    cand = []
    fg = grabcut_border(bgr, p["inset"], p["gc_iter"])
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, K)
    cand += ccs(fg * 255, p["amin"])
    if p["augment"]:
        w = whiteness(bgr)
        for pct in (78, 90):
            th = (w >= np.percentile(w, pct)).astype(np.uint8) * 255
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, K)
            cand += ccs(th, p["amin"])

    amax = p["amax_frac"] * H * W
    scored = []
    for c in cand:
        a, sol, ell, asp = egg_metrics(c)
        passes = (p["amin"] <= a <= amax and sol >= p["solidity"]
                  and ell >= p["ellfit"] and asp >= p["aspect"])
        scored.append((sol * ell * np.sqrt(a), passes, c))
    scored.sort(key=lambda x: -x[0])

    kept = []
    for score, passes, c in scored:
        if not passes:
            continue
        if any((c & k).sum() > 0.5 * min(c.sum(), k.sum()) for k in kept):  # NMS
            continue
        kept.append(c)
    if not kept and scored:              # fallback: best single blob
        kept = [scored[0][2]]

    inst = np.zeros((H, W), np.uint16)
    for i, c in enumerate(sorted(kept, key=lambda b: -b.sum()), 1):
        inst[c & (inst == 0)] = i
    return inst

def save_overlay(bgr, inst, path):
    out = bgr.copy()
    for lab in range(1, int(inst.max()) + 1):
        cnts, _ = cv2.findContours((inst == lab).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (0, 255, 0), 2)
    cv2.imwrite(path, out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/egg_v1")
    ap.add_argument("--out", default="outputs/task2")
    ap.add_argument("--fast", action="store_true", help="skip whiteness augmentation")
    args = ap.parse_args()
    if args.fast:
        P["augment"] = False

    files = []
    for e in ("*.jpg", "*.jpeg", "*.png"):
        files += glob.glob(os.path.join(args.data, "**", e), recursive=True)
    files = sorted(set(files))
    for sub in ("binary", "instance", "overlay"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    for i, f in enumerate(files, 1):
        stem = os.path.splitext(os.path.basename(f))[0]
        bgr = cv2.imread(f)
        inst = segment(bgr)
        cv2.imwrite(os.path.join(args.out, "binary", stem + ".png"),
                    (inst > 0).astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(args.out, "instance", stem + ".png"), inst.astype(np.uint8))
        save_overlay(bgr, inst, os.path.join(args.out, "overlay", stem + ".jpg"))
        if i % 10 == 0 or i == len(files):
            print(f"[{i:3d}/{len(files)}] {stem[:36]:36s} eggs={int(inst.max())}")
    print(f"[done] wrote predictions to {args.out}")

if __name__ == "__main__":
    main()
