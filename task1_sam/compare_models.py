"""
Task 1 - model comparison (Windows-safe import order).

Fixes the native crash: on Windows, importing cv2 before torch/sklearn can
cause a DLL access violation. So we import torch + transformers (which pulls
scikit-learn) FIRST, then cv2. SAM-vit-huge is loaded on CPU directly (it
segfaults a 4GB GPU), while sam-vit-base and FastSAM run on GPU.

Run:
  python task1_sam/compare_models2.py --data data/egg_v1 --meta outputs/task1/meta.json \
         --out outputs/task1_compare --n 10
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # avoid duplicate OpenMP runtime crash
import argparse, glob, json, time
import numpy as np
import torch                                            # import BEFORE cv2
from transformers import SamModel, SamProcessor         # pulls sklearn early, BEFORE cv2
import cv2                                               # safe now

def list_images(data):
    fs = []
    for e in ("*.jpg", "*.jpeg", "*.png"):
        fs += glob.glob(os.path.join(data, "**", e), recursive=True)
    return {os.path.splitext(os.path.basename(f))[0]: f for f in sorted(set(fs))}

class SamRunner:
    def __init__(self, model_id, device):
        self.proc = SamProcessor.from_pretrained(model_id)
        self.model = SamModel.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.name = model_id.split("/")[-1] + ("(cpu)" if device == "cpu" else "")
    def masks(self, bgr, boxes):
        from PIL import Image
        img = Image.fromarray(bgr[:, :, ::-1])
        inp = self.proc(img, input_boxes=[boxes], return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inp)
        m = self.proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())[0]
        sc = out.iou_scores.cpu().numpy()[0]
        union = np.zeros(bgr.shape[:2], bool)
        for j in range(m.shape[0]):
            union |= m[j, sc[j].argmax()].numpy().astype(bool)
        return union

class FastSamRunner:
    def __init__(self, device):
        from ultralytics import FastSAM
        self.model = FastSAM("FastSAM-s.pt")
        self.device = device; self.name = "FastSAM-s"
    def masks(self, bgr, boxes):
        res = self.model(bgr[:, :, ::-1], bboxes=[list(map(int, b)) for b in boxes],
                         device=self.device, verbose=False, retina_masks=True)
        union = np.zeros(bgr.shape[:2], bool)
        if res and res[0].masks is not None:
            for m in res[0].masks.data.cpu().numpy():
                union |= cv2.resize(m, (bgr.shape[1], bgr.shape[0])) > 0.5
        return union

def overlay(bgr, mask, color=(0, 0, 255)):
    out = bgr.copy()
    out[mask] = (0.5 * out[mask] + 0.5 * np.array(color)).astype(np.uint8)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (0, 255, 255), 2)
    return out

def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 1.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/egg_v1")
    ap.add_argument("--meta", default="outputs/task1/meta.json")
    ap.add_argument("--out", default="outputs/task1_compare")
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    gpu = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] gpu available: {gpu}")

    imgs = list_images(args.data)
    meta = json.load(open(args.meta))
    stems = [s for s in meta if meta[s]["boxes"]]
    stems = stems[:: max(1, len(stems) // args.n)][:args.n]
    print(f"[info] comparing on {len(stems)} images")

    runners = []
    for mid, dev in (("facebook/sam-vit-base", gpu),
                     ("facebook/sam-vit-huge", "cpu")):     # huge on CPU on purpose
        try:
            print(f"[load] {mid} on {dev}"); runners.append(SamRunner(mid, dev))
        except Exception as e:
            print(f"[skip] {mid}: {e}")
    try:
        print("[load] FastSAM-s"); runners.append(FastSamRunner(gpu))
    except Exception as e:
        print(f"[skip] FastSAM: {e}")

    times = {r.name: [] for r in runners}
    masks_by = {r.name: {} for r in runners}
    rows = []
    for k, s in enumerate(stems, 1):
        bgr = cv2.imread(imgs[s]); boxes = meta[s]["boxes"]
        tiles = [cv2.resize(bgr, (256, 256))]
        for r in runners:
            t0 = time.time(); m = r.masks(bgr, boxes); times[r.name].append(time.time() - t0)
            masks_by[r.name][s] = m
            tiles.append(cv2.resize(overlay(bgr, m), (256, 256)))
        for t, lab in zip(tiles, ["image"] + [r.name for r in runners]):
            cv2.putText(t, lab, (4, 18), 0, 0.45, (0, 255, 0), 2)
        rows.append(np.hstack(tiles))
        print(f"  [{k}/{len(stems)}] {s[:28]}")
    cv2.imwrite(os.path.join(args.out, "comparison.jpg"), np.vstack(rows))

    print("\n=== speed (sec/image) ===")
    for r in runners:
        print(f"  {r.name:20s} {np.mean(times[r.name]):.3f}")
    if len(runners) >= 2:
        print("\n=== mask agreement IoU (vs first model) ===")
        base = runners[0].name
        for r in runners[1:]:
            ius = [iou(masks_by[base][s], masks_by[r.name][s]) for s in stems]
            print(f"  {base} vs {r.name:20s} {np.mean(ius):.3f}")
    print(f"\n[done] wrote {args.out}/comparison.jpg")

if __name__ == "__main__":
    main()
