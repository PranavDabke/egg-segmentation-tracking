"""
Task 1 - Segmentation using a pretrained model (Grounded-SAM).

Pipeline per image:
  1. Grounding DINO (open-vocabulary detector) with the text prompt "egg"
     -> bounding box(es) around every egg.
  2. SAM (Segment Anything) prompted with each box -> one high-quality mask per egg.

These masks are treated as PSEUDO GROUND TRUTH for Tasks 2 and 3.

Outputs (under --out):
  binary/<stem>.png      union of all egg masks, 0/255   (used as GT in eval)
  instance/<stem>.png    labelled mask, 0=bg,1..N=eggs    (used for tracking/counts)
  overlay/<stem>.jpg      original + red egg overlay       (for the report / QA)
  meta.json               per-image boxes, scores, #eggs

Run (GPU auto-detected):
  python task1_sam/make_masks.py --data data/egg_v1 --out outputs/task1
Options:
  --sam facebook/sam-vit-base   (use sam-vit-huge for best quality if VRAM allows)
  --box-th 0.25 --text-th 0.25  (lower if some eggs are missed)
"""
import argparse, json, os, glob
import numpy as np
from PIL import Image
import cv2
import torch
from scipy.ndimage import binary_fill_holes, label as cc_label

def list_images(data_dir):
    exts = ("*.jpg", "*.jpeg", "*.png")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(data_dir, "**", e), recursive=True)
    # ignore anything we might have written before
    files = [f for f in files if "outputs" not in f.replace("\\", "/").split("/")]
    return sorted(set(files))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/egg_v1")
    ap.add_argument("--out", default="outputs/task1")
    ap.add_argument("--gdino", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--sam", default="facebook/sam-vit-base")
    ap.add_argument("--box-th", type=float, default=0.25)
    ap.add_argument("--text-th", type=float, default=0.25)
    args = ap.parse_args()

    from transformers import (AutoProcessor, AutoModelForZeroShotObjectDetection,
                              SamModel, SamProcessor)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device = {device}"
          f"{' / ' + torch.cuda.get_device_name(0) if device=='cuda' else ''}")

    files = list_images(args.data)
    assert files, f"No images found under {args.data}"
    print(f"[info] {len(files)} images")

    for sub in ("binary", "instance", "overlay"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    print(f"[info] loading detector {args.gdino}")
    gproc = AutoProcessor.from_pretrained(args.gdino)
    gmodel = AutoModelForZeroShotObjectDetection.from_pretrained(args.gdino).to(device).eval()

    print(f"[info] loading SAM {args.sam}")
    sproc = SamProcessor.from_pretrained(args.sam)
    smodel = SamModel.from_pretrained(args.sam).to(device).eval()

    meta = {}
    for i, f in enumerate(files, 1):
        stem = os.path.splitext(os.path.basename(f))[0]
        image = Image.open(f).convert("RGB")
        W, H = image.size

        # --- 1. detect eggs ---
        boxes = detect(image, gproc, gmodel, device, args.box_th, args.text_th)
        if len(boxes) == 0:  # retry looser once
            boxes = detect(image, gproc, gmodel, device, 0.15, 0.15)

        inst = np.zeros((H, W), np.uint16)
        used = 0
        if len(boxes):
            masks = sam_masks(image, boxes, sproc, smodel, device)
            # sort by area so bigger eggs get lower ids (stable, cosmetic)
            order = np.argsort([-(m.sum()) for m in masks])
            for m in (masks[o] for o in order):
                if m.sum() < 80:            # drop tiny spurious masks
                    continue
                used += 1
                inst[m & (inst == 0)] = used

        binary = (inst > 0).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(args.out, "binary", stem + ".png"), binary)
        cv2.imwrite(os.path.join(args.out, "instance", stem + ".png"),
                    inst.astype(np.uint8))
        save_overlay(np.array(image)[:, :, ::-1], inst,
                     os.path.join(args.out, "overlay", stem + ".jpg"))
        meta[stem] = {"n_eggs": int(used),
                      "boxes": [[round(float(x), 1) for x in b] for b in boxes]}
        print(f"[{i:3d}/{len(files)}] {stem[:40]:40s} eggs={used}")

    with open(os.path.join(args.out, "meta.json"), "w") as fp:
        json.dump(meta, fp, indent=2)
    tot = sum(v["n_eggs"] for v in meta.values())
    miss = [k for k, v in meta.items() if v["n_eggs"] == 0]
    print(f"\n[done] {tot} eggs over {len(files)} images; "
          f"{len(miss)} images with 0 eggs: {miss[:10]}")

def detect(image, proc, model, device, box_th, text_th):
    inputs = proc(images=image, text="egg.", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    res = proc.post_process_grounded_object_detection(
        out, inputs.input_ids, threshold=box_th, text_threshold=text_th,
        target_sizes=[image.size[::-1]])[0]
    return [b.tolist() for b in res["boxes"]]

def sam_masks(image, boxes, proc, model, device):
    inputs = proc(image, input_boxes=[boxes], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    masks = proc.image_processor.post_process_masks(
        out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu())[0]        # (nb, 3, H, W) bool
    scores = out.iou_scores.cpu().numpy()[0]            # (nb, 3)
    best = [masks[j, scores[j].argmax()].numpy().astype(bool)
            for j in range(masks.shape[0])]
    return best

def save_overlay(bgr, inst, path):
    out = bgr.copy()
    red = np.zeros_like(bgr); red[:, :, 2] = 255
    m = inst > 0
    out[m] = (0.5 * out[m] + 0.5 * red[m]).astype(np.uint8)
    for lab in range(1, inst.max() + 1):
        cnts, _ = cv2.findContours((inst == lab).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (0, 255, 255), 2)
    cv2.imwrite(path, out)

if __name__ == "__main__":
    main()
