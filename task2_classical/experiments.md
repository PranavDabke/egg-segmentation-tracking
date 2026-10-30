# Task 2 — Experiment Log (Classical Egg Segmentation)

All trials were tuned and measured on the **dev split only** (88 images); the
eval split (22 images) was never used for tuning. Ground truth = Task-1
Grounded-SAM masks. Primary metric = mean per-image **IoU**; we also watch
**precision/recall** because they reveal *how* a method fails (too-small vs
too-large masks).

> Reproduce any row: edit the parameters in `segment.py` and
> re-run `segment.py` then `evaluate.py`.

| # | Method | Dev IoU | Prec | Rec | Kept? | Why |
|---|--------|:------:|:----:|:---:|:-----:|-----|
| 1 | Otsu on grayscale, largest blob | 0.33 | – | – | ✗ | Picks skin/background, not egg |
| 2 | Otsu + morphology (open/close/fill) | 0.33 | – | – | ✗ | Morphology alone doesn't fix *what* is selected |
| 3 | **Whiteness** `V·(1−S)` + Otsu + largest blob | 0.39 | 0.63 | 0.49 | ✗ | Best threshold feature, but masks too **small** (misses shaded egg side); fails when background is brighter than egg |
| 4 | Whiteness + CLAHE | 0.36 | – | – | ✗ | CLAHE amplifies background texture → worse |
| 5 | Whiteness + shape filter (solidity/ellipse) | 0.33 | – | – | ✗ | On bright backgrounds the egg *merges* with the background into one non-elliptical blob → rejected |
| — | *Diagnosis: oracle best-component IoU ≈ 0.54* | | | | | Even the best blob is capped — **candidate generation** is the bottleneck, because the egg is often **not the brightest thing** (white bags, sinks, sparkly ash beat it) |
| 6 | White **top-hat** (extract compact bright features) | 0.23–0.34 | – | – | ✗ | Only keeps the egg's bright *rim*; interior suppressed |
| 7 | Low-gradient (smoothness) regions | 0.35 | – | – | ✗ | Plain backgrounds are smooth too → merges |
| 8 | Edge + ellipse-fit detector | 0.35 | – | – | ✗ | Canny edges fragment; contours don't close around the egg |
| **9** | **GrabCut from border + shape filter (v2)** | **0.47** | 0.53 | **0.86** | ✅ | **Finds the egg by colour contrast, not brightness.** Recall jumps 0.49→0.86. Over-grabs boundaries & a few bright-clutter false positives (precision 0.53) |
| 10 | v2 without whiteness augment (`--fast`) | 0.46 | 0.49 | 0.94 | ✗ | Even looser; whiteness pass wasn't the precision problem |
| 11 | v2 + **local-brightness tighten** (v3) | 0.41 | 0.49 | 0.62 | ✗ | Tightening **collapsed real eggs** (several `eggs=0`); recall crashed |
| 12 | v2 + **border-touch rejection** only | 0.38 | 0.40 | 0.73 | ✗ | Also deleted valid (hand-held, edge-touching) eggs |

## Key findings
1. **Brightness is the wrong primary cue.** The decisive diagnostic (trial 5→6)
   was that the egg is frequently darker than parts of the background, so any
   threshold-first method has a low ceiling (oracle IoU ≈ 0.54).
2. **GrabCut (region/colour based) is the right primitive** for *locating* the
   egg — it drove recall from 0.49 to 0.86 (trial 9).
3. **Precision is hard to add back.** Every tightening/rejection heuristic we
   tried (trials 11–12) removed real eggs faster than it removed errors — a
   genuine precision/recall trade-off that classical rules couldn't win here.

## Final choice
**Trial 9 (GrabCut + shape)** is the final method: highest IoU, best recall,
simplest story. It is implemented in **`segment.py`** — the single classical
segmentation script, and the one used for all reported results. The earlier
trials above (v1 whiteness, v3 tighten/border-reject) are documented here with
their dev numbers; their parameters can be reproduced by editing the gates in
`segment.py`.

Final: **dev IoU 0.473 / recall 0.857**, eval IoU 0.336. The dev→eval drop is
honest generalisation (the 22 eval images contain more hard white-on-white and
in-water cases). Where it falls short of SAM, and why, is analysed in the report
(Task 3).
