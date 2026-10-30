"""
Task 4 - Track individual eggs in a video.

Two parts:
  1. SEGMENT each frame. The video is white eggs on a dark table (high
     contrast), so GrabCut (used in Task 2) is both unnecessary and far too
     slow (~2 s/frame). We use the fast brightness+shape path instead:
     whiteness threshold -> morphology -> egg-shaped components. Same *ideas*
     as Task 2 (whiteness + shape filtering), just without the expensive
     GrabCut stage. This is the allowed "modify the algorithm" step.
  2. TRACK them across frames with a lightweight SORT-style tracker:
     - predict each track's next box from its velocity (constant-velocity),
     - match detections to tracks with IoU + Hungarian assignment,
     - keep a track alive for `max_age` frames through occlusion (the hand),
     - confirm a track after `min_hits` detections (kills flicker).
   Each egg gets a stable ID and a fixed colour.

Run:
  python task4_tracking/track.py --video data/video.mp4 --out outputs/task4/tracked.mp4
"""
import argparse, os
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

# ---------------- fast per-frame egg segmentation ----------------
# Same ideas as Task 2 (whiteness + shape), tuned for this video:
#  - a SMOOTHNESS gate separates the smooth egg from the textured checked
#    shirt (both are bright, but the shirt has high local variance),
#  - a brightness gate + min-area drop the dim glossy table reflections.
def whiteness(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    return (hsv[:, :, 2] / 255.0 * (1 - hsv[:, :, 1] / 255.0) * 255).astype(np.uint8)

def local_std(gray, k=7):
    g = gray.astype(np.float32)
    m = cv2.blur(g, (k, k)); m2 = cv2.blur(g * g, (k, k))
    return np.sqrt(np.maximum(m2 - m * m, 0))

def detect_eggs(bgr, amin=1300, amax_frac=0.35, solidity=0.90, ellfit=0.80,
                aspect=0.50, std_max=18, w_min=150):
    H, W = bgr.shape[:2]
    w = whiteness(bgr)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(w, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    smooth = (local_std(g, 7) < std_max).astype(np.uint8) * 255
    m = cv2.bitwise_and(bright, smooth)
    K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, K)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < amin or a > amax_frac * H * W or len(c) < 5:
            continue
        hull = cv2.contourArea(cv2.convexHull(c))
        sol = a / hull if hull > 0 else 0
        (_, _), (MA, ma), _ = cv2.fitEllipse(c)
        e = np.pi * MA * ma / 4
        ef = min(a, e) / max(a, e) if e > 0 else 0
        asp = min(MA, ma) / max(MA, ma) if max(MA, ma) > 0 else 0
        blob = np.zeros((H, W), np.uint8); cv2.drawContours(blob, [c], -1, 1, -1)
        if sol >= solidity and ef >= ellfit and asp >= aspect and w[blob > 0].mean() >= w_min:
            x, y, bw, bh = cv2.boundingRect(c)
            dets.append([x, y, x + bw, y + bh])
    return np.array(dets, float).reshape(-1, 4)

# ---------------- SORT-style tracker ----------------
def iou(b1, b2):
    xx1 = np.maximum(b1[0], b2[0]); yy1 = np.maximum(b1[1], b2[1])
    xx2 = np.minimum(b1[2], b2[2]); yy2 = np.minimum(b1[3], b2[3])
    w = max(0, xx2 - xx1); h = max(0, yy2 - yy1); inter = w * h
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1]); a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-6)

class Track:
    _n = 0
    def __init__(self, box):
        Track._n += 1
        self.id = Track._n
        self.box = np.array(box, float)
        self.vel = np.zeros(4)
        self.age = 0; self.hits = 1; self.time_since = 0
        self.color = tuple(int(x) for x in np.random.RandomState(self.id).randint(60, 255, 3))
    def predict(self):
        self.box = self.box + self.vel
        self.time_since += 1
        return self.box
    def update(self, box):
        box = np.array(box, float)
        self.vel = 0.5 * self.vel + 0.5 * (box - self.box)
        self.box = box; self.hits += 1; self.time_since = 0

class Sort:
    def __init__(self, iou_thr=0.2, max_age=15, min_hits=3):
        self.iou_thr = iou_thr; self.max_age = max_age; self.min_hits = min_hits
        self.tracks = []
    def step(self, dets):
        for t in self.tracks:
            t.predict()
        if len(self.tracks) and len(dets):
            M = np.zeros((len(dets), len(self.tracks)))
            for i, d in enumerate(dets):
                for j, t in enumerate(self.tracks):
                    M[i, j] = iou(d, t.box)
            r, c = linear_sum_assignment(-M)
            matched = set()
            for i, j in zip(r, c):
                if M[i, j] >= self.iou_thr:
                    self.tracks[j].update(dets[i]); matched.add(i)
            for i, d in enumerate(dets):
                if i not in matched:
                    self.tracks.append(Track(d))
        else:
            for d in dets:
                self.tracks.append(Track(d))
        self.tracks = [t for t in self.tracks if t.time_since <= self.max_age]
        return [t for t in self.tracks if t.hits >= self.min_hits and t.time_since == 0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/video.mp4")
    ap.add_argument("--out", default="outputs/task4/tracked.mp4")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"cannot open {args.video}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    tracker = Sort()

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        dets = detect_eggs(frame)
        tracks = tracker.step(dets)
        for t in tracks:
            x1, y1, x2, y2 = t.box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), t.color, 2)
            cv2.putText(frame, f"Egg {t.id}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, t.color, 2)
        cv2.putText(frame, f"frame {n}  eggs tracked: {len(tracks)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        vw.write(frame)
        if n % 50 == 0:
            print(f"[{n}] dets={len(dets)} active_tracks={len(tracks)}")
        if args.max_frames and n >= args.max_frames:
            break
    cap.release(); vw.release()
    print(f"[done] {n} frames -> {args.out}  (total IDs created: {Track._n})")

if __name__ == "__main__":
    main()
