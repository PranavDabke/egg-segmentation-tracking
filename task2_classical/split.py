"""
Leak-free dev/eval split (4:1).

The Roboflow set has many *re-shoots of the same scene* (same egg, same
background, slightly different angle/zoom). If near-duplicates land on both
sides of the split, the eval set leaks information from dev. So we:

  1. Compute a perceptual hash (pHash) for every image.
  2. Union-find images whose pHash Hamming distance <= --dist into "scenes".
  3. Assign WHOLE scenes to dev or eval (greedy) until eval ~= 1/5 of images.

Output: outputs/split.json  {"dist":..,"dev":[stems],"eval":[stems],"groups":..}

Run:  python task2_classical/split.py --data data/egg_v1 --out outputs
"""
import argparse, glob, json, os
from PIL import Image
import imagehash

def list_images(data_dir):
    files = []
    for e in ("*.jpg", "*.jpeg", "*.png"):
        files += glob.glob(os.path.join(data_dir, "**", e), recursive=True)
    return sorted(set(files))

class UF:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b): self.p[self.find(a)] = self.find(b)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/egg_v1")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--dist", type=int, default=8, help="pHash Hamming threshold")
    ap.add_argument("--eval-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = list_images(args.data)
    stems = [os.path.splitext(os.path.basename(f))[0] for f in files]
    hashes = [imagehash.phash(Image.open(f).convert("L").resize((128, 128)))
              for f in files]
    n = len(files)

    uf = UF(n)
    for i in range(n):
        for j in range(i + 1, n):
            if (hashes[i] - hashes[j]) <= args.dist:
                uf.union(i, j)
    groups = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)
    groups = sorted(groups.values(), key=len, reverse=True)

    # greedy: fill eval with whole groups up to eval_frac of images
    import random
    rng = random.Random(args.seed)
    order = groups[:]; rng.shuffle(order)
    target = int(round(args.eval_frac * n))
    eval_idx, dev_idx = [], []
    for g in order:
        if len(eval_idx) + len(g) <= target:
            eval_idx += g
        else:
            dev_idx += g
    # any leftover groups that didn't fit -> dev
    placed = set(eval_idx) | set(dev_idx)
    dev_idx += [i for i in range(n) if i not in placed]

    dev = sorted(stems[i] for i in dev_idx)
    ev  = sorted(stems[i] for i in eval_idx)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "split.json"), "w") as fp:
        json.dump({"dist": args.dist, "n_groups": len(groups),
                   "dev": dev, "eval": ev}, fp, indent=2)

    multi = [len(g) for g in groups if len(g) > 1]
    print(f"{n} images -> {len(groups)} scenes "
          f"({len(multi)} multi-image scenes, biggest={max(len(g) for g in groups)})")
    print(f"dev = {len(dev)} images, eval = {len(ev)} images "
          f"({len(ev)/n:.0%})")
    print(f"wrote {os.path.join(args.out, 'split.json')}")

if __name__ == "__main__":
    main()
