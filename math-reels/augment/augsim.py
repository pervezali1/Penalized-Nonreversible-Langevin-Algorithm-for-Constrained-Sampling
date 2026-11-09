"""Fix the one-pixel problem of reel 52 without touching the network (data augmentation). The handwriting network of
reel 49 (64 -> 32 -> 10, same start, same 300 Adam steps) trained two ways on the same 1,257 training digits:
  * plain: the 1,257 pictures as they are (the network of reel 49, retrained exactly);
  * augmented: 5 copies of each, as it is and moved one pixel left, right, up and down (6,285 pictures);
then tested on the same 540 unseen digits, unmoved and moved 1 or 2 pixels (what slides in is blank; the shift is
convolution/cnnsim.py's). Also: the same on six other random splits and starts.    python augsim.py -> aug_results.json"""

import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for d in ("digits", "adversarial", "convolution", "nearest"):
    sys.path.insert(0, os.path.join(ROOT, d))
import dgsim  # noqa: E402
from advsim import trained  # noqa: E402
from cnnsim import shift  # noqa: E402
from nnsim import OTHER, net, split  # noqa: E402

MOVES = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]                 # (right, down) in pixels
TESTS = [("unmoved", 0, 0), ("1 right", 1, 0), ("1 left", -1, 0), ("1 up", 0, -1), ("1 down", 0, 1), ("2 right", 2, 0), ("2 left", -2, 0)]


def augment(X, y):
    return np.vstack([shift(X, dx, dy) for dx, dy in MOVES]), np.tile(y, len(MOVES))


def scores(ps, Xte, yte):
    return {name: float(np.mean(dgsim.forward(ps, shift(Xte, dx, dy))[0].argmax(1) == yte)) for name, dx, dy in TESTS}


def main():
    plain, Xte, yte = trained()
    Xtr, ytr, _, _ = split(dgsim.SPLIT_SEED)
    Xa, ya = augment(Xtr, ytr)
    aug = net(Xa, ya, dgsim.SEED)
    show = json.load(open(os.path.join(ROOT, "digits", "dg_results.json")))["show"]
    reads = {}
    for name, dx, dy in TESTS:
        Xs = shift(Xte[show], dx, dy)
        reads[name] = dict(plain=dgsim.forward(plain, Xs)[0].argmax(1).tolist(), aug=dgsim.forward(aug, Xs)[0].argmax(1).tolist())
    others = []
    for sp, sd in OTHER:
        a, b, c, d = split(sp)
        others.append(dict(split=sp, start=sd, plain=scores(net(a, b, sd), c, d), aug=scores(net(*augment(a, b), sd), c, d)))
    out = dict(n_train=len(Xtr), n_aug=len(Xa), moves=MOVES, tests=[t[0] for t in TESTS], plain=scores(plain, Xte, yte), aug=scores(aug, Xte, yte),
               show=show, reads=reads, others=others, steps=dgsim.STEPS)
    with open(os.path.join(HERE, "aug_results.json"), "w") as fh:
        json.dump(out, fh)
    for k in out["tests"]:
        print(f"{k:8s} plain {out['plain'][k]:.4f}  augmented {out['aug'][k]:.4f}")
    for k in ("1 right", "1 left", "2 right"):
        print(k, "other splits plain", [round(o["plain"][k], 3) for o in others], "aug", [round(o["aug"][k], 3) for o in others])
    print("unmoved other splits", [round(o["plain"]["unmoved"], 3) for o in others], [round(o["aug"]["unmoved"], 3) for o in others])
    print("reads 1 right", reads["1 right"])


if __name__ == "__main__":
    main()
