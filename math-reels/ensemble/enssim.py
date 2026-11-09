"""Ten networks, the same mistakes. The handwriting network of reel 49 (64 -> 32 -> 10, 300 Adam steps on the same
1,257 training digits) trained ten times from ten random starts (seeds 1-10; seed 3 is reel 49's network). Recorded:
each network's score on the 540 unseen digits, the score of their majority vote (a tie goes to the smaller digit), the
digits at least one network gets wrong, the digits all ten get wrong with every network's answer, and what nearest
neighbour (reel 57) reads for those; also the score of averaging the ten networks' probabilities.    python enssim.py  -> ens_results.json"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for d in ("digits", "nearest"):
    sys.path.insert(0, os.path.join(ROOT, d))
import dgsim  # noqa: E402
from nnsim import dist, net, split  # noqa: E402

SEEDS = list(range(1, 11))


def main():
    Xtr, ytr, Xte, yte = split(dgsim.SPLIT_SEED)
    P = np.array([dgsim.forward(net(Xtr, ytr, sd), Xte)[0] for sd in SEEDS])
    pred = P.argmax(2)
    single = (pred == yte).mean(1)
    votes = np.array([np.bincount(pred[:, i], minlength=10) for i in range(len(yte))])
    vote = votes.argmax(1)
    wrong_any = np.flatnonzero((pred != yte).any(0))
    wrong_all = np.flatnonzero((pred != yte).all(0))
    nn = ytr[dist(Xte[wrong_all], Xtr).argmin(1)]
    fooled = []
    for i, j in zip(wrong_all, nn):
        vals, cnt = np.unique(pred[:, i], return_counts=True)
        fooled.append(dict(i=int(i), true=int(yte[i]), answers=pred[:, i].tolist(), top=int(vals[cnt.argmax()]), top_count=int(cnt.max()),
                           conf=float(P[:, i].mean(0).max()), nn=int(j), pixels=Xte[i].tolist()))
    out = dict(seeds=SEEDS, n_test=len(yte), single=single.tolist(), single_right=[int(round(s * len(yte))) for s in single],
               median_right=float(np.median((pred == yte).sum(1))), vote=float(np.mean(vote == yte)), vote_right=int(np.sum(vote == yte)),
               wrong_any=len(wrong_any), wrong_all=len(wrong_all), unanimous=sum(f["top_count"] == len(SEEDS) for f in fooled),
               nn_right=int(np.sum(nn == yte[wrong_all])), avg_prob_right=int(np.sum(P.mean(0).argmax(1) == yte)), fooled=fooled)
    with open(os.path.join(HERE, "ens_results.json"), "w") as fh:
        json.dump(out, fh)
    print("single networks:", out["single_right"], "median", out["median_right"], "of", len(yte))
    print("averaging the ten probabilities instead:", out["avg_prob_right"])
    print("vote of 10:", out["vote_right"], f"({out['vote']:.4f}); wrong by at least one: {out['wrong_any']}; by all ten: {out['wrong_all']} ({out['unanimous']} with one shared answer)")
    print("fooled:", [(f["true"], f["top"], f["top_count"], f["nn"]) for f in fooled], "nearest neighbour right on", out["nn_right"])


if __name__ == "__main__":
    main()
