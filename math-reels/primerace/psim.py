"""The prime race (Chebyshev's bias), computed exactly up to a billion.

Every odd prime leaves remainder 1 or 3 when divided by 4: Team 1 (5, 13, 17, 29, ...) or Team 3 (3, 7, 11, 19, ...).
D(x) = (Team 3 primes up to x) - (Team 1 primes up to x). A segmented sieve walks every prime up to 10^9 and records:

* D at every whole number up to 30,000, and at 4,000 log-spaced points up to 10^9 (for the charts);
* every stretch where Team 1 leads (D < 0) and where the teams tie (D = 0);
* how much of the time each team leads: per whole number up to 10^9, and weighted by 1/x (the logarithmic scale
  on which Rubinstein and Sarnak computed 99.59%, assuming the zeros of the relevant L-function are on the line and a
  linear independence hypothesis);
* the "fair" race F(x) = D(x) - (number of odd primes q with q^2 <= x) / 2: counting each prime square as half a
  point, as Riemann-style explicit formulas do. Every odd square leaves remainder 1, so this gives Team 1 its
  squares back.
    python psim.py <out.npz>  -> out.npz and out.json"""

import json
import math
import sys

import numpy as np

N = 10 ** 9
SEG = 10 ** 7
LIN = 30_000
N_LOG = 4000
X_LOG0 = 10.0


def primes_upto(n):
    s = np.ones(n + 1, bool)
    s[:2] = False
    for p in range(2, int(n ** 0.5) + 1):
        if s[p]:
            s[p * p::p] = False
    return np.flatnonzero(s)


def main(out):
    small = primes_upto(int(N ** 0.5) + 1)
    sq = small[1:].astype(np.int64) ** 2                   # odd prime squares up to N (all leave remainder 1)
    sq = sq[sq <= N]
    xs_log = np.exp(np.linspace(math.log(X_LOG0), math.log(N), N_LOG))
    want = np.floor(xs_log + 1e-6).astype(np.int64)
    want[-1] = N
    d_log = np.zeros(N_LOG, np.int64)
    f_log = np.zeros(N_LOG)
    d_lin = np.zeros(LIN + 1, np.int64)
    pi_log = np.zeros(N_LOG, np.int64)                     # odd primes up to each log sample
    ahead_log = np.zeros(N_LOG, np.int64)                  # numbers 1..x with Team 3 ahead
    cum_pos = 0
    BIG = 10 ** 9
    d_lo = np.full(N_LOG, BIG, np.int64); d_hi = np.full(N_LOG, -BIG, np.int64)     # D's range between samples k-1 and k
    f_lo = np.full(N_LOG, float(BIG)); f_hi = np.full(N_LOG, -float(BIG))
    pi1 = pi3 = 0
    D = 0                                                  # D just before the current segment
    nsq = 0                                                # odd prime squares counted so far
    cnt = dict(pos=0, zero=0, neg=0, fpos=0, fzero=0, fneg=0)
    lw = dict(pos=0.0, zero=0.0, neg=0.0, fpos=0.0, fneg=0.0, fzero=0.0)
    neg_runs, ties = [], 0
    in_neg, neg_start = False, 0
    dmin, dmin_at, dmax, dmax_at = 0, 0, 0, 0
    checkpoints = {}
    first_neg = None
    for lo in range(0, N + 1, SEG):
        hi = min(lo + SEG, N + 1)                          # numbers lo .. hi-1
        s = np.ones(hi - lo, bool)
        if lo == 0:
            s[:3] = False                                  # 0, 1, 2 are not in the race
        for p in small:
            start = max(p * p, (lo + p - 1) // p * p)
            if start < hi:
                s[start - lo::p] = False
        ps = lo + np.flatnonzero(s)
        step = np.where(ps % 4 == 3, 1, -1)
        # events: primes (D +-1) and odd prime squares (fair count -1/2), sorted; D is constant between events
        sqs = sq[(sq >= lo) & (sq < hi)]
        ev_x = np.concatenate([ps, sqs])
        ev_d = np.concatenate([step, np.zeros(len(sqs), np.int64)])
        ev_s = np.concatenate([np.zeros(len(ps), np.int64), np.ones(len(sqs), np.int64)])
        o = np.argsort(ev_x, kind="stable")
        ev_x, ev_d, ev_s = ev_x[o], ev_d[o], ev_s[o]
        starts = np.concatenate([[lo], ev_x])
        ends = np.concatenate([ev_x, [hi]])
        Dv = D + np.concatenate([[0], np.cumsum(ev_d)])     # D on [starts[k], ends[k])
        Sv = nsq + np.concatenate([[0], np.cumsum(ev_s)])
        Fv = Dv - Sv / 2
        ln = (ends - starts).astype(np.int64)
        keep = ln > 0
        a, b, Dk, Fk = starts[keep], ends[keep], Dv[keep], Fv[keep]
        L = b - a
        cnt["pos"] += int(L[Dk > 0].sum()); cnt["zero"] += int(L[Dk == 0].sum()); cnt["neg"] += int(L[Dk < 0].sum())
        cnt["fpos"] += int(L[Fk > 0].sum()); cnt["fzero"] += int(L[Fk == 0].sum()); cnt["fneg"] += int(L[Fk < 0].sum())
        aa = np.maximum(a, 2).astype(np.float64)
        w = np.log(b / aa)                                  # weight 1/x, integrated over [a, b)
        w[b <= 2] = 0
        lw["pos"] += float(w[Dk > 0].sum()); lw["zero"] += float(w[Dk == 0].sum()); lw["neg"] += float(w[Dk < 0].sum())
        lw["fpos"] += float(w[Fk > 0].sum()); lw["fzero"] += float(w[Fk == 0].sum()); lw["fneg"] += float(w[Fk < 0].sum())
        # stretches where Team 1 leads, ties, extremes
        for x0, x1, dv in zip(a, b, Dk):
            if dv < 0 and not in_neg:
                in_neg, neg_start = True, int(x0)
                if first_neg is None:
                    first_neg = int(x0)
            elif dv >= 0 and in_neg:
                in_neg = False
                neg_runs.append((neg_start, int(x0)))
        ties += int(((Dk == 0) & (a >= 3)).sum())
        i = int(Dk.argmin())
        if Dk[i] < dmin:
            dmin, dmin_at = int(Dk[i]), int(a[i])
        i = int(Dk.argmax())
        if Dk[i] > dmax:
            dmax, dmax_at = int(Dk[i]), int(a[i])
        # samples
        m = (want >= lo) & (want < hi)
        if m.any():
            k = np.searchsorted(a, want[m], side="right") - 1
            d_log[m], f_log[m] = Dk[k], Fk[k]
            posL = np.concatenate([[0], np.cumsum(L * (Dk > 0))])
            ahead_log[m] = cum_pos + posL[k] + (want[m] - a[k] + 1) * (Dk[k] > 0)
            pi_log[m] = pi1 + pi3 + np.searchsorted(ps, want[m], side="right")
        cum_pos += int(L[Dk > 0].sum())
        # the range of D (and F) inside each gap between samples, so short leads are never skipped
        ba = np.searchsorted(want, a, side="left")
        bb = np.searchsorted(want, b - 1, side="left")
        ok = ba < N_LOG
        one = ok & (ba == bb)
        np.minimum.at(d_lo, ba[one], Dk[one]); np.maximum.at(d_hi, ba[one], Dk[one])
        np.minimum.at(f_lo, ba[one], Fk[one]); np.maximum.at(f_hi, ba[one], Fk[one])
        for i in np.flatnonzero(ok & (ba != bb)):
            j1 = min(bb[i], N_LOG - 1)
            d_lo[ba[i]:j1 + 1] = np.minimum(d_lo[ba[i]:j1 + 1], Dk[i]); d_hi[ba[i]:j1 + 1] = np.maximum(d_hi[ba[i]:j1 + 1], Dk[i])
            f_lo[ba[i]:j1 + 1] = np.minimum(f_lo[ba[i]:j1 + 1], Fk[i]); f_hi[ba[i]:j1 + 1] = np.maximum(f_hi[ba[i]:j1 + 1], Fk[i])
        if lo == 0:
            xx = np.arange(LIN + 1)
            k = np.searchsorted(a, xx, side="right") - 1
            d_lin[:] = np.where(xx < a[0], 0, Dk[np.maximum(k, 0)])
        pi1 += int((step < 0).sum()); pi3 += int((step > 0).sum())
        D = int(Dv[-1]); nsq = int(Sv[-1])
        for e in range(1, 10):
            if lo < 10 ** e + 1 <= hi:
                below = ps[ps <= 10 ** e]
                st = np.where(below % 4 == 3, 1, -1)
                checkpoints[f"1e{e}"] = dict(team3=int((pi3 - (step > 0).sum()) + (st > 0).sum()),
                                           team1=int((pi1 - (step < 0).sum()) + (st < 0).sum()))
    if in_neg:
        neg_runs.append((neg_start, N + 1))
    tot = sum(lw[k] for k in ("pos", "zero", "neg"))
    small_primes = primes_upto(200)[1:]
    facts = dict(n=N, team1=pi1, team3=pi3, lead_at_n=pi3 - pi1, checkpoints=checkpoints,
                 first_team1_lead=first_neg, team1_lead_runs=neg_runs[:40], team1_lead_runs_count=len(neg_runs),
                 team1_lead_numbers=cnt["neg"], tie_stretches_from_3=ties,
                 share_team3_ahead=cnt["pos"] / N, share_tied=cnt["zero"] / N, share_team1_ahead=cnt["neg"] / N,
                 log_share_team3_ahead=lw["pos"] / tot, log_share_tied=lw["zero"] / tot, log_share_team1_ahead=lw["neg"] / tot,
                 fair_share_team3_ahead=cnt["fpos"] / N, fair_share_tied=cnt["fzero"] / N, fair_share_team1_ahead=cnt["fneg"] / N,
                 fair_log_share_team3_ahead=lw["fpos"] / tot, fair_log_share_team1_ahead=lw["fneg"] / tot,
                 biggest_team3_lead=dmax, biggest_team3_lead_at=dmax_at,
                 biggest_team1_lead=-dmin, biggest_team1_lead_at=dmin_at,
                 odd_prime_squares=int(len(sq)), first_primes=small_primes.tolist())
    lin_primes = primes_upto(LIN)[1:]
    pi_lin = np.searchsorted(lin_primes, np.arange(LIN + 1), side="right")
    assert np.all(np.abs(d_lin) <= pi_lin) and np.all((pi_lin + d_lin) % 2 == 0)
    assert np.all((pi_log + d_log) % 2 == 0)
    assert ahead_log[-1] == cnt["pos"], (ahead_log[-1], cnt["pos"])
    empty = d_lo == BIG                                    # repeated sample points at small x: no numbers in between
    d_lo[empty], d_hi[empty], f_lo[empty], f_hi[empty] = d_log[empty], d_log[empty], f_log[empty], f_log[empty]
    assert np.all(d_lo <= d_log) and np.all(d_hi >= d_log) and (d_lo < 0).sum() > 0
    np.savez(out, xs_log=xs_log, want=want, d_log=d_log, f_log=f_log, pi_log=pi_log, ahead_log=ahead_log,
             d_lo=d_lo, d_hi=d_hi, f_lo=f_lo, f_hi=f_hi,
             d_lin=d_lin, pi_lin=pi_lin, neg_runs=np.array(neg_runs, np.int64))
    with open(out.replace(".npz", ".json"), "w") as fh:
        json.dump(facts, fh, indent=1)
    print(json.dumps({k: v for k, v in facts.items() if k not in ("first_primes",)}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
