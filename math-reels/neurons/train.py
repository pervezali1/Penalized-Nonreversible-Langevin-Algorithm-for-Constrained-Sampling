"""Train three real MLPs (2, 8 and 64 hidden neurons) on the same two-arm spiral with Adam,
and record the weights at every video frame so the renderer shows the true decision field."""

import json
import sys

import numpy as np

FPS = 60
N_FRAMES = 1200                       # 20.000 s
SEED = 14

# stage: (hidden sizes, first training frame, last training frame, total steps)
# stage: (hidden sizes, first training frame, last training frame, total steps, front-loading exponent)
STAGES = [([2], 30, 240, 1500, 1.6),
          ([8], 270, 510, 2000, 1.6),
          ([32, 32], 540, 900, 1500, 1.0)]
RESETS = [(240, 270), (510, 540)]     # glitch windows between stages (frames)


def spiral(rng, n=200, turns=2.0, noise=0.06):
    t = np.sqrt(np.linspace(0.02, 1, n))
    X, y = [], []
    for c, ph in ((0, 0.0), (1, np.pi)):
        a = t * turns * 2 * np.pi + ph
        r = t * 0.9
        X.append(np.stack([r * np.cos(a), r * np.sin(a)], 1) + rng.normal(0, noise, (n, 2)))
        y.append(np.full(n, c))
    return np.concatenate(X), np.concatenate(y).astype(np.float64)


def init(rng, sizes):
    ps = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        ps.append(rng.normal(0, np.sqrt(2 / a), (a, b)))
        ps.append(np.zeros(b))
    return ps


def forward(ps, X):
    h = X
    acts = [h]
    for i in range(0, len(ps), 2):
        z = h @ ps[i] + ps[i + 1]
        h = np.tanh(z) if i < len(ps) - 2 else z
        acts.append(h)
    return h[:, 0], acts


def predict(ps, X):
    return 1 / (1 + np.exp(-forward(ps, X)[0]))


def train(rng, sizes, X, y, steps, lr=0.01):
    ps = init(rng, sizes)
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    hist = [[p.copy() for p in ps]]
    acc, loss = [], []
    for s in range(1, steps + 1):
        logit, acts = forward(ps, X)
        p = 1 / (1 + np.exp(-logit))
        acc.append(float(((p > 0.5) == y).mean()))
        loss.append(float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))))
        g = (p - y)[:, None] / len(y)
        grads = []
        for i in range(len(ps) - 2, -1, -2):
            grads.insert(0, acts[i // 2].T @ g)
            grads.insert(1, g.sum(0))
            if i > 0:
                g = (g @ ps[i].T) * (1 - acts[i // 2] ** 2)
        for k, (pk, gk) in enumerate(zip(ps, grads)):
            m[k] = 0.9 * m[k] + 0.1 * gk
            v[k] = 0.999 * v[k] + 0.001 * gk * gk
            pk -= lr * (m[k] / (1 - 0.9 ** s)) / (np.sqrt(v[k] / (1 - 0.999 ** s)) + 1e-8)
        hist.append([p_.copy() for p_ in ps])
    p = predict(ps, X)
    acc.append(float(((p > 0.5) == y).mean()))
    loss.append(loss[-1])
    return hist, np.array(acc), np.array(loss)


def frame_step(f, f0, f1, steps, expo=1.6):
    """Steps are spent slightly front-loaded (expo > 1) so the early, fast learning is visible."""
    if f < f0:
        return 0
    if f >= f1:
        return steps
    u = (f - f0) / (f1 - f0)
    return int(round(steps * (1 - (1 - u) ** expo)))


def main(out_path):
    rng = np.random.default_rng(SEED)
    X, y = spiral(rng)
    frames = []          # per frame: (stage index, step)
    stage_data = []
    for si, (hidden, f0, f1, steps, expo) in enumerate(STAGES):
        hist, acc, loss = train(rng, [2] + hidden + [1], X, y, steps)
        stage_data.append(dict(hidden=hidden, hist=hist, acc=acc, loss=loss, f0=f0, f1=f1, steps=steps, expo=expo))
        print(f"stage {si}: {sum(hidden)} neurons, final acc {acc[-1]:.3f}, max {acc.max():.3f}, "
              f"acc@steps 100/500/1000: {acc[100]:.2f}/{acc[500]:.2f}/{acc[min(1000, steps)]:.2f}")
    for f in range(N_FRAMES):
        si = 0
        for k, (hidden, f0, f1, steps, expo) in enumerate(STAGES):
            if f >= f0 - 30:
                si = k
        st = stage_data[si]
        frames.append((si, frame_step(f, st["f0"], st["f1"], st["steps"], st["expo"])))
    acc_f = np.array([stage_data[si]["acc"][s] for si, s in frames])
    loss_f = np.array([stage_data[si]["loss"][s] for si, s in frames])
    first100 = int(np.argmax(acc_f >= 1.0)) if acc_f.max() >= 1.0 else None
    meta = dict(frames=frames, acc=acc_f.tolist(), loss=loss_f.tolist(),
                final_acc=[float(s["acc"][-1]) for s in stage_data],
                neurons=[int(sum(s["hidden"])) for s in stage_data],
                first100_frame=first100, stages=STAGES, resets=RESETS)
    print("first 100% frame", first100, "-> t =", None if first100 is None else first100 / FPS)
    np.savez(out_path, X=X, y=y,
             **{f"s{si}_p{k}": np.stack([h[k] for h in s["hist"]]) for si, s in enumerate(stage_data)
                for k in range(len(s["hist"][0]))})
    with open(out_path.replace(".npz", ".json"), "w") as fh:
        json.dump(meta, fh)


if __name__ == "__main__":
    main(sys.argv[1])
