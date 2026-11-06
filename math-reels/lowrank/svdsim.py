"""Compress a photo with the singular value decomposition. The photo: "chelsea" from scikit-image's sample data
(Chelsea the cat, 300 x 451 pixels, 3 colours; released with no copyright restrictions, CC0, by Stefan van der
Walt). Each colour channel X (300 x 451 numbers, 0-255) is split as X = sum_i s_i u_i v_i^T (numpy's SVD: s_i the
strengths, largest first; u_i a column of 300 numbers, v_i a row of 451), and rebuilt from the first k pieces.
Keeping k pieces of all three channels takes 3 k (300 + 451 + 1) numbers, against 405,900 for the photo.
Recorded: for each k, the numbers kept, the relative error ||X - X_k|| / ||X|| and the average (root-mean-square)
error in shades of 255; the strengths; the photo's checksum; and, for comparison, one 300 x 451 grid of pure static (seeded random whole
shades 0-255) rebuilt the same way.    python svdsim.py  -> svd_results.json"""

import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KS = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300]


def photo():
    import skimage
    import skimage.data
    path = os.path.join(os.path.dirname(skimage.data.__file__), "chelsea.png")
    return skimage.data.chelsea(), path, skimage.__version__


def decompose(im):
    X = im.astype(np.float64)
    return X, [np.linalg.svd(X[..., c], full_matrices=False) for c in range(3)]


def rebuild(svds, k):
    return np.stack([(U[:, :k] * S[:k]) @ Vt[:k] for U, S, Vt in svds], -1)


def piece(svds, i):
    return np.stack([S[i] * np.outer(U[:, i], Vt[i]) for U, S, Vt in svds], -1)


def main():
    im, path, ver = photo()
    X, svds = decompose(im)
    m, n, _ = X.shape
    tot = np.sqrt((X ** 2).sum())
    rows = []
    for k in KS:
        Xk = rebuild(svds, k)
        rows.append(dict(k=k, numbers=3 * k * (m + n + 1), share=3 * k * (m + n + 1) / X.size,
                         rel_err=float(np.sqrt(((X - Xk) ** 2).sum()) / tot), rms=float(np.sqrt(((X - Xk) ** 2).mean())),
                         rms_clipped=float(np.sqrt(((X - np.clip(np.rint(Xk), 0, 255)) ** 2).mean()))))
    noise = np.random.default_rng(0).integers(0, 256, (m, n)).astype(np.float64)
    Un, Sn, Vn = np.linalg.svd(noise, full_matrices=False)
    static = {k: float(np.sqrt(((noise - (Un[:, :k] * Sn[:k]) @ Vn[:k]) ** 2).mean())) for k in (20, 50, 100, 200)}
    full = rebuild(svds, min(m, n))
    out = dict(shape=[m, n, 3], numbers=int(X.size), skimage=ver, sha256=hashlib.sha256(open(path, "rb").read()).hexdigest(),
               rows=rows, strengths=[S.tolist() for _, S, _ in svds], exact_rebuild_err=float(np.abs(full - X).max()), static_rms=static)
    with open(os.path.join(HERE, "svd_results.json"), "w") as fh:
        json.dump(out, fh)
    for r in rows:
        print(f"k={r['k']:3d}: {r['numbers']:7,d} numbers ({r['share']:.1%}), error {r['rel_err']:.2%}, {r['rms']:.1f} shades (rounded {r['rms_clipped']:.1f})")
    print("static, average error (shades):", {k: round(v, 1) for k, v in static.items()})
    print("strengths (red)", np.round(svds[0][1][:5]), "all 300 pieces rebuild it to", out["exact_rebuild_err"])


if __name__ == "__main__":
    main()
