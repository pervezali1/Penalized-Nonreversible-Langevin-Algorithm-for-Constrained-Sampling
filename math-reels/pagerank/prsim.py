"""PageRank is an eigenvector. A web of six pages (A-F; the links are made up for the example) and the PageRank rule
of Brin and Page (1998): a surfer clicks one of the page's links at random 85% of the time, and 15% of the time jumps
to any of the six pages. Recorded:
  * 100,000 surfers doing exactly that for 60 steps (seeded): the share on each page after every step, and the walks
    of the first 600 (drawn as dots);
  * power iteration r <- G r with G = 0.85 M + 0.15/6, M[i, j] = 1 / (links out of j) if j links to i, and the
    eigenvector of G for its largest eigenvalue (1), to compare;
  * the same rule on Python's standard library (this interpreter's): a link = "module X imports module Y" (any import
    statement in X's source, X's submodules included); modules written in C have no source to read, so no links out.
    python prsim.py  -> pr_results.json"""

import ast
import importlib.util
import json
import os
import platform
import sys
import sysconfig

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = "ABCDEF"
LINKS = {"A": "BC", "B": "C", "C": "A", "D": "C", "E": "DFC", "F": "EC"}
DAMP = 0.85
N_SURF, N_SHOW, STEPS, SEED = 100_000, 600, 60, 59
SKIP = ("test", "idlelib", "turtledemo")


def link_matrix(names, links):
    n = len(names)
    M = np.zeros((n, n))
    for s, ts in links.items():
        for t in ts:
            M[names.index(t), names.index(s)] = 1 / len(ts)
    return M


def surfers(M):
    n = M.shape[0]
    rng = np.random.default_rng(SEED)
    pos = rng.integers(0, n, N_SURF)
    outs = [np.flatnonzero(M[:, j]) for j in range(n)]
    shares, walks = [np.bincount(pos, minlength=n) / N_SURF], [pos[:N_SHOW].copy()]
    for _ in range(STEPS):
        jump = rng.random(N_SURF) >= DAMP
        new = pos.copy()
        for j in range(n):
            here = np.flatnonzero((pos == j) & ~jump)
            new[here] = outs[j][rng.integers(0, len(outs[j]), len(here))]
        new[jump] = rng.integers(0, n, int(jump.sum()))
        pos = new
        shares.append(np.bincount(pos, minlength=n) / N_SURF)
        walks.append(pos[:N_SHOW].copy())
    return np.array(shares), np.array(walks)


def stdlib_graph():
    root = sysconfig.get_paths()["stdlib"]
    mods = sorted(m for m in sys.stdlib_module_names if not m.startswith("_") and m not in SKIP)
    present, edges, has_src = [], {}, set()
    for m in mods:
        try:
            if importlib.util.find_spec(m) is None:
                continue
        except Exception:
            continue
        present.append(m)
        p = os.path.join(root, m)
        files = []
        if os.path.isdir(p):
            for dp, dn, fn in os.walk(p):
                dn[:] = sorted(x for x in dn if x not in ("test", "tests", "__pycache__", "idle_test"))
                files += [os.path.join(dp, f) for f in sorted(fn) if f.endswith(".py")]
        elif os.path.exists(p + ".py"):
            files = [p + ".py"]
        if files:
            has_src.add(m)
        out = set()
        for f in files:
            try:
                tree = ast.parse(open(f, encoding="utf-8", errors="replace").read())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    out |= {a.name.split(".")[0] for a in node.names}
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    out.add(node.module.split(".")[0])
        edges[m] = out
    S = set(present)
    edges = {m: sorted((v & S) - {m}) for m, v in edges.items()}
    return present, edges, has_src


def pagerank(nodes, edges, iters=300):
    idx = {m: i for i, m in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n))
    for s, ts in edges.items():
        for t in ts:
            A[idx[t], idx[s]] = 1
    out = A.sum(0)
    M = np.where(out > 0, A / np.maximum(out, 1), 1 / n)        # a module with no links out: jump anywhere
    r = np.full(n, 1 / n)
    for _ in range(iters):
        r = DAMP * M @ r + (1 - DAMP) / n
    return r, A


def main():
    n = len(NAMES)
    M = link_matrix(NAMES, LINKS)
    G = DAMP * M + (1 - DAMP) / n
    r = np.full(n, 1 / n)
    power = [r.tolist()]
    for _ in range(STEPS):
        r = G @ r
        power.append(r.tolist())
    w, V = np.linalg.eig(G)
    k = int(np.argmax(w.real))
    v = np.abs(V[:, k].real)
    v /= v.sum()
    assert np.abs(v - r).max() < 1e-12 and abs(w[k].real - 1) < 1e-12
    P = np.array(power)
    settle = int(next(i for i in range(len(P)) if np.abs(P[i] - v).max() < 0.001))
    shares, walks = surfers(M)
    nodes, edges, has_src = stdlib_graph()
    rk, A = pagerank(nodes, edges)
    order = np.argsort(-rk, kind="stable")
    indeg = A.sum(1)
    by_in = np.argsort(-indeg, kind="stable")
    place = {nodes[i]: p + 1 for p, i in enumerate(order)}
    in_place = {nodes[i]: p + 1 for p, i in enumerate(by_in)}
    importers = {m: sorted(place[s] for s, ts in edges.items() if m in ts) for m in ("functools", "os")}
    top = [dict(module=nodes[i], rank=float(rk[i]), imported_by=int(indeg[i]), place=p + 1, place_by_imports=in_place[nodes[i]])
           for p, i in enumerate(order[:8])]
    out = dict(names=NAMES, links=LINKS, damping=DAMP, n_surfers=N_SURF, n_show=N_SHOW, steps=STEPS, M=M.tolist(),
               eigenvalue=float(w[k].real), eigvec=v.tolist(), power=power, settle_step=settle,
               shares=shares.tolist(), walks=walks.tolist(), inbound={c: sum(c in ts for ts in LINKS.values()) for c in NAMES},
               python=platform.python_version(), n_modules=len(nodes), n_links=int(A.sum()), n_with_source=len(has_src),
               c_examples=[m for m in ("sys", "math", "time", "itertools", "builtins") if m in nodes and m not in has_src],
               top=top, importers_best_place={m: v_[0] for m, v_ in importers.items()}, importers_places={m: v_[:5] for m, v_ in importers.items()},
               place=dict(functools=place["functools"], os=place["os"]), in_place=dict(functools=in_place["functools"], os=in_place["os"]))
    with open(os.path.join(HERE, "pr_results.json"), "w") as fh:
        json.dump(out, fh)
    print("eigenvector", dict(zip(NAMES, np.round(v, 4))), "eigenvalue", w[k].real, "settled (<0.1%) at step", settle)
    print("surfers after 60 steps", dict(zip(NAMES, shares[-1].round(4))), "max gap to eigenvector", np.abs(shares[-1] - v).max().round(4))
    print(f"Python {out['python']}: {out['n_modules']} modules, {out['n_links']} import links")
    for t in top:
        print(f"  {t['place']}. {t['module']:12s} {t['rank']:.4f}  imported by {t['imported_by']} (#{t['place_by_imports']} by imports)")
    print("best-placed importer: functools", importers["functools"][:5], "os", importers["os"][:5])


if __name__ == "__main__":
    main()
