# Posting kit — *Google's first ranking: one eigenvector.* (dark theme, narrated, 76 s)

## Files

* Reel: `output/pagerank_reel.mp4` — 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (narration over the gentle track).
* Cover: `output/cover.png` — the six-page web with the settled crowd of surfers, the shares against the eigenvector,
  and G r = r.
* Runs: `pr_results.json` (from `python prsim.py`, a few seconds; the Python part reads this interpreter's library).
* Voice: Kokoro (open-source text-to-speech, kokoro-onnx v1.0), `am_michael` at 1.05×. Script in `scene.py` (`SCRIPT`).

## What's on screen (all of it is real)

A web of six pages, A–F (the links are made up for the example: A → B, C; B → C; C → A; D → C; E → C, D, F;
F → C, E), and the PageRank rule: each step a surfer clicks one of the page's links at random with chance 0.85, or
jumps to any of the six pages with chance 0.15.

| Page | Links in | 100,000 simulated surfers after 60 steps | Eigenvector of G (PageRank) |
| --- | --- | --- | --- |
| A | 1 (from C) | 34.19% | 34.25% |
| B | 1 (from A) | 17.03% | 17.06% |
| C | 5 | 37.45% | 37.35% |
| D | 1 | 3.68% | 3.65% |
| E | 1 | 4.00% | 4.05% |
| F | 1 | 3.65% | 3.65% |

* G = 0.85 M + 0.15 · (1/6), with M[i, j] = 1/(links out of j) if j links to i. Its largest eigenvalue is 1
  (computed: 0.9999999999999993) and its eigenvector, scaled to add up to 1, is the PageRank column. Power iteration
  (r ← G r, from equal shares) is within 0.1 points of it by step 12.
* The 600 dots are the first 600 of the 100,000 surfers; the bars are the shares of all 100,000 (seeded).
* Python 3.13.16's standard library: 185 public modules (159 written in Python, 26 in C), 1,254 import links
  ("X imports Y" anywhere in X's source, its submodules included), the same 0.85/0.15 rule:

| PageRank place | Module | PageRank | Imported by (place by that count) |
| --- | --- | --- | --- |
| 1 | sys | 7.3% | 108 (#1) |
| 2 | functools | 4.4% | 37 (#8) |
| 3 | warnings | 3.5% | 51 (#5) |
| 4 | re | 3.4% | 60 (#3) |
| 5 | os | 3.3% | 79 (#2) |
| 6 | abc | 3.2% | 12 (#25) |

  functools is imported by the modules in places 3, 4 and 7 (warnings, re, types); the best-placed module that
  imports os is in place 20.

## Honesty notes

* "Google's original ranking": PageRank as Brin and Page described it in 1998 and 1999; Google's ranking also used
  text matching from the start, and today it uses far more signals.
* The six-page web is invented to show the idea; every number on it is computed.
* "Within a dozen steps": the exact shares (power iteration) are within 0.1 points of the eigenvector by step 12;
  the simulated crowd also carries sampling noise of about ±0.15 points.
* Modules written in C (sys, math, time, itertools, builtins, ...) have no Python source to read, so they get no
  links out (their surfers jump anywhere). Imports inside functions and try blocks count; private modules
  (names starting with "_") and the test packages are left out. Other choices would shuffle the lower places.

## Caption (paste as-is)

```
Google's original ranking was a single eigenvector. Here it is, on a web of six pages.

Each arrow is a link. Page C has 5 links pointing at it. Page A has just 1.

Drop 100,000 surfers on random pages. Every step, each one clicks a random link, or, 15% of the time, jumps to any page.

Within a dozen steps the crowd stops changing. C ends up with 37% of the surfers. A, with its one link, gets 34%, because that link comes from C.

That steady crowd is an eigenvector: write the clicks as a matrix, and multiplying by it leaves the crowd unchanged. Eigenvalue 1. That's PageRank.

Then I ran it on Python's standard library: 185 modules, 1,254 imports. After sys, second place isn't os (imported by 79 modules). It's functools, imported by only 37, because the modules that matter import it.

One matrix, multiplied until it stops changing.

(The six-page web is made up for the example; every number on screen is computed.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#linearalgebra #pagerank #eigenvectors #math #python #algorithms #datascience
```

## YouTube

Title:

```
Google's First Algorithm Was One Eigenvector #shorts
```

Description:

```
PageRank, computed: a random surfer, a matrix, and its eigenvector.

A six-page web (made up for the example) with the PageRank rule (follow a random link with probability 0.85, jump to a random page with probability 0.15). 100,000 simulated surfers settle on A 34.2%, B 17.0%, C 37.4%, D 3.7%, E 4.0%, F 3.7%, matching the eigenvector of the Google matrix G = 0.85 M + 0.15/6 for eigenvalue 1 to within 0.1 points. Page A has one incoming link but ranks second, because the link comes from the top page.

The same computation on Python 3.13's standard library (185 modules, 1,254 import links): 1. sys, 2. functools (imported by 37 modules), 3. warnings, 4. re, 5. os (imported by 79).

Voice: Kokoro, an open-source text-to-speech model (am_michael voice).

#linearalgebra #pagerank #eigenvector #shorts
```

## Pinned comment

```
Why does the 15% jump matter? Without it, a surfer can get stuck: a page with no links out, or a group of pages that only link to each other, would soak up the whole crowd, and the answer would depend on where you started. Jumping to a random page 15% of the time makes the matrix have exactly one steady crowd, and multiplying by it again and again is guaranteed to find it.
```
