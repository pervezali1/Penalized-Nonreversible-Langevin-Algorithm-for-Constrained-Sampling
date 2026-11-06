# Posting kit — *Every Fibonacci number: one matrix, multiplied.* (dark theme, narrated, 92 s)

## Files

* Reel: `output/fibmatrix_reel.mp4` — 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (narration over the gentle track).
* Cover: `output/cover.png` — three arrows from three starts, settled on the eigenvector line (φ, 1), the dashed
  second eigenvector, and the ratios closing in on φ = 1.6180339887…
* Runs: `fib_results.json` (from `python fibsim.py`, about 25 seconds: it times the millionth Fibonacci number).
* Voice: Kokoro (open-source text-to-speech, kokoro-onnx v1.0), `am_michael` at 1.05×. Script in `scene.py` (`SCRIPT`).

## What's on screen (all of it is real)

A = [[1, 1], [1, 0]] sends the pair (a, b) to (a + b, a), so Aⁿ (1, 0) = (F(n+1), F(n)). Everything is exact
integers (Python) or 80-digit decimals.

* The pairs: from (1, 0): (1, 1), (2, 1), (3, 2), (5, 3), (8, 5), …; from (2, −1): (1, 2), (3, 1), (4, 3), (7, 4),
  (11, 7), … (the Lucas numbers); from (1, −2): (−1, 1), (0, −1), (−1, 0), (−1, −1), (−2, −1), … (the Fibonacci
  numbers, negated). The direction plot draws every arrow the same length; the first two settle along (φ, 1) at
  31.72°, the third along (−φ, −1).
* The eigenvalues: λ² = λ + 1, so φ = (1 + √5)/2 = 1.6180339887… and ψ = (1 − √5)/2 = −0.6180339887…; the
  eigenvectors are (φ, 1) and (ψ, 1) (the dashed line, at 121.72°).
* The ratios: 13/8 = 1.625, 21/13 = 1.61538…, 34/21 = 1.61905…, 55/34 = 1.61765…, 89/55 = 1.61818…, 144/89 =
  1.61798…, 233/144 = 1.61806… (green: the decimal places already matching φ).
* φ¹⁰⁰/√5 = 354,224,848,179,261,915,075.000000000000000000000564…, and F(100) = 354,224,848,179,261,915,075: the
  difference is ψ¹⁰⁰/√5 = 5.6 × 10⁻²².
* F(1,000,000) has 208,988 digits (195328212870…838242546875). On this machine (Python 3.13.16, x86-64), median of
  3 runs: a million additions 7.66 s; A^1,000,000 by repeated squaring, 25 matrix multiplications (19 squarings
  and 6 more), 0.243 s. Both give the same number.

## Honesty notes

* The timings depend on the machine and on Python's big-number multiplication; the ratio, not the seconds, is the
  point, and it grows with n. For small n, plain adding is faster.
* "Start it anywhere": any start off the dashed line. That line has an irrational slope, so every whole-number start
  is off it; a start exactly on it would shrink toward 0 instead.
* "Two hundred and nine thousand digits" rounds 208,988; "a quarter of a second" rounds 0.243 s; "seven point seven"
  rounds 7.66.
* "Off by less than a billionth of a trillionth": 5.6 × 10⁻²² < 10⁻²¹.

## Caption (paste as-is)

```
Every Fibonacci number is hiding in the powers of one tiny matrix. And its eigenvalue is the golden ratio.

Take the matrix [1 1; 1 0]. Times the pair (1, 0), you get (1, 1). Again: (2, 1). Then (3, 2), (5, 3). Each step adds the pair and shifts it: the Fibonacci rule.

Now watch the arrow's direction. Start it anywhere: within a few steps it swings onto the same line, and stays. That line is an eigenvector: along it, the matrix only stretches.

And every step stretches it by the same factor. 13/8, 21/13, 89/55: 1.618. The golden ratio. It's the eigenvalue: λ² = λ + 1.

The other eigenvalue, −0.618, shrinks away. So the 100th Fibonacci number is φ¹⁰⁰/√5, off by less than a billionth of a trillionth.

And squaring the matrix is fast. The millionth Fibonacci number has 208,988 digits. A million additions: 7.7 seconds. 25 matrix multiplications: a quarter of a second.

One matrix. Its eigenvector is the golden direction, its eigenvalue the golden ratio, and every Fibonacci number falls out.

(Exact integers; the timings are from one machine. Every number on screen is computed.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#linearalgebra #fibonacci #goldenratio #eigenvalues #math #python #mathematics
```

## YouTube

Title:

```
The Golden Ratio Is an Eigenvalue (Fibonacci in One Matrix) #shorts
```

Description:

```
The Fibonacci matrix A = [[1, 1], [1, 0]], computed exactly.

Aⁿ (1, 0) = (F(n+1), F(n)). From any start (off the second eigenvector) the direction of Aⁿv settles on the eigenvector (φ, 1), and from then on each step multiplies its length by (very nearly) the eigenvalue φ = (1 + √5)/2 = 1.6180339887…, the root of λ² = λ + 1. The other eigenvalue, ψ = −0.618…, shrinks away, which gives Binet's formula: F(n) = (φⁿ − ψⁿ)/√5, and φ¹⁰⁰/√5 = F(100) + 5.6 × 10⁻²².

Repeated squaring computes A^1,000,000 with 25 matrix multiplications: F(1,000,000), 208,988 digits, in 0.24 s, against 7.7 s for a million additions (Python 3.13, one machine, median of 3 runs).

Voice: Kokoro, an open-source text-to-speech model (am_michael voice).

#linearalgebra #fibonacci #goldenratio #shorts
```

## Pinned comment

```
Why does the direction always settle? Any starting arrow is a mix of the two eigenvectors. Each step multiplies the first part by 1.618 and the second by −0.618, so after n steps the first part has grown 1.618ⁿ times while the second has shrunk to 0.618ⁿ. After 10 steps the ratio between them is about 15,000 to 1, and the arrow points almost exactly along the golden direction.
```
