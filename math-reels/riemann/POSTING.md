# Posting kit — *The $1,000,000 math problem: the Riemann Hypothesis in 30 seconds.* (dark theme, no narration, 30 s)

## Files

* Reel: `output/riemann_reel.mp4` — 30.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −14 LUFS.
* Cover: `output/cover.png` — "…and the primes appear": the prime staircase built from 183 zeros, the zeta curve
  looping through 0.
* Computation: `rsim.py` (`python rsim.py out.npz` writes every number below; about two minutes).

## What's on screen (all of it is computed)

* **Top: the prime count.** The amber staircase is the true number of primes up to x, for x up to 50 (15 primes).
  The cyan curve is Riemann's explicit formula using only the zeros found so far:
  π(x) = Σ μ(n)/n · J(x^(1/n)), J(y) = li(y) − Σ over zeros Ei(ρ ln y) − ln 2 + ∫ dt / (t (t² − 1) ln t), with
  n = 1, 2, 3, 5. With no zeros it is a smooth curve; each zero adds one wave.
* **Bottom left: the line.** A point climbs the critical line (real part ½); the number beside it is its height.
* **Bottom right: the zeta function's output** at that point, traced as a curve (mpmath, sampled every 0.02).
  Every time the curve passes through 0, a zero is marked on the line (first three: heights 14.13, 21.02, 25.01).
* **563 zeros** are found by the end of the loop, up to height 892.5 (506 by the 30-second mark; the first 2.4 s of
  the reel continue the climb). They are all on the line: the number of sign changes of the function on the line
  equals the exact count of all zeros in the strip up to that height (563 = 563). A small but real check of the
  hypothesis.
* **How good is the cyan curve?** Largest error midway between whole numbers (2.5, 3.5, …, 49.5):

| Zeros used | 0 | 10 | 100 | 200 | 563 |
| --- | --- | --- | --- | --- | --- |
| Largest error | 0.71 | 0.45 | 0.14 | 0.07 | 0.04 |

  From 10 zeros on, rounding the curve gives every prime count up to 50 exactly.
* **"12 trillion zeros checked":** Platt and Trudgian (2021) verified that the lowest 12,363,153,437,138 zeros, up to
  height 3,000,175,332,800, are on the line. Cross-check in `rsim.py`: the main term of the zero-counting formula at
  that height gives 12,363,153,437,137.1.

## Honesty notes

* **The news.** The reel says only: "Nov 20, 2026: a proof was announced." What the coverage found by web search on
  Nov 23, 2026 agrees on: Levent Alpöge, a mathematician at Anthropic, announced a proof written with the AI system
  Claude, published as a formal proof in the Lean proof assistant; the Lean and Mathlib organisations reported that
  the formal proof checks; no human-readable proof has been published and mathematicians' reactions are mixed
  (headlines: Washington Post, Nature, Scientific American). The articles themselves could not be opened from the
  build environment, so re-read the latest status before posting, and edit the caption's last paragraph if it has
  changed. This reel was made with Claude, the same family of AI named in the story.
* "Unsolved for 167 years": Riemann's paper is from 1859; 2026 − 1859 = 167. The $1,000,000 is the Clay Millennium
  Prize (2000); its rules ask for publication and a waiting period, so no prize has been awarded.
* Each zero's own wave is drawn four times taller than it is (labelled) so it can be seen; the cyan curve uses the
  real size.
* The zeta picture zooms out as the climb speeds up (the dashed circle has radius 1), and its rare very large loops
  run off the panel. The climb is slow for the first three zeros, then 75 per second, then 30 per second.
* At a step itself the formula passes halfway up and overshoots a little on each side (like any sum of waves), which
  is why the errors are measured midway between whole numbers.
* "Primes lock your cards and messages": encryption such as RSA is built on prime numbers. A proof of the hypothesis
  would not break it (pinned comment).
* "Whole areas of maths already assume it": many theorems are proved "assuming the Riemann Hypothesis" (or its
  generalisations).
* The hum is the zeros themselves: the first eight heights played as pitches at 10 Hz per unit (141 Hz, 210 Hz,
  250 Hz, 304 Hz, 329 Hz, 376 Hz, 409 Hz, 433 Hz), each entering when its zero is found.

## Caption (paste as-is)

```
The most famous unsolved problem in maths may have just fallen. On Nov 20, 2026, a proof of the Riemann Hypothesis was announced. So what is it?

Start with the primes: 2, 3, 5, 7, 11, 13… They look random.

In 1859, Bernhard Riemann built a function (zeta) and fed it the numbers on one line. Sometimes the output hits exactly zero. Each of those zeros is a wave. Add the waves to the prime count, and the primes appear, step by step. In this reel: 563 zeros, every prime up to 50.

Riemann's guess: every zero sits on that one line. The first 12 trillion have been checked. All on it. But a proof for ALL of them has been missing for 167 years, and it's worth $1,000,000.

Why care? Primes lock your cards and messages, and this is the law of how they're spread. Whole areas of maths already assume it's true.

About the new proof: it was written with AI and checked line by line by a computer. Mathematicians are still working through it, so for now it's a claim.

(Everything on screen is computed: the zeta function, its zeros and Riemann's formula.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#math #riemannhypothesis #primenumbers #mathematics #science
```

Exactly five hashtags.

## YouTube

Title:

```
The Riemann Hypothesis in 30 Seconds (a Proof Was Just Announced) #shorts
```

Description:

```
What is the Riemann Hypothesis, and why does it matter? On Nov 20, 2026, a proof was announced; it is still being checked.

Top: the number of primes up to x (amber) and Riemann's explicit formula built from the zeta zeros found so far (cyan). Bottom: a point climbs the critical line while the zeta function's output is traced; each time it passes through 0, that zero's wave is added. 563 zeros (up to height 892.5) rebuild every prime up to 50; from 10 zeros on, rounding the curve already gives every count exactly.

The hypothesis (1859): every such zero lies on that one line. Checked for the first 12,363,153,437,138 zeros (Platt and Trudgian, 2021). A proof carries the $1,000,000 Clay Millennium Prize.

The hum is the first eight zeros played as pitches.

#math #riemannhypothesis #primes #shorts
```

## Pinned comment

```
Two things people ask. (1) Will a proof break encryption? No. Encryption like RSA is safe because factoring huge numbers is hard, and a proof of the Riemann Hypothesis doesn't make factoring easier. It confirms what mathematicians already assumed about how primes are spread. (2) What does it actually say about primes? Each zero is a wave in the prime count, and "on the line" means every wave grows only like the square root of x. So the count of primes below x stays as close to Riemann's formula as it possibly can. Example: below 1,000,000,000 there are 50,847,534 primes. Riemann's smooth formula, with no zeros at all, says 50,847,455. Off by 79.
```

## Audio

Keep the built-in track: a plink for every prime as the number line is swept, a bell for each of the first three
zeros, a tick for every zero after that, the hum of the first eight zeros as pitches, a bell and impact when the
primes appear, a chime for the hypothesis.

## Replies to expect

* "Is it really solved?" → a proof was announced on Nov 20, 2026 and a computer checked it; mathematicians are still
  reading it. The reel explains the problem, not the proof.
* "Did AI solve it?" → the announced proof was written with an AI system; whether it stands is for mathematicians.
* "Does this break Bitcoin / banking?" → no (pinned comment).
* "Why only up to 50?" → so the steps can be seen; the same formula works for any x, it just needs more zeros.
