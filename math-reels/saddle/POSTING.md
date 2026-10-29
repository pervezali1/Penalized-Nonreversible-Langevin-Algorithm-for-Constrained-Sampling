# Posting kit — *The saddle.* (white style, one ball, the maths on screen, 40 s)

## Files

* Reel: `output/saddle_reel.mp4` — 40.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (gentle track).
* Cover: `output/cover.png` — step 20: the ball tipping off the pass, the maths card showing the offset multiplying.

## What's on screen (all of it is real)

The two-valley hill with no tilt, f(u, v) = 2.2·((u/2)² − 1)² + 1.8v²: two equal valleys with bottoms at
0 and a pass between them at height 2.2 whose top is a true saddle point (slope zero in every
direction; curvature −2.2 along the pass, +3.6 across it). Rendered by `../valleys/hillgen.py` in the
reel-23 look on the faint tinted grid.

One ball, dropped 0.01 off the line through the pass, up the back wall at height 3.66. The usual rule,
step = slope × 0.1, 40 steps of 0.5 s. Two readouts: height, and the distance from the saddle line
(the number the height hides). A card at the bottom of the frame carries the maths of each stage:

| Time | Card | What it says |
| --- | --- | --- |
| 3–9 s | The hill, and why its pass is a saddle | f, both slopes zero at (0, 0), ∂²f/∂u² = −2.2 (down), ∂²f/∂v² = +3.6 (up): one negative curvature makes a saddle |
| 10–13 s | The rule | u → u − 0.1·∂f/∂u, v → v − 0.1·∂f/∂v; across the pass v → 0.64·v: 0.9 → 0.58 → 0.37 → 0.24 |
| 13–20 s | On the pass: the offset multiplies | ∂f/∂u = −2.2·u near the pass, so u → 1.22·u; a live line "step k: u = …, next 1.22 × … = …"; 0.01 reaches 0.5 after 20 steps; exactly 0 stays 0 forever |
| 20–30 s | Off the pass | the full slope 2.2·((u/2)² − 1)·u; a live line with u and f; the bottom at u = ±2, reached (below 0.01) at step 32 |
| 30–39 s | In n dimensions | a flat point needs all n slopes zero, a bottom all n curvatures positive; if each sign were a coin flip, one flat point in 2ⁿ is a bottom; n = 20: one in a million |

The run: on the pass by step 6 (height 2.20), 14 steps parked while the offset goes 0.01 → 0.027 (step
5) → 0.073 (10) → 0.20 (15) → 0.52 (20), off the pass at step 20, bottom (below 0.01) at step 32, height
0.000 after 40. The 1.22 is exact for small offsets: 1 − 0.1 × (−2.2). Doubling every 3.5 steps.

## Caption (paste as-is)

```
The saddle. Slope zero, and not a bottom.

Two valleys with a pass between them. The top of the pass is flat in every direction, so gradient descent sees slope zero there, exactly as it would at a bottom. But it isn't one: the ground curves down along the pass (curvature −2.2) and up across it (+3.6). A bottom needs both positive. One negative curvature makes a saddle.

One ball, dropped 0.01 off the line through the pass. Same rule as always, step = slope × 0.1.

It rolls down to the pass fast: across the pass the ground is steep and each step keeps only 64% of the distance. Then it sits at height 2.2 and looks parked. It isn't. Near the pass the slope along it is −2.2 × offset, so each step turns an offset u into 1.22 × u. The maths card does it live: 0.0100, 0.0122, 0.0149 … doubling every 3.5 steps. Fourteen steps parked, then at step 20 it tips over the edge, and by step 32 it is at the bottom of a valley. Height 0.000.

Only an offset of exactly zero would have stayed: 1.22 × 0 = 0 forever. Anything else leaves. In a network with millions of weights, exactly zero never happens, and the same counting argument says almost every flat point is a saddle: a bottom needs all n curvatures positive, which, if each sign were a coin flip, happens one time in 2ⁿ. For n = 20 that is one in a million.

So saddles don't trap training. They slow it: the long plateau in a loss curve where nothing seems to happen, while a tiny offset quietly multiplies, followed by the sudden drop.

(The hill is the real function 2.2·((u/2)² − 1)² + 1.8v², rendered from its actual height; every arrow is its true gradient, every step is exactly slope × 0.1, and every number on the maths card is computed from the run.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #gradientdescent #saddlepoint #neuralnetworks #ai #math #deeplearning #calculus
```

Short alternative:

```
A pass between two valleys: slope zero, not a bottom. Drop a ball 0.01 off the line and it looks parked for fourteen steps while its offset multiplies by 1.22 a step. Step 20: it tips off. Step 32: bottom. Only an offset of exactly zero would stay, and in millions of dimensions, exactly never happens. Saddles slow training down; they almost never stop it.

Follow for more.

#machinelearning #gradientdescent #saddlepoint #ai
```

## YouTube description

```
Two equal valleys with a pass between them. The top of the pass is a saddle point: both slopes zero, curvature −2.2 along the pass and +3.6 across it, height 2.2. One ball, dropped 0.01 off the line through the pass, with the usual rule, step = slope × 0.1, for 40 steps. This time the maths is on screen.

The rule: u → u − 0.1·∂f/∂u, v → v − 0.1·∂f/∂v. Across the pass ∂f/∂v = 3.6·v, so v → 0.64·v: the ball reaches the pass by step 6 (height 2.20).

On the pass: ∂f/∂u = −2.2·u, so u → u + 0.22·u = 1.22·u. The offset multiplies by 1.22 every step, doubling every 3.5: 0.01, 0.027 (step 5), 0.073 (10), 0.20 (15), 0.52 (20). The card shows the live value and the next one. Fourteen steps parked, then at step 20 the ball is off the pass.

Off the pass: the slope is the full 2.2·((u/2)² − 1)·u, no longer small. The ball reaches the bottom (below 0.01) at step 32 and ends at 0.000.

In n dimensions: a flat point needs all n slopes to be zero; a bottom needs all n curvatures positive. If each curvature's sign were a coin flip, a flat point would be a bottom one time in 2ⁿ: for n = 20, one in a million. Almost every flat point in a big network is a saddle, and a saddle holds only an offset of exactly zero. That is why saddles slow training down (the plateau in a loss curve) and almost never stop it (the drop that follows).

Honest note: the ball parked exactly on the line would stay forever, in the maths and in floating point; 0.0001 off would take about 43 steps instead of 20; the coin-flip argument is a heuristic, not a theorem, but it is the right order of magnitude.

Every number on screen is computed, not drawn: the hill is the real function 2.2·((u/2)² − 1)² + 1.8v², rendered from its actual height, every arrow is its true gradient, every step is exactly slope × 0.1, and the maths card is filled from the run.

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #gradientdescent #saddlepoint #neuralnetworks #ai #math #deeplearning #calculus
```

## Pinned comment

```
The whole reel in two lines of calculus: at a flat point the first derivatives are zero, and the second derivatives decide what it is. Along the pass f_uu = −2.2, so a step of size 0.1 multiplies the offset by 1 − 0.1 × (−2.2) = 1.22; across the pass f_vv = +3.6, so a step multiplies the distance across by 1 − 0.1 × 3.6 = 0.64. One direction grows, one shrinks: a saddle. At a bottom every direction shrinks.

Honesty check: the 1.22 is exact only near the pass (the slope is 2.2·((u/2)² − 1)·u, which is −2.2·u for small u); away from it the ball accelerates and then decelerates, which is why the fall takes 12 steps. The "one in 2ⁿ" line assumes independent coin-flip signs, which real networks don't quite have, but the conclusion (nearly all flat points are saddles) holds up in practice and in theory. Momentum (reel 25) and mini-batch noise (reels 27 and 29) both shorten the parked stretch.

Next, same white style: learning-rate decay. Start big, end small, and why that one schedule fixes three problems from this series.
```

## Audio

Keep the built-in track: the soft chord bed; one pluck per landing walking down the scale with the ball's
height: quick at first, the same quiet note repeated while it sits on the pass, then walking down again as
it falls; a low tone when it settles; bells at the headings. −16 LUFS, 40 s loop.

## Replies to expect

* "Why 1.22?" → 1 − step size × curvature along the pass = 1 − 0.1 × (−2.2). A negative curvature makes the factor bigger than 1, so the offset grows; a positive one (across the pass, 3.6) gives 0.64, so that distance shrinks.
* "What if it started exactly on the line?" → 1.22 × 0 = 0: it would sit on the pass forever. In floating point that is exactly what happens; in a real network an exact zero never occurs.
* "How long would 0.0001 take?" → about 43 steps to reach 0.5 instead of 20. Longer, not forever.
* "Is the 2ⁿ thing a real theorem?" → it's a heuristic (independent signs), not a theorem, but the conclusion that saddles vastly outnumber minima in high dimensions is well established.
* "Is this my loss plateau?" → very often: a long flat stretch followed by a sudden drop is the signature of creeping past a saddle. Momentum and noise shorten it.
```
