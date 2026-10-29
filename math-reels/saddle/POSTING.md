# Posting kit — *The saddle.* (white style, the pass between two valleys)

## Files

* Reel: `output/saddle_reel.mp4` — 30.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (gentle track).
* Cover: `output/cover.png` — step 20: A sitting on the pass, B just tipping off it, C already down in a valley.

## What's on screen (all of it is real)

The two-valley hill with no tilt, f(u, v) = 2.2·((u/2)² − 1)² + 1.8v²: two equal valleys with bottoms at
0 and, between them, a pass at height 2.2 whose top is a true saddle point: slope zero in every
direction, a minimum across the pass and a maximum along it. Rendered by `../valleys/hillgen.py` in the
reel-23 look on the faint tinted grid.

Three balls, same rule (step = slope × 0.1), 40 steps of 0.42 s, all dropped up the back wall at
height 3.66:

| Ball | Start | What happens | After 40 steps |
| --- | --- | --- | --- |
| A (vermilion) | exactly on the saddle line (u = 0) | rolls down to the pass by step 6 and stops: its slope along the pass is exactly zero | **2.20**, on the pass |
| B (blue) | 0.01 off the line | its offset grows 22% a step (doubling every 3.5), so it creeps along the flat top: 0.01 → 0.027 (step 5) → 0.073 (10) → 0.20 (15) → 0.52 (20), then slides into a valley | **0.000** |
| C (magenta) | 0.01 off, slope measured on 4 of the 1,000 tilted examples of reel 29 | the batch noise throws it off the pass at step 3; it rattles down into a valley and keeps rattling | **0.188** |

The "off the line" readout under each column is the distance from the saddle line, |u|, which is the
number the height hides: B's height sits at 2.19 to 2.16 for fifteen steps while its offset is
multiplying. The grey arrow beside C's magenta arrow is its true slope. The 22% per step is
1 − 0.1 × f_uu(0) = 1 + 0.1 × 2.2, exact for small offsets.

Honesty: A's fate needs u to be exactly zero, which in floating point it is (the slope along the pass is
computed as exactly 0 there), and which in a real network never happens; that is the closing line. C's
noisy run is seed 2 of the batch-of-4 data; most seeds leave the pass within 2 to 7 steps (12 of 12
tested), and this one was chosen among them for staying inside the drawn hill.

## Caption (paste as-is)

```
The saddle. Slope zero, and not a bottom.

Two valleys with a pass between them. On top of the pass the ground is flat in every direction: it's the lowest point along one axis and the highest along the other. Three balls, same rule, step 0.1 times the slope.

A starts exactly on the line through the pass. It rolls down to the top of the pass and stops, at height 2.2, forever. The slope there is zero, and plain descent cannot tell a pass from a bottom.

B starts 0.01 off that line. You can't see the difference. For fifteen steps it looks exactly like A, parked on the pass at the same height. But its tiny offset is growing 22% every step, doubling every 3.5, and at step 20 it tips over the edge and slides to the bottom of a valley. Height 0.

C also starts 0.01 off, but measures its slope on 4 examples instead of all 1,000. The noise throws it off the pass at step 3. It never even noticed the flat ground.

This is the other half of the local-minimum story. Real training surfaces, with millions of weights, are full of saddles and almost empty of true traps. And a saddle only holds you if you land on it exactly, which in millions of dimensions never happens. What a saddle does is waste your time: long flat stretches where the error barely moves while a tiny offset quietly grows. Momentum, noise and a decent step size all shorten the wait.

(The hill is the real function 2.2·((u/2)² − 1)² + 1.8v², rendered from its actual height; every arrow is its true gradient and every step is exactly slope × 0.1. C's noise is the real mini-batch noise of reel 29's data. Heights after 40 steps: 2.20, 0.000, 0.188.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #gradientdescent #saddlepoint #neuralnetworks #ai #math #deeplearning
```

Short alternative:

```
A pass between two valleys: slope zero, not a bottom. Place a ball exactly on it and it sits there forever. Place one 0.01 off and it looks parked for fifteen steps, then tips over at step 20. Give one a noisy slope and it's gone by step 3. Saddles slow training down. They almost never stop it.

Follow for more.

#machinelearning #gradientdescent #saddlepoint #ai
```

## YouTube description

```
Two equal valleys with a pass between them. The top of the pass is a saddle point: slope zero in every direction, the lowest point along one axis and the highest along the other, height 2.2.

Three balls, same rule, step = slope × 0.1, dropped from height 3.66.

A starts exactly on the line through the pass. It reaches the top of the pass by step 6 and stops there: the slope is exactly zero, and plain descent cannot tell a pass from a bottom. 2.20 after 40 steps.

B starts 0.01 off the line. For fifteen steps it looks parked next to A at nearly the same height, but its offset grows 22% per step (1 + 0.1 × the pass's curvature of 2.2), doubling every 3.5 steps: 0.01, 0.027, 0.073, 0.20, 0.52. At step 20 it tips off the pass and slides to 0.000.

C starts 0.01 off too but measures its slope on 4 of 1,000 examples (the tilted data of the previous reel). The noise throws it off the pass at step 3 and it rattles down into a valley, ending at 0.188 and still rattling.

Why this matters: in millions of dimensions, the flat points of a loss surface are overwhelmingly saddles, not minima, because a true minimum needs the ground to curve up in every single direction at once. A saddle only holds you if you sit on it exactly, which never happens; what it does is slow you down, sometimes for a long time, while a tiny offset grows exponentially. That is the long plateau in a training curve followed by a sudden drop. Momentum (reel 25), noise (reels 27 and 29) and a healthy step size (reel 24) all shorten it.

Every number on screen is computed, not drawn: the hill is the real function 2.2·((u/2)² − 1)² + 1.8v², rendered from its actual height, every arrow is its true gradient, every step is exactly slope × 0.1, and C's noise is real mini-batch sampling noise.

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #gradientdescent #saddlepoint #neuralnetworks #ai #math #deeplearning
```

## Pinned comment

```
Why saddles and not traps: a point with zero slope is a minimum only if the surface curves upward in every direction. With n weights that is n coin flips all coming up the same way, so in a big network almost every flat point is a saddle, curving down in at least one direction. Gradient descent on the exact slope slows to a crawl near one (B's fifteen parked steps) and then escapes along the downhill direction, with the offset growing by a fixed factor per step: here 1.22, from 1 − step size × curvature with curvature −2.2.

Honesty check: A stays only because its offset is exactly zero, in the maths and in the floating point. Any offset at all, however small, leaves eventually; 0.01 took 20 steps, 0.0001 would take about 37. And C's run is one noisy seed; all twelve seeds tested left the pass within 2 to 7 steps, this one was picked for staying inside the drawing.

Next, same white style: learning-rate decay. Start big, end small, and why that one schedule fixes three of the problems in this series.
```

## Audio

Keep the built-in track: the soft chord bed; one pluck per ball per landing, walking down the scale with
height. A reaches the pass and repeats the same quiet note forever; B holds one note through its creep and
then walks down; C's note jumps with its noisy steps and walks down early, with a quiet tick on each noisy
measurement. A low tone when the run ends, bells at the headings. −16 LUFS.

## Replies to expect

* "So A is stuck forever?" → yes, and only because its offset is exactly zero. In floating point that happens here because the slope along the pass is computed as exactly 0 at u = 0; in a real network it never does.
* "How long would 0.0001 off take?" → the offset grows 22% per step, so about 37 steps to reach 0.5 instead of 20. Still not forever.
* "Does momentum help?" → yes: a ball arriving with speed rolls across the pass instead of parking on it, and once the offset starts growing momentum compounds it.
* "Is this the plateau in my loss curve?" → often, yes: a long flat stretch followed by a sudden drop is the signature of creeping past a saddle.
* "Why does C keep rattling at the bottom?" → mini-batch noise doesn't switch off at the bottom (reel 27). Learning-rate decay is what calms it; next reel.
```
