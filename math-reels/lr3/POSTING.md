# Posting kit — *The step size.* (white style, three balls)

## Files

* Reel: `output/lr3_reel.mp4` — 30.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (gentle track).
* Cover: `output/cover.png` — the two surviving balls mid-descent, the magenta zigzag of the one that left.

## What's on screen (all of it is real)

The same cut hill as reel 23 (f = 0.5u² + 1.8v² on axes rotated 32°, shaded cutaway, contour lines),
three balls, same rule, different step sizes. Each ball's step is exactly its step size times the
slope under it, computed in the bowl's own axes (u ← u(1 − 2·0.5·η), v ← v(1 − 2·1.8·η), the exact
gradient-descent update for this function). 24 steps every 0.62 s.

| Ball | Step size η | Start (u, v), height | Landings | What it does |
| --- | --- | --- | --- | --- |
| TOO SMALL (green) | 0.02 | (2.6, −0.95), 5.00 | … → **1.33** at step 24 | each step shrinks the distance by only 2% along the valley; still high up |
| JUST RIGHT (vermilion) | 0.09 | (2.85, −0.64), 4.80 | … → **0.044** at step 24 | the reel-23 path: long steps first, short steps last, settled |
| TOO BIG (magenta) | 0.9 | (3.0, 0.02), 4.50 | 0.049 → 0.019 → 0.091 → 0.46 → 2.29 → **off the hill** (11.5) | one huge step to the floor, then crosses the valley every step, 2.24× wider each time, out on step 6 |

Why the big one leaves: a gradient step multiplies the distance from the bottom along a direction of
curvature c by (1 − 2cη). Along the valley c = 0.5, so η = 0.9 gives a factor of 0.1: one step takes
it 90% of the way to the floor (4.50 → 0.049, lower than the small ball ever gets). Across the valley
c = 1.8, so the factor is 1 − 2·1.8·0.9 = −2.24: every step it lands on the other side of the floor,
2.24 times farther from it. Start a hair off the centre line (v = 0.02) and that sideways error goes
0.02 → −0.045 → 0.10 → −0.23 → 0.50 → −1.13 → 2.53, and at 2.53 the surface height is 11.5, twice
the rim. Six steps and it is gone. The cut plane sits at v = 0.6 (reel 23 had 0.22) so the crossing
at v = 0.50 is still on the drawn surface. The readouts hold each landing's height (the live height
would dip as the ball flies across the floor), and the TOO BIG column reads "left the hill" from step
6; the ball itself fades out during the slide that takes it off, and its trail stays.

## Caption (paste as-is)

```
Same hill as last time. Same rule: feel the slope, step downhill. Three balls, and the only thing that differs is how far each one steps: 0.02 times the slope, 0.09 times the slope, 0.9 times the slope.

Too small is safe and slow. Every step shrinks its distance to the bottom by 2%. After 24 steps it is still at height 1.3, less than halfway along the valley floor, and it would need about 84 more steps to get where the middle ball already is.

Just right takes long steps while it's steep and short steps when it's flat. Below height 0.1 by step 20, at 0.044 by step 24.

Too big is the one to watch. Its first step is enormous: 4.50 to 0.049, almost on the floor, lower than the small ball gets in 24 steps. For one step it looks like the winner. Then watch it. Every step after that crosses the valley floor to the other side, and every crossing is 2.2 times wider than the last: 0.019, 0.091, 0.46, 2.29, and on step 6 it leaves the hill. Not unlucky, inevitable: a step this big is fine along the valley and too big across it, and the across direction wins.

That number, the step size, is the learning rate. It is the one number you have to choose by hand when you train a neural network, and this is what the choice means: too small wastes time, too big flies off, and the right value depends on how steep your hill is in its steepest direction.

(Every arrow is the real negative gradient of 0.5u² + 1.8v² on rotated axes, scaled by that ball's step size; every step is exactly that. Heights: 5.00 → 1.33, 4.80 → 0.044, 4.50 → 0.049 → 0.019 → 0.091 → 0.46 → 2.29 → 11.5 and off the hill.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #gradientdescent #learningrate #neuralnetworks #ai #math #deeplearning
```

Short alternative:

```
Same hill, same rule, three step sizes. 0.02 × slope: still a quarter of the way down after 24 steps. 0.09 × slope: settled by step 20. 0.9 × slope: almost at the bottom in one step, then it crosses the floor every step, wider each time, and leaves the hill on step 6. That number is the learning rate.

Follow for more.

#machinelearning #gradientdescent #learningrate #ai
```

## Pinned comment

```
Too small wastes time. Too big flies off.

The maths: on a bowl with curvature c along a direction, a gradient step multiplies your distance from the bottom along that direction by (1 − 2cη). Along this valley c = 0.5, so η = 0.02 keeps 98% of the distance each step (that's why the green ball crawls), η = 0.09 keeps 91%, and η = 0.9 keeps only 10%: that is the huge first step. Across the valley c = 1.8, and there η = 0.9 gives 1 − 2·1.8·0.9 = −2.24: over the floor and 2.24 times farther away, every step. That is why the big ball's sideways wobble goes from invisible (0.02) to off the hill (2.5) in six steps. The stable range for a direction is 0 < η < 1/c, so this hill allows up to 2 along the valley but only 0.56 across it, and one step size has to serve both: 0.9 is a great step size for the valley and a fatal one for the walls. Real loss surfaces have different curvatures in every direction, so a single learning rate is always too big for the steep ones or too small for the flat ones; that is the whole reason momentum and Adam exist (reels 4, 7 and 8).

Honesty check: the big ball starts 0.02 off the valley's centre line, on purpose. Exactly on the line it would never feel the steep direction and would settle in three steps (factor 0.1) and look like the best of the three. In a real network you are never exactly on the line, and the wobble you can't see at step 1 is the one that throws you out at step 6.

Next, same white style: momentum, the ball that remembers which way it was going.
```

## Audio

Keep the built-in track: the soft chord bed from reel 23; at every landing one sound per ball. Just
right: a plucked note walking down a scale as its height falls. Too small: a quiet tick that barely
moves down the scale. Too big: a round bouncy tone at each of its five landings, climbing as the crossings
widen, a rising whoosh as it leaves the hill, silence after. Bells at the headings. −16 LUFS.

## Replies to expect

* "The big one was winning, why did it lose?" → along the valley 0.9 is a great step size (it keeps only 10% of the distance), so step 1 is huge; across the valley it's above the limit (0.56), so a sideways error of 0.02 grows 2.24× per step and throws it out on step 6. One step size, two curvatures.
* "Isn't the start cherry-picked?" → it starts 0.02 off the centre line so the steep direction is in play at all; any offset, however small, ends the same way, it just takes a few more steps (0.002 would take about nine).
* "How do you choose it in practice?" → try a few on a log scale, watch the loss, pick the largest that still decreases smoothly; or use a schedule that starts big and shrinks. Adam makes the choice less sensitive but doesn't remove it.
* "The small one would get there eventually" → yes: about 108 steps to reach where the middle ball is at step 24. Eventually is the problem.
* "What about momentum?" → next reel.
