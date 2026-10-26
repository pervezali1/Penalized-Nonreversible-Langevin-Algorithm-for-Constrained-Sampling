# Posting kit — *The step size.* (white style, three balls)

## Files

* Reel: `output/lr3_reel.mp4` — 30.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (gentle track).
* Cover: `output/cover.png` — the three balls mid-descent, the magenta zigzag climbing the far wall.

## What's on screen (all of it is real)

The same cut hill as reel 23 (f = 0.5u² + 1.8v² on axes rotated 32°, shaded cutaway, contour lines),
three balls, same rule, different step sizes. Each ball's step is exactly its step size times the
slope under it, computed in the bowl's own axes (u ← u(1 − 2·0.5·η), v ← v(1 − 2·1.8·η), the exact
gradient-descent update for this function). 24 steps every 0.62 s.

| Ball | Step size η | Start (u, v), height | Landings | What it does |
| --- | --- | --- | --- | --- |
| TOO SMALL (green) | 0.03 | (2.6, −0.95), 5.00 | 4.47 → 4.02 → … → **0.79** at step 24 | keeps 97% of its distance along the valley each step; still high up |
| JUST RIGHT (vermilion) | 0.2 | (2.85, −0.64), 4.80 | 2.66 → 1.67 → 1.07 → … → **<0.001** at step 24 | long steps first, short steps last; below 0.1 by step 9, below 0.01 by step 14 |
| TOO BIG (magenta) | 0.6 | (2.9, −0.42), 4.52 | 1.10 → 0.68 → 0.79 → 1.04 → 1.40 → 1.88 → 2.54 → 3.41 → 4.59 → **off the hill** (6.18) | drops fast, then crosses the valley floor every step, each crossing 16% wider; out on step 10 |

Why the big one leaves: a gradient step multiplies the distance from the bottom along a direction of
curvature c by (1 − 2cη). Along the valley c = 0.5, so η = 0.6 keeps 40% of the distance each step:
its first two steps are the fastest of the three (4.52 → 1.10 → 0.68). Across the valley c = 1.8, so
the factor is 1 − 2·1.8·0.6 = −1.16: every step it lands on the other side of the floor, 16% farther
out. The sideways distance goes 0.42 → 0.49 → 0.57 → … → 1.60 → 1.85, and at 1.85 the surface
height is 6.18, above the rim (5.6). Ten steps and it is gone. The limit for this hill is 1/1.8 =
0.556: 0.54 would zigzag inward (the previous version of this reel), 0.6 zigzags outward.

Presentation notes. The cut plane sits at v = 1.25 (reel 23 had 0.22) so the widening crossings stay
on the drawn surface as long as possible; the ninth landing (sideways 1.60, height 4.59) is in the
cut-away region in front of the cut plane, where the surface is not drawn, so the ball fades out on
the slide to it while its readout shows the true 4.59; the tenth landing is above the rim and the
readout says "left the hill" from there. The readouts hold each landing's height (the live height
would dip as the ball flies across the floor). The zigzag ball's pill rides to its upper right so it
never covers the just-right ball's.

## Caption (paste as-is)

```
Same hill as last time. Same rule: feel the slope, step downhill. Three balls, and the only thing that differs is how far each one steps: 0.03 times the slope, 0.2 times the slope, 0.6 times the slope.

Too small is safe and slow. Every step shrinks its distance to the bottom by 3%. After 24 steps it is still at height 0.79, and it would need about 96 steps to get below 0.01, where the middle ball is at step 14.

Just right takes long steps while it's steep and short steps when it's flat. Below 0.1 by step 9, below 0.01 by step 14, and effectively at the bottom by step 24.

Too big is the one to watch. For two steps it is the fastest ball on the hill: 4.52 to 1.10 to 0.68. Then look at how it moves. Every step jumps clean across the valley floor to the other side, and each crossing is 16% wider than the last. 0.79, 1.04, 1.40, 1.88, 2.54, 3.41, 4.59, and on step 10 it leaves the hill. It was never going to arrive.

That number, the step size, is the learning rate. It is the one number you have to choose by hand when you train a neural network, and this is what the choice means: too small wastes time, too big never arrives, and the limit is set by the steepest direction of your hill, not the one you are walking along.

(Every arrow is the real negative gradient of 0.5u² + 1.8v² on rotated axes, scaled by that ball's step size; every step is exactly that. Heights after 24 steps: 0.79, under 0.001, gone at step 10.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #gradientdescent #learningrate #neuralnetworks #ai #math #deeplearning
```

Short alternative:

```
Same hill, same rule, three step sizes. 0.03 × slope: still at 0.79 after 24 steps. 0.2 × slope: at the bottom by step 14. 0.6 × slope: fastest for two steps, then it crosses the valley floor every step, 16% wider each time, and leaves the hill on step 10. That number is the learning rate.

Follow for more.

#machinelearning #gradientdescent #learningrate #ai
```

## Pinned comment

```
Too small wastes time. Too big never arrives.

The maths: on a bowl with curvature c along a direction, a gradient step multiplies your distance from the bottom along that direction by (1 − 2cη). Along this valley c = 0.5, so η = 0.03 keeps 97% of the distance each step (that's why the green ball crawls), η = 0.2 keeps 60%, and η = 0.6 keeps 40%: that's its fast start. Across the valley c = 1.8, and there η = 0.6 gives 1 − 2·1.8·0.6 = −1.16: over the floor to the other side, 16% farther out, every step. Negative means it alternates sides, that's the zigzag; bigger than 1 in size means the zigzag grows, and that's why it leaves. The stable range for a direction is 0 < η < 1/c, so this hill allows up to 2 along the valley but only 0.556 across it: 0.6 is a fine step size for the valley and a fatal one for the walls. One step size has to serve both, and the steep direction sets the limit. Real loss surfaces have different curvatures in every direction, so a single learning rate is always too big for the steep ones or too small for the flat ones; that is the whole reason momentum and Adam exist (reels 4, 7 and 8).

Honesty check: the big ball starts a little off the valley's centre line (sideways 0.42), on purpose. Exactly on the line it would never feel the steep direction and would settle in a few steps. In a real network you are never exactly on the line. And its ninth landing is in the part of the hill that is cut away in the drawing, so the ball fades out one step before its height passes the rim; the readout shows the real height (4.59) for that step, and "left the hill" only once it is truly above the rim.

Next, same white style: momentum, the ball that remembers which way it was going.
```

## Audio

Keep the built-in track: the soft chord bed from reel 23; at every landing one sound per ball. Just
right: a plucked note walking down a scale as its height falls. Too small: a quiet tick that barely
moves down the scale. Too big: a round bouncy tone at each landing, one per crossing of the floor,
climbing back up the scale as the crossings widen, then a rising whoosh as it leaves the drawn hill,
silence after. Bells at the headings. −16 LUFS.

## Replies to expect

* "The big one was winning, why did it lose?" → along the valley 0.6 is a great step size (it keeps only 40% of the distance), so its first steps are huge; across the valley it's above the limit (0.556), so each crossing of the floor is 16% wider than the last. One step size, two curvatures.
* "What if it were 0.5?" → the crossings would shrink 20% per step instead of growing 16%; it would zigzag in and settle. 0.54 (the previous cut of this reel) zigzags in so slowly it is still at 0.096 after 24 steps.
* "Isn't the start cherry-picked?" → it starts 0.42 off the centre line so the zigzag is visible from the first step; any offset, however small, ends the same way, it just takes longer (0.04 would take about 16 more steps).
* "How do you choose it in practice?" → try a few on a log scale, watch the loss, pick the largest that still decreases smoothly; or use a schedule that starts big and shrinks. Adam makes the choice less sensitive but doesn't remove it.
* "The small one would get there eventually" → yes: about 96 steps to get below 0.01. Eventually is the problem.
* "What about momentum?" → next reel.
