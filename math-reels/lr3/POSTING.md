# Posting kit — *The step size.* (white style, three balls)

## Files

* Reel: `output/lr3_reel.mp4` — 30.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (gentle track).
* Cover: `output/cover.png` — the two surviving balls mid-descent, the magenta line of the one that left spanning the whole valley.

## What's on screen (all of it is real)

The same cut hill as reel 23 (f = 0.5u² + 1.8v² on axes rotated 32°, shaded cutaway, contour lines),
three balls, same rule, different step sizes, each ten times the last. Each ball's step is exactly its
step size times the slope under it, computed in the bowl's own axes (u ← u(1 − 2·0.5·η),
v ← v(1 − 2·1.8·η), the exact gradient-descent update for this function). 24 steps every 0.62 s.

| Ball | Step size η | Start (u, v), height | Landings | What it does |
| --- | --- | --- | --- | --- |
| TOO SMALL (green) | 0.02 | (2.6, −0.95), 5.00 | 4.65 → 4.32 → … → **1.33** at step 24 | keeps 98% of its distance along the valley each step; still high up |
| JUST RIGHT (vermilion) | 0.2 | (2.85, −0.64), 4.80 | 2.66 → 1.67 → 1.07 → … → **<0.001** at step 24 | long steps first, short steps last; below 0.1 by step 9, below 0.01 by step 14 |
| TOO BIG (magenta) | 2 | (3.0, −0.002), 4.50 | 4.50 → 4.51 → 4.91 → **off the hill** (20.2) | jumps the whole valley to the mirror point at the same height, three times, then the sideways wobble throws it out on step 4 |

Why the big one does that: a gradient step multiplies the distance from the bottom along a direction
of curvature c by (1 − 2cη). Along the valley c = 0.5, so η = 2 gives exactly −1: every step lands at
the mirror point, same distance from the bottom, same height, other side (that is the knife edge
between creeping in and flying out). Across the valley c = 1.8, so the factor is 1 − 2·1.8·2 = −6.2:
any sideways error is 6.2 times bigger after every step. The ball starts 0.002 off the centre line,
and that goes 0.002 → 0.012 → 0.077 → 0.48 → 2.96, where the surface height is 20.2, far above the
rim (5.6). So: 4.50, 4.50, 4.51, 4.91, gone. On the exact centre line it would bounce between the two
ends forever at 4.50.

Presentation notes. The readouts hold each landing's height (the live height would dip as the ball
flies across the floor); the TOO BIG column reads "left the hill" from step 4, the ball fades out
early on that slide (its landing is far above the frame), its trail stays. The cut plane sits at
v = 0.7 (reel 23 had 0.22). The big ball's pill rides to its upper right so it never covers the
just-right ball's. The cue lines are generated from the run (the knife-edge wording appears only when
the along-valley factor is exactly −1; the just-right line reports whether it settled within 24 steps).

## Caption (paste as-is)

```
Same hill as last time. Same rule: feel the slope, step downhill. Three balls, and the only thing that differs is how far each one steps: 0.02 times the slope, 0.2 times the slope, 2 times the slope. Each one ten times the last.

Too small is safe and slow. Every step shrinks its distance to the bottom by 2%. After 24 steps it is still at height 1.33, and it would need about 145 steps to get below 0.01, where the middle ball is at step 14.

Just right takes long steps while it's steep and short steps when it's flat. Below 0.1 by step 9, below 0.01 by step 14, and effectively at the bottom by step 24.

Too big is the one to watch. Its first step is so long it jumps the entire valley and lands on the far wall at exactly the same height, 4.50. Then it jumps back. 4.50 again. Then across again, 4.51. It is never getting closer, and then it gets worse: it started a hair off the centre line, that hair is multiplied by 6.2 on every step, and on step 4 it leaves the hill. Four steps, gone.

That number, the step size, is the learning rate. It is the one number you have to choose by hand when you train a neural network, and this is what the choice means: too small wastes time, too big never arrives, and the limit is set by the steepest direction of your hill, not the one you are walking along.

(Every arrow is the real negative gradient of 0.5u² + 1.8v² on rotated axes, scaled by that ball's step size; every step is exactly that. Heights after 24 steps: 1.33, under 0.001, gone at step 4.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #gradientdescent #learningrate #neuralnetworks #ai #math #deeplearning
```

Short alternative:

```
Same hill, same rule, three step sizes, each ten times the last. 0.02 × slope: still at 1.33 after 24 steps. 0.2 × slope: at the bottom by step 14. 2 × slope: jumps the whole valley to the same height on the far side, three times, then leaves the hill on step 4. That number is the learning rate.

Follow for more.

#machinelearning #gradientdescent #learningrate #ai
```

## YouTube description

```
Three balls on the same hill, following the same rule: feel the slope, take a step downhill. The only difference is the step size: 0.02 times the slope, 0.2 times the slope, 2 times the slope. Each one ten times the last.

0.02 crawls. It keeps 98% of its distance to the bottom on every step and is still at height 1.33 after 24 steps. It would need about 145 steps to get below 0.01.

0.2 settles. Long steps while the hill is steep, short steps when it flattens. Below 0.1 by step 9, below 0.01 by step 14, under 0.001 by step 24.

2 jumps the entire valley. Its first step lands on the far wall at exactly the height it started from, 4.50. The next step brings it back, 4.50 again. Then 4.51. It is not getting closer, and then the small sideways error it started with, multiplied by 6.2 on every step, throws it off the hill on step 4.

Why: a gradient step multiplies your distance from the bottom along a direction of curvature c by (1 − 2cη). Along this valley c = 0.5, so a step size of 2 gives exactly −1: the mirror point, every time. Across the valley c = 1.8, so 2 gives −6.2: any sideways wobble grows 6.2× per step. The stable range is 0 < η < 1/c, so this hill allows up to 2 along the valley but only 0.556 across it. One step size has to serve both directions, and the steepest one sets the limit. That is the whole learning-rate problem, and the reason momentum and Adam exist.

Every number on screen is computed, not drawn. The surface is the real function 0.5u² + 1.8v² on rotated axes, every arrow is its actual negative gradient scaled by that ball's step size, and every step is exactly that. The big ball starts 0.002 off the centre line; on the exact line it would bounce between the two ends forever.

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #gradientdescent #learningrate #neuralnetworks #ai #math #deeplearning
```

## Pinned comment

```
Too small wastes time. Too big never arrives.

The maths: on a bowl with curvature c along a direction, a gradient step multiplies your distance from the bottom along that direction by (1 − 2cη). Along this valley c = 0.5, so η = 0.02 keeps 98% of the distance each step (that's why the green ball crawls), η = 0.2 keeps 60%, and η = 2 gives exactly −1: same distance, other side, forever. That is the knife edge; 1.9 would creep in, 2.1 would fly out along the valley. Across the valley c = 1.8, and there η = 2 gives 1 − 2·1.8·2 = −6.2: any sideways error is 6.2 times bigger after every step, which is what throws the ball out on step 4. The stable range for a direction is 0 < η < 1/c, so this hill allows up to 2 along the valley but only 0.556 across it. One step size has to serve both, and the steep direction sets the limit. Real loss surfaces have different curvatures in every direction, so a single learning rate is always too big for the steep ones or too small for the flat ones; that is the whole reason momentum and Adam exist (reels 4, 7 and 8).

Honesty check: the big ball starts 0.002 off the valley's centre line, on purpose. Exactly on the line it would bounce between the two ends at 4.50 forever and never leave. In a real network you are never exactly on the line, so this is the realistic version.

Next, same white style: momentum, the ball that remembers which way it was going.
```

## Audio

Keep the built-in track: the soft chord bed from reel 23; at every landing one sound per ball. Just
right: a plucked note walking down a scale as its height falls. Too small: a quiet tick that barely
moves down the scale. Too big: a round bouncy tone on the same pitch at each of its three landings,
because it lands at the same height each time, then a rising whoosh as it leaves, silence after.
Bells at the headings. −16 LUFS.

## Replies to expect

* "Why does it land at exactly the same height?" → along the valley the slope is 2·0.5·u = u, so a step of 2 × slope moves it by 2u, from u to exactly −u. Same distance from the bottom, same height, other side.
* "Then why does it leave?" → the sideways direction is steeper (curvature 1.8), and there a step of 2 × slope multiplies the sideways error by −6.2. It started 0.002 off the line: 0.012, 0.077, 0.48, 2.96, gone.
* "Isn't 2 cherry-picked?" → yes, deliberately: it's the exact boundary along the valley, so you can see "never arrives" in its purest form before the explosion. 0.6 (an earlier cut of this reel) zigzags out in 10 steps instead.
* "How do you choose it in practice?" → try a few on a log scale, watch the loss, pick the largest that still decreases smoothly; or use a schedule that starts big and shrinks. Adam makes the choice less sensitive but doesn't remove it.
* "The small one would get there eventually" → yes: about 145 steps to get below 0.01. Eventually is the problem.
* "What about momentum?" → next reel.
