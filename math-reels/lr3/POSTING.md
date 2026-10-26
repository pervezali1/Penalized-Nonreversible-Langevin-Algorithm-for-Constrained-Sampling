# Posting kit — *The step size.* (white style, three balls)

## Files

* Reel: `output/lr3_reel.mp4` — 30.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (gentle track).
* Cover: `output/cover.png` — the three balls mid-descent, the magenta zigzag crossing the valley floor.

## What's on screen (all of it is real)

The same cut hill as reel 23 (f = 0.5u² + 1.8v² on axes rotated 32°, shaded cutaway, contour lines),
three balls, same rule, different step sizes. Each ball's step is exactly its step size times the
slope under it, computed in the bowl's own axes (u ← u(1 − 2·0.5·η), v ← v(1 − 2·1.8·η), the exact
gradient-descent update for this function). 24 steps every 0.62 s.

| Ball | Step size η | Start (u, v), height | Landings | What it does |
| --- | --- | --- | --- | --- |
| TOO SMALL (green) | 0.02 | (2.6, −0.95), 5.00 | … → **1.33** at step 24 | each step shrinks the distance by only 2% along the valley; still high up |
| JUST RIGHT (vermilion) | 0.09 | (2.85, −0.64), 4.80 | … → **0.044** at step 24 | the reel-23 path: long steps first, short steps last, settled |
| TOO BIG (magenta) | 0.54 | (2.7, −0.92), 5.17 | 2.13 → 1.37 → 1.11 → 0.97 → … → **0.096** at step 24 | drops fast, then crosses the valley floor on every step, each crossing 6% narrower than the last; still zigzagging at the end |

Why the big one zigzags: a gradient step multiplies the distance from the bottom along a direction of
curvature c by (1 − 2cη). Along the valley c = 0.5, so η = 0.54 keeps 46% of the distance each step:
fast. Across the valley c = 1.8, so the factor is 1 − 2·1.8·0.54 = −0.944: every step it lands on the
other side of the floor, 94.4% as far out as before. That is a zigzag that converges, but slowly: the
sideways distance goes 0.92 → 0.87 → 0.82 → … → 0.23 over 24 steps, and the height from 5.17 to
0.096. It is ahead of the just-right ball for 13 steps (its first landing is 2.13 against 3.70) and is
passed at step 14 (0.29 against 0.30). It would need 31 steps to get where just right is at 24. Any
step size above 0.556 would make the crossings wider instead of narrower and throw it off the hill.
The cut plane sits at v = 0.95 (reel 23 had 0.22) so the near-side crossings stay on the drawn
surface. The readouts hold each landing's height (the live height would dip as the ball flies across
the floor). Pills above each ball carry its step size; the zigzag ball's pill rides to its upper
right so it never covers the just-right ball's.

## Caption (paste as-is)

```
Same hill as last time. Same rule: feel the slope, step downhill. Three balls, and the only thing that differs is how far each one steps: 0.02 times the slope, 0.09 times the slope, 0.54 times the slope.

Too small is safe and slow. Every step shrinks its distance to the bottom by 2%. After 24 steps it is still at height 1.3, less than halfway along the valley floor, and it would need about 84 more steps to get where the middle ball already is.

Just right takes long steps while it's steep and short steps when it's flat. Below height 0.1 by step 20, at 0.044 by step 24.

Too big is the one to watch. It drops fast: 5.17 to 2.13 in one step, and it stays ahead of the middle ball for 13 steps. But look at how it moves. Every step jumps clean across the valley floor to the other side, and each crossing is only 6% narrower than the last. Most of every step is spent going sideways. The middle ball passes it at step 14, and after 24 steps it is still zigzagging, at 0.096, where the middle ball was at step 20. A little bigger still, and the crossings would get wider instead of narrower, and it would leave the hill.

That number, the step size, is the learning rate. It is the one number you have to choose by hand when you train a neural network, and this is what the choice means: too small wastes time, too big wastes most of every step, and the right value depends on the steepest direction of your hill, not the one you are walking along.

(Every arrow is the real negative gradient of 0.5u² + 1.8v² on rotated axes, scaled by that ball's step size; every step is exactly that. Heights after 24 steps: 1.33, 0.044, 0.096.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #gradientdescent #learningrate #neuralnetworks #ai #math #deeplearning
```

Short alternative:

```
Same hill, same rule, three step sizes. 0.02 × slope: still a quarter of the way down after 24 steps. 0.09 × slope: settled by step 20. 0.54 × slope: fast at first, then it crosses the valley floor every single step, each crossing only 6% narrower than the last, and it is still zigzagging at the end. That number is the learning rate.

Follow for more.

#machinelearning #gradientdescent #learningrate #ai
```

## Pinned comment

```
Too small crawls. Too big zigzags. Just right settles.

The maths: on a bowl with curvature c along a direction, a gradient step multiplies your distance from the bottom along that direction by (1 − 2cη). Along this valley c = 0.5, so η = 0.02 keeps 98% of the distance each step (that's why the green ball crawls), η = 0.09 keeps 91%, and η = 0.54 keeps 46%: that's its fast start. Across the valley c = 1.8, and there η = 0.54 gives 1 − 2·1.8·0.54 = −0.944: over the floor to the other side, 94.4% as far out, every step. Negative means it alternates sides, that's the zigzag; 0.944 means it does shrink, just slowly, and that's why it ends at 0.096 instead of 0.044. The stable range for a direction is 0 < η < 1/c, so this hill allows up to 2 along the valley but only 0.56 across it. One step size has to serve both, and the steep direction sets the limit. Real loss surfaces have different curvatures in every direction, so a single learning rate is always too big for the steep ones or too small for the flat ones; that is the whole reason momentum and Adam exist (reels 4, 7 and 8).

Honesty check: the big ball starts further up the valley wall than the others (sideways 0.92, close to the most this hill allows under its rim), so the zigzag is wide enough to see; the maths is the same from any start. And 0.54 is chosen to sit just below the 0.556 limit: at 0.6 it would leave the hill in a few steps, at 0.4 the zigzag would die out in ten.

Next, same white style: momentum, the ball that remembers which way it was going.
```

## Audio

Keep the built-in track: the soft chord bed from reel 23; at every landing one sound per ball. Just
right: a plucked note walking down a scale as its height falls. Too small: a quiet tick that barely
moves down the scale. Too big: a round bouncy tone at every landing, one per crossing of the floor,
walking down the scale only slowly. Bells at the headings. −16 LUFS.

## Replies to expect

* "The big one was winning, why did it lose?" → along the valley 0.54 is a great step size (it keeps only 46% of the distance), so its first steps are huge; across the valley it's just under the limit (0.556), so nearly all of each step goes into crossing the floor and back. One step size, two curvatures.
* "Why not make it bigger if it's fast at first?" → at 0.556 the crossings stop shrinking; above it they grow, and the ball leaves the hill. The last reel's version at 0.9 was out in six steps.
* "Isn't the start cherry-picked?" → it starts high on the valley wall so the zigzag is wide enough to see. From any start the factor per crossing is the same −0.944.
* "How do you choose it in practice?" → try a few on a log scale, watch the loss, pick the largest that still decreases smoothly; or use a schedule that starts big and shrinks. Adam makes the choice less sensitive but doesn't remove it.
* "The small one would get there eventually" → yes: about 108 steps to reach where the middle ball is at step 24. Eventually is the problem.
* "What about momentum?" → next reel.
