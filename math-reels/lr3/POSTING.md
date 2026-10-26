# Posting kit — *The step size.* (white style, three balls)

## Files

* Reel: `output/lr3_reel.mp4` — 30.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (gentle track).
* Cover: `output/cover.png` — the two surviving balls mid-descent, the magenta trail of the one that left.

## What's on screen (all of it is real)

The same cut hill as reel 23 (f = 0.5u² + 1.8v² on axes rotated 32°, shaded cutaway, contour lines),
three balls, same rule, different step sizes. Each ball's step is exactly its step size times the
slope under it, computed in the bowl's own axes (u ← u(1 − 2·0.5·η), v ← v(1 − 2·1.8·η), the exact
gradient-descent update for this function). 24 steps every 0.62 s.

| Ball | Step size η | Start (u, v), height | Landings | What it does |
| --- | --- | --- | --- | --- |
| TOO SMALL (green) | 0.02 | (2.6, −0.95), 5.00 | … → **1.33** at step 24 | each step shrinks the distance by only 2% along the valley; still high up |
| JUST RIGHT (vermilion) | 0.09 | (2.85, −0.64), 4.80 | … → **0.044** at step 24 | the reel-23 path: long steps first, short steps last, settled |
| TOO BIG (magenta) | 1.5 | (3.0, −0.05), 4.50 | 1.21 → 1.97 → **off the hill** (32.7) | overshoots the bottom, comes back over it higher, third step leaves the hill |

Why the big one leaves: a gradient step multiplies the distance from the bottom along a direction of
curvature c by (1 − 2cη). Along the valley c = 0.5, so η = 1.5 gives a factor of −0.5: it jumps over
the bottom but lands closer, which is why its first landing (1.21) is lower than its start. Across
the valley c = 1.8, so the factor is 1 − 2·1.8·1.5 = −4.4: every step it is 4.4 times farther from the
valley floor, on the other side. Start a hair off the centre line (v = −0.05) and that sideways error
goes −0.05 → 0.22 → −0.97 → 4.26, and at 4.26 the surface height is 32.7, far above the rim. Three
steps and it is gone. The readouts hold each landing's height (the live height would dip as the ball
flies across the bottom), and the TOO BIG column reads "left the hill" from step 3; the ball itself
fades out during the slide that takes it off, and its trail stays.

## Caption (paste as-is)

```
Same hill as last time. Same rule: feel the slope, step downhill. Three balls, and the only thing that differs is how far each one steps: 0.02 times the slope, 0.09 times the slope, 1.5 times the slope.

Too small is safe and slow. Every step shrinks its distance to the bottom by 2%. After 24 steps it is still at height 1.3, less than halfway along the valley floor, and it would need about 84 more steps to get where the middle ball already is.

Just right takes long steps while it's steep and short steps when it's flat. Below height 0.1 by step 20, at 0.044 by step 24.

Too big is the one to watch. Its first step is so long it jumps clean over the bottom and lands on the far side (4.50 → 1.21). Its second step jumps back over, and lands higher than before (1.97). Its third step leaves the hill. Three steps, gone. Not because it was unlucky: with a step this big, every step lands 4.4 times farther from the valley floor than the last one, on the opposite side.

That number, the step size, is the learning rate. It is the one number you have to choose by hand when you train a neural network, and this is what the choice means: too small wastes time, too big flies off, and the right value depends on how steep your hill is.

(Every arrow is the real negative gradient of 0.5u² + 1.8v² on rotated axes, scaled by that ball's step size; every step is exactly that. Heights: 5.00 → 1.33, 4.80 → 0.044, 4.50 → 1.21 → 1.97 → 32.7 and off the drawn surface.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #gradientdescent #learningrate #neuralnetworks #ai #math #deeplearning
```

Short alternative:

```
Same hill, same rule, three step sizes. 0.02 × slope: still a quarter of the way down after 24 steps. 0.09 × slope: settled by step 20. 1.5 × slope: over the bottom, back over it higher, and off the hill in three steps. That number is the learning rate.

Follow for more.

#machinelearning #gradientdescent #learningrate #ai
```

## Pinned comment

```
Too small wastes time. Too big flies off.

The maths: on a bowl with curvature c along a direction, a gradient step multiplies your distance from the bottom along that direction by (1 − 2cη). Along this valley c = 0.5, so η = 0.02 keeps 98% of the distance each step (that's why the green ball crawls), η = 0.09 keeps 91%, and η = 1.5 gives −0.5: over the bottom, but closer. Across the valley c = 1.8, and there η = 1.5 gives 1 − 2·1.8·1.5 = −4.4: over the floor and 4.4 times farther away, every step. That is why the big ball's sideways wobble goes from invisible to off the hill in three steps. The stable range for a direction is 0 < η < 1/c, so this hill allows up to 2 along the valley but only 0.56 across it, and one step size has to serve both. Real loss surfaces have different curvatures in every direction, so a single learning rate is always too big for the steep ones or too small for the flat ones; that is the whole reason momentum and Adam exist (reels 4, 7 and 8).

Honesty check: the big ball starts 0.05 off the valley's centre line, on purpose. Exactly on the line it would never feel the steep direction and would bounce inward (factor −0.5) instead of leaving. In a real network you are never exactly on the line.

Next, same white style: momentum, the ball that remembers which way it was going.
```

## Audio

Keep the built-in track: the soft chord bed from reel 23; at every landing one sound per ball. Just
right: a plucked note walking down a scale as its height falls. Too small: a quiet tick that barely
moves down the scale. Too big: a round bouncy tone at each of its two overshoot landings, a rising
whoosh as it leaves the hill, silence after. Bells at the headings. −16 LUFS.

## Replies to expect

* "Why does the first big step land lower if it's too big?" → along the valley 1.5 is below the limit (2), so that direction still shrinks; it's the steep direction (limit 0.56) that blows up, and that starts tiny.
* "Isn't the start cherry-picked?" → it starts 0.05 off the centre line so the steep direction is in play at all; any offset, however small, ends the same way, it just takes a few more steps.
* "How do you choose it in practice?" → try a few on a log scale, watch the loss, pick the largest that still decreases smoothly; or use a schedule that starts big and shrinks. Adam makes the choice less sensitive but doesn't remove it.
* "The small one would get there eventually" → yes: about 108 steps to reach where the middle ball is at step 24. Eventually is the problem.
* "What about momentum?" → next reel.
