# Posting kit — *The step size.* (white style, three balls)

## Files

* Reel: `output/lr3_reel.mp4` — 30.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (gentle track).
* Cover: `output/cover.png` — the three balls mid-descent, the magenta arrow spanning the valley.

## What's on screen (all of it is real)

The same cut hill as reel 23 (f = 0.5u² + 1.8v² on axes rotated 32°, shaded cutaway, contour lines),
three balls, same rule, different step sizes. Each ball's step is exactly its step size times the
slope under it, computed in the bowl's own axes (u ← u(1 − 2·0.5·η), v ← v(1 − 2·1.8·η), the exact
gradient-descent update for this function). 24 steps every 0.62 s.

| Ball | Step size η | Start (u, v), height | After 24 steps | What it does |
| --- | --- | --- | --- | --- |
| TOO SMALL (green) | 0.02 | (2.6, −0.95), 5.00 | **1.33** | each step shrinks the distance by only 2% along the valley; still high up |
| JUST RIGHT (vermilion) | 0.09 | (2.85, −0.64), 4.80 | **0.044** | the reel-23 path: long steps first, short steps last, settled |
| TOO BIG (magenta) | 2.0 | (3.0, 0), 4.50 | **4.50** | jumps clean over the bottom and lands at the same height on the far side, every step, forever |

Why the big one bounces forever: along the valley the slope is 2·0.5·u = u, so a step of 2 × slope
moves it by 2u, from u to exactly −u. Same distance from the bottom, same height, other side. Any
step size above 2 would land *farther* out each time and leave the hill; any step size below 2
would creep inward. 2.0 is the knife edge and the cleanest picture of "never arrives".

The readouts hold each landing's height (the live height would dip as the big ball flies across the
bottom, which would read as if it had got there). Pills above each ball carry its step size.

## Caption (paste as-is)

```
Same hill as last time. Same rule: feel the slope, step downhill. Three balls, and the only thing that differs is how far each one steps: 0.02 times the slope, 0.09 times the slope, 2 times the slope.

Too small is safe and slow. Every step shrinks its distance to the bottom by 2%. After 24 steps it is still at height 1.3, less than halfway along the valley floor, and it would need about 84 more steps to get where the middle ball already is.

Just right takes long steps while it's steep and short steps when it's flat. Below height 0.1 by step 20, at 0.044 by step 24.

Too big is the one to watch. Its first step is so long it jumps clean over the bottom and lands on the far side, at exactly the same height. Then it does it again. And again. 24 steps later it is at 4.50, precisely where it started, having crossed the valley 24 times. It never arrives, and it never will.

That number, the step size, is the learning rate. It is the one number you have to choose by hand when you train a neural network, and this is what the choice means: too small wastes time, too big never arrives, and the right value depends on how steep your hill is.

(Every arrow is the real negative gradient of 0.5u² + 1.8v² on rotated axes, scaled by that ball's step size; every step is exactly that. The big ball runs along the valley's centre line, where a step of 2 × slope takes it from u to exactly −u. Reels 5, 7 and 8 are what happens off that line.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #gradientdescent #learningrate #neuralnetworks #ai #math #deeplearning
```

Short alternative:

```
Same hill, same rule, three step sizes. 0.02 × slope: still a quarter of the way down after 24 steps. 0.09 × slope: settled by step 20. 2 × slope: jumps clean over the bottom and lands at the same height on the far side, every step, forever. That number is the learning rate.

Follow for more.

#machinelearning #gradientdescent #learningrate #ai
```

## Pinned comment

```
Too small wastes time. Too big never arrives.

The maths: on a bowl with curvature c along a direction, a gradient step multiplies your distance from the bottom along that direction by (1 − 2cη). Along this valley c = 0.5, so η = 0.02 keeps 98% of the distance each step (that's why the green ball crawls), η = 0.09 keeps 91%, and η = 2 gives (1 − 2) = −1: same distance, other side, forever. Above 2 the factor is below −1 and every step lands farther out; the ball would leave the hill. The stable range is 0 < η < 1/c, and the fastest honest choice is near the middle of it. Real loss surfaces have different curvatures in different directions, so one step size can be too big for the steep ones and too small for the flat ones at the same time; that is the whole reason momentum and Adam exist (reels 4, 7 and 8).

Honesty check: the big ball runs exactly along the valley's centre line. A hair off it, the steep direction (curvature 1.8) would have a factor of 1 − 2·1.8·2 = −6.2 and it would fly out sideways in three steps, which is what reel 5 showed. I put it on the line so you can watch the overshoot itself instead of an explosion.

Next, same white style: momentum, the ball that remembers which way it was going.
```

## Audio

Keep the built-in track: the soft chord bed from reel 23; at every landing one sound per ball. Just
right: a plucked note walking down a scale as its height falls. Too small: a quiet tick that barely
moves down the scale. Too big: a round bouncy tone on the same pitch every time, because it lands at
the same height every time. Bells at the headings. −16 LUFS.

## Replies to expect

* "Why doesn't the big one fly off?" → it runs exactly along the valley's centre line, where 2 × slope lands it at the mirror point. Off the line it would explode sideways in three steps; that's reel 5.
* "Isn't 2.0 cherry-picked?" → yes, deliberately: it's the exact boundary between creeping in and flying out, so you can see the overshoot itself. 1.9 would spiral in slowly, 2.1 would leave.
* "How do you choose it in practice?" → try a few on a log scale, watch the loss, pick the largest that still decreases smoothly; or use a schedule that starts big and shrinks. Adam makes the choice less sensitive but doesn't remove it.
* "The small one would get there eventually" → yes: about 108 steps to reach where the middle ball is at step 24. Eventually is the problem.
* "What about momentum?" → next reel.
