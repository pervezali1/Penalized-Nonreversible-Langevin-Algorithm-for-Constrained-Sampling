# Posting kit — *Two valleys.* (white style, a new hill)

## Files

* Reel: `output/valleys_reel.mp4` — 30.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (gentle track).
* Cover: `output/cover.png` — the moment ball B comes over the ridge into the deep valley, ball A already at the bottom.

## What's on screen (all of it is real)

A new hill, drawn in the reel-23 style by a general-surface ray-marcher (`hillgen.py`): along the long
axis a double well, f(u, v) = 2.2·((u/2)² − 1)² + 0.6u + c + 1.8v², lifted so the deep bottom is at
height 0. The ridge between the valleys is at height 3.52, the shallow valley's bottom at 2.39, the
deep one's at 0. The cut plane sits at v = 0.75; the profile on the cut face is the double well itself.
The background is a faint tint with a light grid, by request.

Two balls, same rule (step = slope × 0.1), same start height (4.97), dropped 0.2 apart along the ridge
(u = 0.28 ± 0.1, up the back wall at v = −0.9). 40 steps of 0.38 s.

| Steps | Ball A (vermilion) | Ball B (blue) |
| --- | --- | --- |
| 0–17, plain descent | rolls into the deep valley: 0.09 at step 18 | rolls into the shallow valley: 2.46 at step 18 |
| 18–33, B is shaken | stays: 0.000 | noise of size 5 added to each slope component it feels; it rattles, climbs to 4.36 at worst, and comes over the ridge at step 24 |
| 34–39, shaking off | 0.000 | settles in the deep valley: **0.008** |

The grey arrow beside B's blue arrow during the shaken steps is the true slope's step, so you see what
the noise did to each step. The noise is stated, not hidden: Gaussian, standard deviation 5 on each
component of the slope, the same recipe as a small mini-batch (reel 27) but on a hill that is not a
least-squares bowl. Honest number in the closing cue: with this noise and these 16 steps, 19 of 100
random runs end in the deep valley; the run shown is one of them, chosen for staying on the drawn
surface (seed 59).

## Caption (paste as-is)

```
Two valleys. Where you start decides where you end.

A hill with two bottoms: one valley shallow, one deep. Two balls, dropped 0.2 apart on the ridge between them. Same rule, same step size. Plain descent only sees the slope under its feet, so whichever way the ground tilts under a ball, that's the way it goes. A rolls into the deep valley and settles at 0.09. B rolls into the shallow one and settles at 2.46, and it would sit there forever. It has no way of knowing there is a deeper valley next door.

Now shake B: add noise to the slope it feels, the way a small mini-batch does (last reel). The grey arrow is the true slope, the blue arrow is what it felt. It rattles, climbs almost to the rim, and at step 24 it comes over the ridge and drops into the deep valley. Stop the shaking and it settles at 0.008, a place it could never have walked to.

That's the other side of the noise from last time. Noise isn't only a cost. It's also what lets a ball climb out of a valley it would otherwise never leave, and it's one reason training with small batches so often ends up somewhere better than it "should".

Honest number: with this much noise, about 19 runs in 100 climb out within 16 shaken steps. This is one of them.

(The hill is a real double well, 2.2·((u/2)² − 1)² + 0.6u + 1.8v², rendered from its actual height; every step is exactly slope × 0.1, with stated Gaussian noise of size 5 on the slope during the shaken steps.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #gradientdescent #localminimum #sgd #neuralnetworks #ai #math #deeplearning
```

Short alternative:

```
A hill with two bottoms. Two balls dropped 0.2 apart roll into different valleys, and plain descent leaves them there forever. Shake the one in the shallow valley (noise on the slope, like a small batch) and it climbs over the ridge into the deep one. Where you start decides where you end. Unless you shake.

Follow for more.

#machinelearning #gradientdescent #localminimum #ai
```

## YouTube description

```
A hill with two bottoms: a shallow valley (height 2.4) and a deep one (height 0), with a ridge between them at 3.5. Two balls, same rule (step = slope × 0.1), dropped 0.2 apart on the ridge.

Steps 1–18: plain descent. Ball A rolls into the deep valley and settles at 0.09. Ball B rolls into the shallow valley and settles at 2.46. Plain descent only ever sees the slope under its feet, so B would sit there forever: that is a local minimum.

Steps 19–34: ball B is shaken. Gaussian noise of size 5 is added to each component of the slope it feels, the same kind of noise a small mini-batch produces (previous reel). The grey arrow is the true slope, the blue arrow is what it felt. It rattles, climbs to 4.36 at worst, and at step 24 it comes over the ridge and falls into the deep valley.

Steps 35–40: shaking off. It settles at 0.008.

Noise isn't only a cost. In the last reel it made the ball jitter at the bottom of a single valley; here it is the only thing that gets a ball out of the wrong valley. This is one of the reasons small-batch training often finds better solutions than the "clean" version, and why people sometimes add noise or restart from several starting points on purpose.

Honest numbers: with this noise and 16 shaken steps, 19 of 100 random runs end in the deep valley. The run shown is one of them, picked from those that stay on the drawn surface. The hill is a real double well, f = 2.2·((u/2)² − 1)² + 0.6u + 1.8v² lifted so the deep bottom is 0, and it is rendered from its actual height function; every step is exactly slope × 0.1.

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #gradientdescent #localminimum #sgd #neuralnetworks #ai #math #deeplearning
```

## Pinned comment

```
The maths in one line: gradient descent moves to wherever the slope under its feet is zero, and it has no way to compare that place with any other. On a hill with two bottoms that makes the starting point the whole decision, which is why training the same network twice from different random starts can give different results.

Noise changes the game because a noisy step can go uphill. How often it gets out depends on the barrier height against the size of the kicks, which here means step size × noise: 19 of 100 runs make it over this ridge in 16 steps with step size 0.1 and noise 5. Make the ridge higher, the kicks smaller, or the window shorter and that number drops fast; this is also why the escape mostly happens early in training, when step sizes are large, and why learning-rate decay later "freezes" the ball in whichever valley it has found.

Honesty check: the noise here is added by hand (Gaussian, size 5, stated on screen); the previous reel's noise came from real mini-batches, but this hill is not a least-squares bowl, so there is no dataset behind it. And real loss surfaces in millions of dimensions have far fewer true traps than this picture suggests; what they do have, saddles and flat plateaus, is the next reel.

Next, same white style: the saddle.
```

## Audio

Keep the built-in track: the soft chord bed; at every landing one pluck per ball, each walking down the
scale with its height (A to the bottom, B stopping part-way in the shallow valley). While B is shaken
each landing adds a quiet rattle tick and its note jumps with its height; a low tone when the shaking
stops. Bells at the headings. −16 LUFS.

## Replies to expect

* "Isn't this cherry-picked?" → the run is one of the 19% that climb out within 16 steps; the number is on screen. With more steps or bigger kicks the fraction rises; with a higher ridge it falls. The seed was chosen among the escaping runs for staying on the drawn part of the hill.
* "Why not just start in a lot of places?" → people do (random restarts). Noise gets you some of that for free, during training.
* "Would momentum get over the ridge?" → it carries speed, so from the right approach it can roll over a small ridge plain descent can't; from rest in the shallow valley it can't climb on its own. Noise can, given time.
* "Is the deep valley always better?" → on a training loss, lower is better by definition, but the lowest training loss is not always the best test error (reel 21). Flat wide valleys tend to generalise better than deep narrow ones, and noise also prefers wide ones.
* "Do real networks get stuck like this?" → less than this picture suggests: in very high dimensions most stationary points are saddles, not traps. That's the next reel.
