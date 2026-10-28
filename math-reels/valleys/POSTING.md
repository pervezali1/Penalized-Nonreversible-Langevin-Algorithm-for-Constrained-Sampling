# Posting kit — *Two valleys.* (white style, a new hill, twelve shaken copies)

## Files

* Reel: `output/valleys_reel.mp4` — 30.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (gentle track).
* Cover: `output/cover.png` — mid-shake: twelve small copies of B rattling in the shallow valley, a few already over the ridge, the live count below.

## What's on screen (all of it is real)

A new hill, drawn in the reel-23 style by a general-surface ray-marcher (`hillgen.py`): along the long
axis a double well, f(u, v) = 2.2·((u/2)² − 1)² + 0.6u + c + 1.8v², lifted so the deep bottom is at
height 0. Ridge 3.52, shallow bottom 2.39, deep bottom 0. Cut plane at v = 0.85. The background is a
faint blue-grey tint with a light grid, by request.

**Part 1 (18 steps of 0.34 s).** Two balls, same rule (step = slope × 0.1), dropped 0.2 apart on the
ridge (u = 0.28 ± 0.1, up the back wall at v = −0.9), both at height 4.97. A rolls into the deep
valley and settles at 0.09; B rolls into the shallow valley and settles at 2.46.

**Part 2 (80 steps of 0.11 s).** B becomes twelve copies. Each copy takes the same plain-descent
steps, but with Gaussian noise of standard deviation 3.5 added to each component of the slope it feels
(a kick of about 0.35 per step, the kind of noise a small mini-batch produces, reel 27). Each copy has
its own random stream. The readout counts how many are over the ridge, live. The first crosses at
shaken step 9; by step 80, 5 of 12 are over.

**Part 3 (12 plain steps).** Shaking off. The 5 settle at 0 in the deep valley, the other 7 at 2.39 in
the shallow one.

| | Ball A | Ball B |
| --- | --- | --- |
| after 18 plain steps | 0.09 | 2.46 |
| copies over the ridge after 80 shaken steps | – | 5 of 12 (first at step 9; the twelve first crossings: 9, 10, 13, 15, 38, 54) |
| after shaking off | 0.09 | 5 copies at 0.000, 7 at 2.394 |

**The honest numbers.** Over 1,000 random copies with this noise and these 80 steps, 40.6% end in the
deep valley; that figure is on screen in the closing cue. The twelve shown are the first runs (by
seed) that stay on the drawn part of the hill throughout (kicks across the valley often carry a copy
past the cut plane, where the surface is not drawn; 9.5% of runs stay inside it), chosen so that their
count, 5 of 12, matches the population rate. Seeds 1, 11, 39, 42, 53, 60, 70, 78, 94, 104, 107, 179.
Seed 53 crosses the ridge at step 15 and is kicked back; it ends in the shallow valley.

## Caption (paste as-is)

```
Two valleys. Where you start decides where you end.

A hill with two bottoms, one shallow, one deep. Two balls dropped 0.2 apart on the ridge between them, same rule, same step size. Plain descent only sees the slope under its feet, so whichever way the ground tilts under a ball, that's the way it goes. A rolls into the deep valley and settles at 0.09. B rolls into the shallow one and settles at 2.46, and it would sit there forever. It has no way of knowing there's a deeper valley next door.

So shake it. Twelve copies of B, each feeling the slope plus noise, the kind of noise a small mini-batch gives you (last reel). Watch the count. The first copy comes over the ridge at step 9. Most of them rattle around the shallow valley the whole time. By step 80, five of the twelve are over, and when the shaking stops they settle at 0, a place none of them could have walked to.

That's the other face of the noise from last time. Noise isn't only a cost. It's the only thing in this picture that can go uphill, and it's one reason small-batch training so often ends up somewhere better than it "should".

Honest numbers: over 1,000 random copies with this much noise, 41% climb out within 80 steps. The twelve on screen are the first runs that stayed inside the drawn part of the hill, picked so their count matches that rate.

(The hill is a real double well, 2.2·((u/2)² − 1)² + 0.6u + 1.8v², rendered from its actual height; every step is exactly slope × 0.1, with stated Gaussian noise of size 3.5 on each slope component during the shaken steps.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #gradientdescent #localminimum #sgd #neuralnetworks #ai #math #deeplearning
```

Short alternative:

```
A hill with two bottoms. Two balls dropped 0.2 apart roll into different valleys, and plain descent leaves them there forever. Shake twelve copies of the one in the shallow valley (noise on the slope, like a small batch): five of twelve climb over the ridge into the deep one. Where you start decides where you end. Unless you shake.

Follow for more.

#machinelearning #gradientdescent #localminimum #ai
```

## YouTube description

```
A hill with two bottoms: a shallow valley (height 2.4) and a deep one (height 0), with a ridge between them at 3.5. Two balls, same rule (step = slope × 0.1), dropped 0.2 apart on the ridge.

Part 1, plain descent, 18 steps. Ball A rolls into the deep valley and settles at 0.09. Ball B rolls into the shallow valley and settles at 2.46. Plain descent only ever sees the slope under its feet, so B would sit there forever: that is a local minimum.

Part 2, the shake, 80 steps. B becomes twelve copies. Each takes the same descent steps, but with Gaussian noise of size 3.5 added to each component of the slope it feels, the kind of noise a small mini-batch produces (previous reel), each with its own random stream. The count on screen is live. The first copy comes over the ridge at step 9; by step 80, 5 of 12 are over. One (seed 53) crosses at step 15 and gets kicked back.

Part 3, shaking off, 12 plain steps. The five settle at 0 in the deep valley, the seven at 2.39 in the shallow one.

Noise isn't only a cost. In the last reel it made a ball jitter at the bottom of a single valley; here it is the only thing that gets a ball out of the wrong valley. That is one of the reasons small-batch training often finds better solutions than the "clean" version, and why people sometimes add noise, or restart from several random starts, on purpose.

Honest numbers: over 1,000 random copies with this noise and these 80 steps, 40.6% end in the deep valley. The twelve shown are the first runs (by seed) that stay inside the drawn part of the hill for all 80 steps (sideways kicks often carry a copy past the cut plane), chosen so that their count matches the population rate. The hill is a real double well, f = 2.2·((u/2)² − 1)² + 0.6u + 1.8v² lifted so the deep bottom is 0, rendered from its actual height function; every step is exactly slope × 0.1.

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #gradientdescent #localminimum #sgd #neuralnetworks #ai #math #deeplearning
```

## Pinned comment

```
The maths in one line: gradient descent moves to wherever the slope under its feet is zero, and it has no way to compare that place with any other. On a hill with two bottoms the starting point is the whole decision, which is why training the same network twice from different random starts can give different results.

Noise changes the game because a noisy step can go uphill. How often it gets out depends on the barrier height against the size of the kicks (step size × noise): here 41% of 1,000 copies make it over a ridge 1.1 high in 80 steps with kicks of 0.35. Higher ridge, smaller kicks or a shorter window and that number drops fast; this is also why escapes mostly happen early in training, when step sizes are large, and why learning-rate decay later "freezes" the ball in whichever valley it has found.

Honesty check: the noise here is added by hand (Gaussian, size 3.5, stated on screen); last reel's noise came from real mini-batches, but this hill is not a least-squares bowl, so there is no dataset behind it. The twelve copies shown are the first runs that stayed inside the drawn part of the hill, picked so their 5 of 12 matches the 41%. And real loss surfaces in millions of dimensions have far fewer true traps than this picture suggests; what they do have, saddles and flat plateaus, is the next reel.

Next, same white style: the saddle.
```

## Audio

Keep the built-in track: the soft chord bed; one pluck per ball per landing in part 1, walking down the
scale with height (A to the bottom, B stopping part-way); during the shake a quiet rattle tick per step
and a bright pluck the moment a copy comes over the ridge (a lower one for the copy that is kicked
back); a low tone when the shaking stops. Bells at the headings. −16 LUFS.

## Replies to expect

* "Isn't this cherry-picked?" → the population number is on screen: 41% of 1,000 random copies climb out. The twelve shown were picked to stay inside the drawn part of the hill and to match that rate, 5 of 12. Every seed is in the repo.
* "Why not just start in a lot of places?" → people do (random restarts). Noise gets you some of that for free, during training.
* "Would momentum get over the ridge?" → it carries speed, so from the right approach it can roll over a small ridge plain descent can't; from rest in the shallow valley it can't climb on its own. Noise can, given time.
* "Is the deep valley always better?" → on a training loss, lower is better by definition, but the lowest training loss is not always the best test error (reel 21). Flat wide valleys tend to generalise better than deep narrow ones, and noise also prefers wide ones.
* "Do real networks get stuck like this?" → less than this picture suggests: in very high dimensions most stationary points are saddles, not traps. That's the next reel.
