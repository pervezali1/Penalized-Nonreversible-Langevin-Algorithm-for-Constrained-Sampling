# Posting kit — *The noisy slope.* (white style, mini-batches)

## Files

* Reel: `output/batchw_reel.mp4` — 30.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (gentle track).
* Cover: `output/cover.png` — the three balls a few steps down, the 4-example ball's zigzag against the smooth path.

## What's on screen (all of it is real)

The hill is not a drawn bowl this time: it is the average squared error of a two-weight least-squares
fit over 1,000 real data points (`batchsim.py`). The points are constructed so that this average is
EXACTLY the bowl of reels 23–26, 0.5u² + 1.8v², plus a floor of 0.10 that comes from the labels' own
noise (the inputs are whitened to the exact second moment diag(1, 3.6); the label noise is made
orthogonal to the inputs and to the constant and scaled so its mean square is 0.20). So the drawn hill,
the full-batch slope and the reel-24 path are all exact, and the mini-batch slope is the true slope
plus genuine sampling noise.

Three balls, same rule (step = slope × 0.2), same start (error 4.90). Before every step each ball
measures the slope on a random sample of the 1,000 points, drawn without replacement with a fixed seed:

| Ball | Examples per step | Error after 24 steps | Steps 16–24: mean, jitter (std) | Examples looked at in 24 steps |
| --- | --- | --- | --- | --- |
| 4 EXAMPLES (magenta) | 4 | **0.111** | 0.119 ± 0.019 | 96 |
| 32 EXAMPLES (blue) | 32 | **0.101** | 0.103 ± 0.002 | 768 |
| ALL 1,000 (vermilion) | 1,000 | **0.100** | 0.101 ± 0.001 | 24,000 |

The 4-example ball's measured slope is on average 51° off the true slope (up to 152° on one step),
which is why its path zigzags on the wall and why it keeps bouncing at the bottom. The grey arrow shown
beside each noisy ball's coloured arrow during the arrow phase is the true slope's step.

Two facts the cues state: the expected position of a mini-batch ball is exactly the full-batch path
(the sampling is uniform and the label noise is orthogonal to the inputs, so the expected mini-batch
slope is the true slope); and no ball can go below 0.10, because 0.10 is the error of the perfect
weights. The readout bars carry a small tick at the floor.

## Caption (paste as-is)

```
The noisy slope.

A network never feels the whole hill. Every step, it looks at a few examples, measures the slope on those, and steps. That's a mini-batch, and this is what it does to the ball.

This hill is the average error over 1,000 examples. Three balls, same rule, same step size, same start. The only difference is how many of the 1,000 each ball looks at before every step: 4, 32, or all 1,000.

All 1,000 walks the smooth path from the last reels. 32 wobbles a little. 4 feels a slope that is, on average, 51 degrees off the true one, and you can see it: it zigzags down the wall and never quite sits still at the bottom.

Two things worth knowing. On average the noise cancels: the expected path of the noisy balls is exactly the smooth one. The noise doesn't slow them down, it shakes them. And the floor is 0.10, not 0: the labels themselves are noisy, so even the perfect weights are wrong by that much, and no ball can beat it.

After 24 steps: 0.111, 0.101, 0.100. Still bouncing, a wobble, still.

Now the price. In those 24 steps the three balls looked at 96, 768 and 24,000 examples. Small batches buy cheap steps and pay in noise. That is the whole trade behind the batch size, and it is why training curves are jagged.

(The hill is the real mean squared error of a least-squares fit over 1,000 points, built so that average is exactly the bowl from the last reels plus the noise floor. Every slope is measured on a real random sample of those points; every step is exactly slope × 0.2.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #gradientdescent #sgd #minibatch #neuralnetworks #ai #math #deeplearning
```

Short alternative:

```
Same hill, same rule, three balls. The only difference: how many of the 1,000 examples each one looks at before a step. 4, 32, or all of them. Fewer examples, noisier slope: the 4-ball zigzags and never sits still. But it looked at 96 examples in 24 steps; the smooth one looked at 24,000. That's the mini-batch trade.

Follow for more.

#machinelearning #sgd #gradientdescent #ai
```

## YouTube description

```
Three balls on the same hill, same rule (step = slope × 0.2), same start. The hill is the average error over 1,000 data points. Before every step each ball measures the slope on a random sample of those points: 4 of them, 32 of them, or all 1,000. That sample is a mini-batch.

All 1,000: the smooth path from the previous reels, error 4.90 → 0.100 in 24 steps.
32 examples: a wobble, 4.90 → 0.101.
4 examples: the slope it measures is on average 51° off the true slope. It zigzags down the wall, reaches the bottom about as fast on average, and keeps bouncing there: 4.90 → 0.111, and between steps 16 and 24 it sits at 0.119 ± 0.019.

Two honest facts the reel states. First, the expected path of a noisy ball is exactly the smooth path (the sample is uniform, so the expected mini-batch slope is the true slope): noise doesn't slow the ball on average, it shakes it. Second, the floor of this hill is 0.10, not 0, because the labels are noisy; the perfect weights are still wrong by 0.10, and no ball can beat that.

The price of smoothness: in 24 steps the three balls looked at 96, 768 and 24,000 examples. That is the mini-batch trade. Small batches give cheap, noisy steps; big batches give expensive, clean ones. It is why training loss curves are jagged, why the step size is usually shrunk late in training (to stop the bouncing), and why the batch size is a knob worth thinking about.

Every number on screen is computed, not drawn. The data is 1,000 points whose mean squared error is exactly the bowl 0.5u² + 1.8v² plus the floor 0.10 (inputs whitened to that curvature, label noise orthogonal to the inputs); every slope is measured on a real random subset of them, and the grey arrows are the true slope for comparison.

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #gradientdescent #sgd #minibatch #neuralnetworks #ai #math #deeplearning
```

## Pinned comment

```
The mini-batch slope is the true slope plus noise with zero mean, so E[position] follows the full-batch path exactly, and the noise shows up as variance. Near the bottom that variance settles to roughly (step size × noise per example) / (2 × curvature × batch size): halve the batch and the jitter doubles; halve the step size and it halves too. That's why learning-rate decay stops the bouncing, and why bigger batches let you use bigger step sizes.

Honesty check: 4 examples at step size 0.2 is on the edge. With a different random draw this ball sometimes takes a wild early step out of the drawn bowl (3 of 8 seeds did); the one shown is a typical run, not the best one, picked for staying on the picture. And the floor is a feature, not a bug: real data has label noise, and a training error that goes to exactly zero usually means memorisation (reel 21).

Next, same white style: two valleys. Where you start decides where you end, and a little noise is what lets a ball climb out.
```

## Audio

Keep the built-in track: the soft chord bed; at every landing one sound per ball. All 1,000: a plucked
note walking cleanly down a scale. 32: the same pluck, quieter, on its own slightly shaky note. 4: a
quiet tick and a pluck whose note jumps around with its error. Bells at the headings. −16 LUFS.

## Replies to expect

* "Why would anyone use 4 examples then?" → cost. The 4-ball took 24 steps for the price of 96 slope measurements; the full ball paid 24,000 for the same 24 steps. Per example looked at, the small batch gets vastly more steps, and the noise can be tamed by shrinking the step size later.
* "Is the noise really zero-mean?" → yes: uniform sampling makes the expected mini-batch gradient equal to the full gradient, and in this dataset the label noise is exactly orthogonal to the inputs, so the full gradient is exactly the bowl's. The expected path is the smooth one.
* "Does noise help escape bad valleys?" → that's the next reel. On this hill there is one valley, so the noise only costs.
* "Why 32?" → it's the batch size people type by habit. Here it already looks almost smooth; on harder data it wouldn't.
* "What's the floor?" → the error of the perfect weights, 0.10, because the labels carry noise the weights can't explain. A model that beats the floor on training data is memorising noise.
```
