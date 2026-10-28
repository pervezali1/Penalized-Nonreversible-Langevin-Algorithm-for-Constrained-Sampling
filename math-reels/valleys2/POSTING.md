# Posting kit — *The wrong valley, shaken.* (white style, batch sizes on the two-valley hill)

## Files

* Reel: `output/valleys2_reel.mp4` — 30.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (gentle track).
* Cover: `output/cover.png` — shake 70: 4-example copies scattered over both valleys, the 8- and 32-example ones rattling at the bottom, ball C still, the counts below.

## What's on screen (all of it is real)

The two-valley hill of reel 28 (ridge 3.52, shallow bottom 2.39, deep bottom 0; the ridge is 1.13 above
the shallow bottom), rendered by `../valleys/hillgen.py` in the reel-23 look on the faint tinted grid.

**The data.** The hill is the average error over 1,000 examples. Each example's own error is the hill
plus a tilt, ℓᵢ(w) = f(w) + τᵢ·w, with the 1,000 tilts summing to zero exactly (`tiltsim.py`, tilt
scale 7 per component). So the full-batch slope is exactly ∇f, and a mini-batch of B examples feels
∇f plus the mean tilt of its batch: genuine sampling noise, standard deviation 7/√B per component,
drawn fresh (without replacement) at every step.

**The run.** Ball C sits at the bottom of the shallow valley and looks at all 1,000 examples: its slope
is exactly zero, so it never moves. From the same spot, three groups of eight copies take the same
steps (step size 0.1) on the slope of their own batch: 4, 8 or 32 examples a step. 140 shaken steps of
0.1 s, then 12 steps on the exact slope to settle. The readout counts, live, how many of each group are
over the ridge.

| Examples per step | Kick per step (step size × noise) | First copy over the ridge | Over the ridge after 140 steps | Over 500 random runs |
| --- | --- | --- | --- | --- |
| 4 | 0.35 | step 10 (then 21, 25, 40, 67, 96) | **5 of 8** | **60%** |
| 8 | 0.25 | step 30 (then 65, 112) | **1 of 8** | **12%** |
| 32 | 0.12 | never | **0 of 8** | **0%** |
| all 1,000 | 0 | never | still | 0% |

The five, the one and nobody else settle at 0; everyone else, and C, stay at 2.39.

**Honest numbers.** The population rates (500 random runs per batch size, no filtering) are on screen
in the closing cue, and the on-screen counts match them. The eight copies shown per group are the first
runs, by seed, that never go above the rim of the drawn hill (for the 4-example group that skipped 10
runs out of 18; none for the others). Copies that stray past the cut plane are drawn where they are.

## Caption (paste as-is)

```
The wrong valley, shaken.

Last reel: four balls, plain descent, two of them stuck in the shallow valley forever. This is what batch size does to one of them.

The hill is the average error over 1,000 examples. Ball C looks at all 1,000 before every step, so at the bottom of the shallow valley its slope is exactly zero. It never moves. Then 24 copies of it start measuring the slope on a batch instead: eight look at 4 examples a step, eight at 8, eight at 32. Smaller batch, noisier slope, bigger kicks: 0.35, 0.25 and 0.12 per step against a ridge 1.1 high.

Watch the counts. The first copy over the ridge is a 4-example one, at step 10. By step 140 it's 5 of 8 for the 4-example copies, 1 of 8 for the 8-example ones, 0 of 8 for the 32s. Stop the shaking and the six that made it settle at 0, a place none of them could have walked to. The rest, and C, are still at 2.39.

Over 500 random runs each, the rates are 60%, 12% and 0%.

That's the other face of the noise from reel 27. The same noise that makes a small batch jitter at the bottom of a single valley is the only thing in this picture that can go uphill, and it gets more balls out of the wrong valley the smaller the batch. It is one real reason small-batch training so often ends up somewhere better than it "should".

(The hill is a real double well, 2.2·((u/2)² − 1)² + 0.6u + 1.8v²; every example's error is the hill plus its own tilt, the tilts sum to zero, and every step is exactly slope × 0.1 on a fresh random batch. The eight copies shown per group are the first random runs that stay below the rim of the drawing; the percentages are over all runs.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #gradientdescent #sgd #minibatch #localminimum #neuralnetworks #ai #math #deeplearning
```

Short alternative:

```
A ball stuck in the shallow valley, slope zero, never moving. 24 copies measure the slope on a batch instead: 4, 8 or 32 examples a step. The 4-example copies get out, 5 of 8. The 8s: 1 of 8. The 32s: none. Smaller batch, bigger kicks, more balls over the ridge. Over 500 runs each: 60%, 12%, 0%.

Follow for more.

#machinelearning #sgd #minibatch #localminimum #ai
```

## YouTube description

```
The two-valley hill from the previous reel, and a ball stuck at the bottom of the shallow valley. The hill is the average error over 1,000 examples; each example's own error is the hill plus a tilt, and the tilts sum to zero, so a step that looks at all 1,000 examples feels the exact slope (zero at the bottom: ball C never moves), while a step that looks at a batch of B examples feels the exact slope plus the mean tilt of its batch, real sampling noise of size 7/√B.

Three groups of eight copies take 140 steps (step size 0.1) from the same spot, measuring the slope on 4, 8 or 32 examples a step, each with its own random batches. Kicks per step: 0.35, 0.25 and 0.12. The ridge is 1.1 above the shallow bottom.

4 examples a step: first over the ridge at step 10; 5 of 8 over by step 140.
8 examples a step: first over at step 30; 1 of 8.
32 examples a step: 0 of 8. They wobble at the bottom and never climb.
All 1,000: slope exactly zero, never moves.

With the shaking off (12 exact steps), the six that crossed settle at 0 and the rest stay at 2.39. Over 500 random runs per batch size the rates are 60%, 12% and 0%.

Why it matters: in reel 27 the mini-batch noise was only a cost, jitter at the bottom of a single valley. Here the same noise is the only thing that can carry a ball uphill, and the smaller the batch the more often it gets out of the wrong valley. That is one real reason small-batch training often finds better solutions than the "clean" full-batch version, and why learning-rate decay (which shrinks the kicks) is usually applied late, after the escaping has had its chance.

Honest notes: the eight copies shown per group are the first random runs that never go above the rim of the drawing (10 of 18 were skipped for the 4-example group, none for the others); the percentages are over all runs with no filtering. The hill is a real double well, 2.2·((u/2)² − 1)² + 0.6u + 1.8v², and every step is exactly slope × 0.1 on a fresh random batch.

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #gradientdescent #sgd #minibatch #localminimum #neuralnetworks #ai #math #deeplearning
```

## Pinned comment

```
Mini-batch noise in one line: the slope on a batch of B examples is the true slope plus the average of B random tilts, so its size shrinks like 1/√B. With step size 0.1 that is a kick of 0.35 per step at B = 4, 0.25 at B = 8, 0.12 at B = 32, against a ridge 1.1 high. Getting out is a random walk against a slope, so the escape rate falls off very fast as the kicks shrink: 60%, 12%, 0% here. Bigger step size or more steps would raise all three; learning-rate decay lowers them, which is why it is applied late.

Honesty check: the per-example "tilts" are a stand-in for how much each example's error disagrees with the average; a real dataset's disagreement is not a fixed tilt, but the 1/√B scaling of the batch noise is the same. And in millions of dimensions the ridge you need to cross is rarely this clean; most of what stops training in practice is flat ground, not a wall. That reel, the saddle, is next.
```

## Audio

Keep the built-in track: the soft chord bed; a low tone as the balls appear; a quiet rattle tick on every
shaken step; a pluck the moment a copy comes over the ridge, higher for the 4-example copies and lower
for the 8-example ones; a low tone when the shaking stops. Ball C makes no sound: it never moves. Bells
at the headings. −16 LUFS.

## Replies to expect

* "So small batches are better?" → for getting out of a bad valley, yes; for settling precisely at a bottom, no (reel 27). That's why training usually starts noisy and ends calm: big step size and small batches early, decay late.
* "Why does the full-batch ball never move at all?" → at the bottom the exact slope is zero, so the step is exactly zero. Only noise or momentum can move it.
* "Isn't 8 copies a small sample?" → it is; that's why the 500-run rates are on screen too, and the 8 shown match them (5/8 ≈ 60%, 1/8 ≈ 12%, 0/8).
* "Why are the rates so different for 4 and 8?" → climbing out is a random walk against a slope; its success rate falls off roughly exponentially as the kicks shrink. 0.35 to 0.25 per step is enough to go from 60% to 12%.
* "Would a bigger step size do the same?" → yes, the kick is step size × noise; doubling the step size is like quartering the batch. Too big and you leave the hill entirely (reel 24).
```
