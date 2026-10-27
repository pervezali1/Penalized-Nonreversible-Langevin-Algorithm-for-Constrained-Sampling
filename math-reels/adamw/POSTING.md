# Posting kit — *Adam.* (white style, three acts)

## Files

* Reel: `output/adamw_reel.mp4` — 30.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (gentle track).
* Cover: `output/cover.png` — the 100×-steeper act: plain descent's arrow flung out of the bowl, Adam a few steps down its usual path.

## What's on screen (all of it is real)

The cut hill of reels 23–25 (f = 0.5u² + 1.8v² on axes rotated 32°), three short acts of 12 steps
(0.45 s each), two balls per act from the same start (4.80 on the ×1 hill), the same number 0.25 for
both. Plain descent: step = 0.25 × slope. Adam: Kingma & Ba's update with the default β₁ = 0.9,
β₂ = 0.999, ε = 1e-8 and bias correction, step size 0.25, computed per coordinate in the bowl's own
axes. Between acts the hill is multiplied by 100 and then divided by 100: the same shape, drawn the
same, with the heights (and slopes) scaled. The readout header says which hill is on screen.

| Act | Hill | Plain descent | Adam |
| --- | --- | --- | --- |
| 1 | × 1 | 4.80 → 2.29 → 1.29 → … → **0.004** at step 12 | 4.80 → 3.65 → 2.80 → … → **0.025** at step 12 |
| 2 | × 100 | 480 → **off the hill** on step 1 (factor −24 along the valley, −89 across) | 480 → 365 → 280 → … → **2.53**: the identical path, heights × 100 |
| 3 | ÷ 100 | 0.048 → 0.0477 → … → **0.044**: down 8% in 12 steps | 0.048 → 0.0365 → … → **0.00025**: the identical path, heights ÷ 100 |

Why Adam's path is identical: it divides each slope component by the square root of a running average
of its square, so multiplying every slope by 100 multiplies both numerator and denominator by 100 and
the step does not change (ε = 1e-8 is the only thing that could break this, and it is negligible here;
the three runs agree to the last digit, end point (0.1258, −0.0982) in all three). Plain descent's step
is proportional to the slope, so ×100 makes its first step about 14 bowl-lengths long (92 units) and ÷100 makes it
crawl. Honest note built into the reel: on the ×1 hill plain descent wins (0.004 vs 0.025); Adam is
not faster here, it is indifferent to the hill's scale, and that is the point.

Presentation notes. The far exit in act 2 lands about 14 bowl-lengths away; the arrow and slide are drawn
towards a point 6 units out in the same direction so the line leaves the picture cleanly, and the
ball fades on that slide; the readout keeps the true heights and says "left the hill" from step 1.
Adam's pill rides to its lower right so it never covers plain's when the two balls share a spot.

## Caption (paste as-is)

```
Adam: the step size that doesn't care how steep the hill is.

Same hill as the last three reels, two balls, the same number for both: 0.25.

The red ball is plain descent: step = slope × 0.25. The green ball is Adam: it keeps a running average of how big the slope has been, divides the slope by that, and steps 0.25 times the result. Same number, different meaning. For red it's a fraction of the slope. For green it's roughly a distance.

Act 1, the hill as before. Red: 4.80 → 0.004 in 12 steps. Green: 4.80 → 0.025. Plain descent wins here. Watch what happens when the hill changes.

Act 2, the same hill, 100 times steeper. Red's first step is 100 times longer: it's flung clean out of the bowl on step 1. Green walks exactly the same path as before. Exactly: the same landings to the last digit, because dividing the slope by its own size cancels the 100.

Act 3, the same hill, 100 times flatter. Red's steps are 100 times shorter: it drops 8% in 12 steps. Green walks the same path a third time.

That's what Adam buys you. Not speed. Indifference to scale: the steep weights and the flat weights of a network get steps of about the same size, and a step size that works on one problem tends to work on the next. It's why 0.001 is the number almost everyone types without thinking.

(The hill is the real function 0.5u² + 1.8v² on rotated axes, scaled by 1, 100 and 0.01; plain descent is exact, Adam is the published update with its default β₁ = 0.9, β₂ = 0.999, ε = 1e-8 and bias correction. Every number on screen comes from those runs.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #adam #gradientdescent #neuralnetworks #ai #math #deeplearning #optimization
```

Short alternative:

```
Same hill, same number, two balls. Plain descent steps by slope × 0.25. Adam divides the slope by its own size first. Make the hill 100× steeper: plain descent is flung out in one step, Adam walks the identical path. 100× flatter: plain descent barely moves, Adam walks the identical path again. That's what Adam buys you.

Follow for more.

#machinelearning #adam #gradientdescent #ai
```

## YouTube description

```
Three acts on the same hill, two balls each, the same number 0.25 for both.

Plain descent: step = slope × 0.25. Adam: step = 0.25 × slope ÷ (a running measure of the slope's own size), with momentum on both the slope and its size (β₁ = 0.9, β₂ = 0.999).

Act 1, the hill as before: plain descent 4.80 → 0.004 in 12 steps, Adam 4.80 → 0.025. Plain descent wins.
Act 2, the same hill 100× steeper: plain descent's first step is 100× longer and it leaves the hill on step 1. Adam walks the identical path (heights 480 → 2.53, the same landings to the last digit).
Act 3, the same hill 100× flatter: plain descent moves 100× less and drops 8% in 12 steps. Adam walks the identical path again (0.048 → 0.00025).

Why: Adam's step is slope divided by the square root of a running average of slope², so scaling every slope by 100 scales numerator and denominator alike and the step is unchanged. Plain descent's step scales with the slope, so the same 0.25 is far too big on the steep hill and far too small on the flat one. In a real network every weight sees a different slope scale, so Adam is doing this per weight, all the time: that is why its default step size, 0.001, works on so many problems without tuning, and why it is the default optimizer almost everywhere.

Honest notes: Adam is not faster than plain descent on the ×1 hill here, and near the bottom Adam's normalised step does not shrink as fast as the slope does, so with a large step size it can wander around the minimum (that is why learning-rate decay is still used with Adam). The ε = 1e-8 in the denominator is the one thing that breaks the scale invariance, and it is negligible at these slopes.

Every number on screen is computed, not drawn: the hill is the real function 0.5u² + 1.8v² on rotated axes, plain descent is exact, and Adam is the published update with its defaults.

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #adam #gradientdescent #neuralnetworks #ai #math #deeplearning
```

## Pinned comment

```
Adam in two lines: m ← β₁m + (1 − β₁)g, s ← β₂s + (1 − β₂)g², then w ← w − η · m̂ / (√ŝ + ε). The hats are bias corrections for the first few steps. m is momentum's memory of the direction (reel 25); s is a second memory, of how big the slope has been. Dividing by √s is what makes the step indifferent to scale: multiply every slope by 100 and m and √s both grow 100×, so the step doesn't move. That is what the three acts show, and it is per weight, so steep weights and flat weights get steps of about the same size.

Honesty check: on the ordinary hill plain descent beats Adam here (0.004 vs 0.025 after 12 steps). Adam's win is not speed on one hill, it is working with the same setting on hills of any steepness, which is what you face when you don't know your loss surface. And Adam's step near the bottom shrinks only as fast as m̂/√ŝ does, so with a big step size it can wander around the minimum; that is why learning-rate decay is still used with it.

Next, same white style: the noisy slope. Why a network only looks at a few examples per step, and what that does to the ball.
```

## Audio

Keep the built-in track: the soft chord bed; a low tone as each act's balls drop; at every landing one
sound per ball. Adam: a plucked note walking down a scale as its height falls, the same tune in all
three acts because its path is the same. Plain descent: a quiet tick and a faint pluck while it is on
the hill; in act 2 a rising whoosh as the steep hill throws it off. Bells at the headings. −16 LUFS.

## Replies to expect

* "So plain descent won. Why use Adam?" → because in a network you don't get to tune the step size per weight, and the slopes differ by orders of magnitude between layers. Adam equalises them for you; plain descent needs a step size that is safe for the steepest weight and is then too small for the rest.
* "Is Adam's path really identical?" → to the last digit: the end point is (0.1258, −0.0982) in all three acts. The only thing that could break it is ε = 1e-8 in the denominator, which is nothing at these slopes.
* "Why 12 steps and not 24?" → three acts in 30 seconds. And an honest extra: run to 24 steps at this step size, Adam wanders back up to 0.25 on the ×1 hill, because its normalised step doesn't shrink with the slope. At step size 0.15 it ends at 0.029 after 24 steps. That wander is the reason learning-rate decay is still used with Adam, and it's in the pinned comment.
* "Momentum would also leave on the steep hill?" → yes, momentum's step still scales with the slope. Only the division by √s removes the scale.
* "What about AdamW?" → same update, with weight decay applied separately instead of folded into the slope. The three acts would look identical.
```
