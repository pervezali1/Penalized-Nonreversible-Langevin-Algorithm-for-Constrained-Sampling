# Posting kit — *Every ReLU neuron is a hinge.* (dark theme, no formulas on screen, 50 s)

## Files

* Reel: `output/hinges_reel.mp4` — 50.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (gentle track).
* Cover: `output/cover.png` — 32 neurons: the gold fit on the dashed target, 28 hinges, the pieces below.
* Simulation archive: `hinges_sim.npz` (regenerate with `python hgsim.py`; not committed, 140 KB; the scene
  regenerates it if it is missing).

## What's on screen (all of it is real)

One hidden layer of ReLU neurons, 1 input → N neurons → 1 output (no activation on the output), fitted
to one curve: f(x) = sin(2πx)·(0.55 + 0.45x) on [−1, 1], 256 evenly spaced points, mean squared error,
Adam with step size 0.01, 6,000 steps, hinge positions started spread over [−1, 1]. Each neuron computes
max(0, w·x + b): flat on one side of x = −b/w, a straight ramp on the other. That bend is a hinge. The
output is a weighted sum of these bent lines plus a constant. For N = 1, 2, 4, 8, 16 and 32, six seeds
each; the run with the (lower) median final error is shown, training live from snapshots every 60 steps.

* **Top:** the target (dashed) and the network (gold), with a dot at every hinge that lands inside the
  plot and a tick under it (vermilion if the neuron's output weight is positive, blue if negative).
* **Middle:** the pieces, one bent line per neuron (output weight × max(0, w·x + b)), auto-scaled per
  width (the scale is printed; pieces can be much larger than the curve and cancel).
* **Readouts:** neurons, hinges inside, error (root-mean-square gap to the target over the 256 points),
  all live.

| Neurons | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- |
| Error (shown run) | 0.393 | 0.393 | 0.325 | 0.037 | 0.019 | 0.012 |
| Hinges inside [−1, 1] | 1 | 1 | 4 | 7 | 14 | 28 |
| Error, all six seeds | 0.422 0.393 0.334 0.334 0.408 0.394 | 0.333 0.393 0.408 0.325 0.393 0.403 | 0.325 0.333 0.388 0.229 0.099 0.333 | 0.037 0.212 0.212 0.040 0.035 0.032 | 0.017 0.034 0.020 0.014 0.019 0.029 | 0.012 0.013 0.015 0.007 0.022 0.009 |

Neurons whose hinge lands outside the plot: with 2, the second is off everywhere on [−1, 1] and adds
nothing (so the error equals the one-neuron error exactly, as the reel says); with 8, the extra one is on
everywhere and only tilts the line; with 16, one of each; with 32, four are off everywhere.

## Honesty notes

* Training is not guaranteed to find the best fit for a given width. Four neurons could do much better
  than 0.325 (one seed reached 0.099), and the median four-neuron run got stuck; eight neurons is where
  every seed but two follows the wave. The reel shows the median run and says "Training found a poor fit."
* "Any curve": the universal approximation theorem says one hidden layer with enough ReLU neurons can
  approximate any continuous function on a closed interval as closely as you like. It does not say how
  many neurons are needed or that training will find the weights. A ReLU network's output is always
  piecewise linear: the "curve" is many short straight pieces.

## Caption (paste as-is)

```
Every ReLU neuron is a hinge.

A single neuron is flat on one side of a point and a straight ramp on the other. That bend is a hinge. A layer adds its neurons' bent lines together. So what can a pile of hinges draw?

Same target every time (dashed), one layer of neurons, trained live (gold):

1 neuron: one bend. Error 0.393. A wave is out of reach.
2 neurons: no better. The second neuron landed where it is off everywhere, so it adds nothing.
4 neurons: four hinges, but training found a poor fit. Error 0.325.
8 neurons: now it follows the wave. Error 0.037.
16: 0.019.
32: 0.012. You can barely tell the gold from the dashed.

The pieces underneath are the honest part: each line is one neuron's bent contribution, some huge, some cancelling each other, and their sum is the gold curve.

That's the universal approximation theorem made visible: one layer of hinges can draw any continuous curve, if you give it enough of them. It never says how many, or that training will find them. And it's always straight pieces, never a true curve.

(Median of 6 random seeds per width. Every hinge dot is a real neuron's bend; every error is computed.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #neuralnetworks #relu #deeplearning #ai #math #universalapproximation
```

Short alternative:

```
Every ReLU neuron is a hinge: flat, then a ramp. Add hinges and you can draw a curve. One neuron: error 0.393. Four: 0.325 (training got stuck). Eight: 0.037. Thirty-two: 0.012, almost on top of the target. The universal approximation theorem, one hinge at a time.

Follow for more.

#machinelearning #neuralnetworks #relu #ai
```

## YouTube description

```
A ReLU neuron computes max(0, w·x + b): flat on one side of the point x = −b/w and a straight ramp on the other. One bend: a hinge. A layer of these neurons adds their bent lines together (weighted), plus a constant.

Here one hidden layer of 1, 2, 4, 8, 16 and 32 ReLU neurons is trained (Adam, 6,000 steps, mean squared error) to fit one curve, sin(2πx)(0.55 + 0.45x) on [−1, 1]. The gold line is the network, the dashed line the target, each dot a hinge, and underneath, every neuron's own bent line.

Errors (root-mean-square gap, median of 6 seeds): 1 neuron 0.393; 2 neurons 0.393 (the second one ends up off everywhere on the interval and contributes nothing); 4 neurons 0.325 (the median run gets stuck; one seed reached 0.099); 8 neurons 0.037; 16 neurons 0.019; 32 neurons 0.012.

This is the universal approximation theorem made visible: one hidden layer of ReLU neurons can approximate any continuous function on an interval as closely as you like, given enough neurons. It doesn't say how many, or that gradient descent will find them, and the result is always piecewise linear.

#machinelearning #neuralnetworks #relu #universalapproximation
```

## Pinned comment

```
Why a hinge: max(0, w·x + b) is 0 until w·x + b crosses zero, then grows in a straight line. Multiply by the output weight and you get a bent line with one corner at x = −b/w. Add N of them plus a constant and you get a curve made of up to N + 1 straight pieces, with a corner at each hinge.

Why 2 neurons did no better than 1: the second neuron's hinge ended up at x = 1.37, outside the interval, on the side where it is off. A neuron that is off everywhere contributes exactly nothing (and gets no gradient to escape: the "dead ReLU").

The theorem (Cybenko 1989, Hornik 1991; for ReLU, Leshno et al. 1993): one hidden layer with enough neurons can get within any tolerance of any continuous function on a closed interval. Enough can be a lot. Depth is what makes it cheap: the reels on folds of folds showed why.
```

## Audio

Gentle register, −16 LUFS, true peak under −1 dBTP, circular over 50 s. A soft chord bed; at each width a
rising arpeggio, one pluck per neuron (up to 16); a quiet tick every few training snapshots; a chime at
the end of each stage pitched by its error (lower error, higher note); a low note under the closing
table; bells at the headings.
