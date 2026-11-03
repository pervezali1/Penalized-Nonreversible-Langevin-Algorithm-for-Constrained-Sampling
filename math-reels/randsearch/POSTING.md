# Posting kit — *A million random networks vs gradient descent.* (dark theme, narrated, 73 s)

## Files

* Reel: `output/randsearch_reel.mp4` — 72.5 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (narration over the gentle track).
* Cover: `output/cover.png` — gradient descent's map at 95.5%, the chart with the random search stuck at 69.0%.
* Runs: `rs_results.json` (from `python rssim.py`, about 8 minutes; `RS_IMPROVEMENTS=1 python rssim.py` saves the record-setting networks' maps).
* Voice: Kokoro (open-source text-to-speech, kokoro-onnx v1.0), `am_michael` at 1.05×. Script in `scene.py` (`SCRIPT`).

## What's on screen (all of it is real)

The 8 + 8 network of reels 31, 33 and 41 (2 → 8 → 8 → 1, ReLU, 105 numbers) on the same 400-point spiral.

* **Random search:** 1,000,000 networks, every number drawn at random (weights spread as at the start of training,
  normal with variance 2 / fan-in; biases normal with sd 1). Each one scored by how many of the 400 points it gets
  right. The map shows the first 40 tries one by one, then the best network found so far.
  Best after 10 tries: 57.8%; 1,000: 64.3%; 100,000: 69.0%; 1,000,000: 69.0%. The last record was network #59,327:
  the next 940,673 found nothing better.
* **Gradient descent:** one random start (the start of reel 33), the slope from backpropagation, Adam, step 0.01, as
  reel 31 trained it. Step 0: 56.3%. Step 94: 69.75%, better than the best of the million. Step 271: 90%. Step
  1,500: 95.5% (382 of 400).
* **The work:** the chart puts both on one axis, runs of the network (log scale): random search 1,000,000 runs;
  gradient descent 1,500 steps of one forward and one backward run, 3,000. (Time on this computer: 449 s vs 0.74 s.)
* **Why guessing fails:** ten values for each of 105 numbers is 10¹⁰⁵ networks; the observable universe has about
  10⁸⁰ atoms (the usual estimate).

## Honesty notes

* Random search was tried with other spreads too (100,000 networks each): weights ×0.5, ×2 or ×3, biases sd 0.3 or 3:
  best 65.5–68.0%. None came near gradient descent.
* "Runs" counts a backward pass as one run; reel 41 measured it at about one forward pass.
* Gradient descent's own accuracy wobbles a little at the end (its best along the way was 96.0%); the number shown is
  the final step.

## Caption (paste as-is)

```
I tried a million random networks. The best one: 69%.

Same network as the backpropagation reel: 105 numbers, the same 400-point spiral. Each try picks all 105 numbers at random and counts how many points it gets right.

10 tries: 57.8%. 1,000: 64.3%. 100,000: 69.0%. Then nothing. Not one better network in the last 940,673.

Now gradient descent. One random start, and at every step it follows the slope that backpropagation computes. Step 94: it already beats the best of the million. Step 1,500: 95.5%.

It needed about 3,000 runs of the network. The random search used 1,000,000.

Why guessing can't work: even 10 choices for each of the 105 numbers makes 10¹⁰⁵ networks. The universe has about 10⁸⁰ atoms.

The slope says which way to move every number, all at once. That's what backpropagation buys you.

(Every network, map and score on screen is a real run.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #backpropagation #gradientdescent #neuralnetworks #deeplearning #ai #math
```

## YouTube

Title:

```
A Million Random Networks vs Gradient Descent #shorts
```

Description:

```
Why neural networks are trained with gradients and not by guessing, measured.

The 2 → 8 → 8 → 1 ReLU network (105 numbers) on a 400-point two-arm spiral. Random search: 1,000,000 networks with every weight and bias drawn at random; the best gets 69.0% of the points right, found at try 59,327, and the remaining 940,673 tries find nothing better (other random spreads: 65.5–68.0%). Gradient descent from a single random start, with the gradient from backpropagation and Adam (step 0.01): it passes the best random network at step 94 and reaches 95.5% at step 1,500, about 3,000 runs of the network against a million.

Exhaustive search is hopeless: 10 values per parameter for 105 parameters is 10^105 networks, against roughly 10^80 atoms in the observable universe.

Voice: Kokoro, an open-source text-to-speech model (am_michael voice).

#machinelearning #gradientdescent #backpropagation #shorts
```

## Pinned comment

```
Why the random search stalls: a good network has to get all 105 numbers roughly right at once. Each random draw is a fresh guess in a 105-dimensional space, and the fraction of that space where the spiral is solved is astronomically small, so a million tries barely scratch it. The gradient doesn't guess: from wherever you stand, it says how each of the 105 numbers should change to make the error smaller, and backpropagation computes all 105 of those slopes for the price of about two runs (the backpropagation-speed reel).
```
