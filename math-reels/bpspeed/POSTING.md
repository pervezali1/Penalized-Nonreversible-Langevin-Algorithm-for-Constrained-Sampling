# Posting kit — *Why backpropagation is fast.* (dark theme, narrated, 63 s)

## Files

* Reel: `output/bpspeed_reel.mp4` — 63.4 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (narration over the gentle track).
* Cover: `output/cover.png` — the 105 slopes: gold bars from 210 nudging runs, magenta outlines from 2 backpropagation runs, on top of each other.
* Measurements: `speed_results.json` (from `python speedsim.py`; timings taken on an otherwise idle machine).
* Voice: Kokoro (open-source text-to-speech, kokoro-onnx v1.0), `am_michael` at 1.05×. Script in `scene.py` (`SCRIPT`).

## What's on screen (all of it is real)

**The 8 + 8 network of reels 31 and 33** (2 → 8 → 8 → 1, ReLU; trained as in reel 31 and stopped at step 150, as in
reel 33): 105 numbers (88 weights, 17 biases), the slope of the mean cross-entropy over all 400 spiral points with
respect to each.

* **Nudging** (central differences): for each number w, run the network with w + 0.00001 and with w − 0.00001 and
  divide the change in the loss by 0.00002. Two runs per number: 210 runs. On screen the edges light one by one (a
  bias rings its neuron), the output pulses twice per number, the run counter climbs to 210, and each slope drops into
  the bar chart (bar height: the slope, square-root scale) as it is measured.
* **Backpropagation**: one forward pass (gold sweep) and one backward pass (magenta sweep); all 105 slopes at once,
  drawn as magenta outlines over the gold bars.
* **The same slopes**: the largest difference between the two sets is 7.0 × 10⁻¹¹ (the largest slope is 0.014).

**A network with a million weights** (2 → 1000 → 1000 → 1, ReLU: 1,005,001 numbers), on the same 400 points, timed
on this computer (median of repeated runs):

| | Runs | Time |
| --- | --- | --- |
| One forward pass | 1 | 48.3 ms |
| Backpropagation (forward + backward) | ≈ 2 (measured: 1.95 forward passes) | 94.3 ms |
| Nudging every weight | 2,010,002 | 2,010,002 × 48.3 ms = 27.0 hours (estimated from the measured pass) |

Speed-up: about 1,030,000×. Nudging was spot-checked against backpropagation on 30 random weights of the big network:
they agree to a relative error of 9 × 10⁻¹⁰.

## Honesty notes

* The 27 hours is an estimate (2,010,002 times the measured forward pass), not a run that took 27 hours. Every other
  number is measured.
* Timings depend on the computer; the ratio is what matters: the nudging cost grows with the number of weights, the
  backpropagation cost stays at about two passes. The small network's backpropagation takes 3.4 forward passes because
  at that size fixed overheads dominate; at a million weights it's 1.95.
* Nudging is still useful: it is how backpropagation code is tested (gradient checking).

## Caption (paste as-is)

```
Backpropagation's real superpower isn't cleverness. It's speed.

To train a network you need the slope of the error for every single number in it. This one has 105.

The obvious way: nudge one number a tiny bit, run the network, see how the error changes. Twice per number to measure the slope properly: 105 numbers, 210 runs.

Backpropagation: one run forward, one run backward, and all 105 slopes come out at once. The same slopes: the biggest difference between the two methods is 0.00000000007.

Now a network with a million weights. Nudging needs two million runs. On my computer one run takes 48 milliseconds, so that's about 27 hours, for one training step. Backpropagation gets every one of those million slopes in 94 milliseconds. A million times faster.

That's why deep learning works at all: every slope, for the price of about two runs.

(Everything is measured: the 105 slopes both ways, and the timings on a real million-weight network. The 27 hours is the measured run time × 2,010,002.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #backpropagation #deeplearning #neuralnetworks #ai #calculus #math
```

## YouTube

Title:

```
Why Backpropagation Is a Million Times Faster #shorts
```

Description:

```
Why backpropagation made deep learning possible: it gets every slope for the price of about two passes through the network.

On the 8 + 8 network from the earlier reels (105 numbers): nudging each number by ±0.00001 (central differences) takes 2 runs per number, 210 runs; backpropagation takes one forward and one backward pass. The two give the same 105 slopes, to within 7 × 10⁻¹¹.

On a network with 1,005,001 weights (2 → 1000 → 1000 → 1), timed on my computer: one forward pass takes 48.3 ms, backpropagation takes 94.3 ms (1.95 passes), and nudging every weight would take 2,010,002 passes ≈ 27 hours (estimated from the measured pass). Spot-check: nudging 30 random weights agrees with backpropagation to 9 × 10⁻¹⁰.

The cost of nudging grows with the number of weights; backpropagation's stays at a few passes. That's reverse-mode automatic differentiation, and it's what every deep learning framework does.

Voice: Kokoro, an open-source text-to-speech model (am_michael voice).

#machinelearning #backpropagation #deeplearning #shorts
```

## Pinned comment

```
Why is the backward pass only about as expensive as the forward one? It goes through the same layers once, in reverse, doing the same kind of matrix multiplications: each layer's blame is the blame above times the weights. No matter how many weights there are, every one of them is touched a fixed number of times.

Nudging isn't useless: it's how people check that their backprop code is right ("gradient checking"). It's just hopeless for training.
```
