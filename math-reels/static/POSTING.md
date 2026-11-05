# Posting kit — *I showed a neural network pure static.* (dark theme, narrated, 77 s)

## Files

* Reel: `output/static_reel.mp4` — 77.5 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (narration over the gentle track).
* Cover: `output/cover.png` — a static image read as "a 4" at 99.7%, the histogram of how sure it is (static against real
  digits), ten more static images and what it reads.
* Runs: `static_results.json` (from `python staticsim.py`, a few seconds).
* Voice: Kokoro (open-source text-to-speech, kokoro-onnx v1.0), `am_michael` at 1.05×. Script in `scene.py` (`SCRIPT`).

## What's on screen (all of it is real)

The handwriting network of reel 49 (64 pixels → 32 neurons → 10 probabilities; 96.7% of the 540 unseen digits right),
retrained exactly. "Static": every one of the 64 pixels a random whole shade of ink from 0 to 16.

* **1,000 static images:** read with more than 90% certainty: 817; more than 99%: 617; median certainty 99.8%. Read as
  4: 696, 1: 264, 2: 29, 7: 11. The image shown big: "a 4", p = 0.997.
* **The histogram:** how sure it is of its answer, for the 1,000 static images (817 in the 90–100% column) and the 540
  real digits (485 there).
* **Why:** softmax turns the network's ten outputs into probabilities that always add up to 1. There is no "none of
  these": some digit has to win.
* **A letter A** (drawn in 8 × 8): "a 4", p = 0.9996.
* **The fix:** an 11th answer, "not a digit", trained on 600 more static images alongside the same 1,257 digits (same
  start, same 300 steps). New static images: 999 of 1,000 called "not a digit"; real digits: 96.5% read right.
* **Other noise:** scattered dots (each pixel inked with chance 0.3), never seen in training: only 72.8% caught. The
  example shown is one it misses ("a 3", 88.7%). The letter A is caught, but only with p(none) = 0.57.

## Honesty notes

* The 11-answer network is a simple version of the usual fix (training on an "other" class); real systems also use
  confidence thresholds, ensembles and out-of-distribution detectors. None of them is perfect, which is the point.
* The static here is uniform random shades; the numbers depend on the kind of noise, as the dots show.
* A blank image is read as "a 4" too, but with only 24% certainty: the network isn't sure about everything.

## Caption (paste as-is)

```
I showed this network pure static. It's 99.7% sure it's a 4.

It's the network from the handwriting reel: 96.7% right on digits it has never seen.

So I made 1,000 images of random static. For 817 of them, it was more than 90% sure of a digit. Mostly a 4, or a 1.

It isn't broken. Its answer is 10 probabilities that always add up to 100%. There's no option for "none of these". It can only ever say which digit.

Not just static: draw it a letter A, and it's 99.96% sure it's a 4.

The usual fix: give it an 11th answer, "not a digit", and train it on 600 static images too. Now new static gets "not a digit" 999 times out of 1,000, and it still reads 96.5% of real digits.

But show it a different kind of noise, scattered dots, and it catches only 73%. A network only learns to reject what it was shown.

99% sure means sure among the answers it knows. Not sure that it's right.

(Every image, probability and count on screen is computed from the real network.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #neuralnetworks #deeplearning #ai #softmax #aisafety #datascience
```

## YouTube

Title:

```
I Showed a Neural Network Pure Static. It Was 99.7% Sure. #shorts
```

Description:

```
Overconfidence, measured: what a classifier says about inputs that aren't in any of its classes.

The handwriting network (64 → 32 → 10 softmax, trained on the UCI digits, 96.7% on 540 held-out digits) shown 1,000 images of uniform random static: 817 are classified with p > 0.9, 617 with p > 0.99 (median 0.998), mostly as 4 or 1. A hand-drawn letter A: "4", p = 0.9996. Softmax has no "none of the above": the ten probabilities always sum to 1.

Adding an 11th "not a digit" class trained on 600 static images catches 999 of 1,000 new static images while keeping 96.5% digit accuracy, but only 72.8% of a different noise (sparse random dots) it never saw. Out-of-distribution inputs are a known open problem.

Voice: Kokoro, an open-source text-to-speech model (am_michael voice).

#machinelearning #neuralnetwork #aisafety #shorts
```

## Pinned comment

```
Why it's so sure: the network turns 64 pixels into 10 scores, and softmax turns the scores into probabilities by exaggerating the differences (e to the power of each score). If one score ends up a few points above the others, softmax reports that gap as near-certainty, whatever the input was. The probability measures which answer wins, not whether any answer fits.
```
