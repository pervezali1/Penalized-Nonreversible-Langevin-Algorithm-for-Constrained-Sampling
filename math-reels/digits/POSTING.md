# Posting kit — *A network learns to read handwriting.* (dark theme, narrated, 74 s)

## Files

* Reel: `output/digits_reel.mp4` — 73.5 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (narration over the gentle track).
* Cover: `output/cover.png` — the blame carried back to the pixels of an unseen 3.
* Data: `digits.csv` — the UCI handwritten digits (Alpaydin & Kaynak 1998), as shipped with scikit-learn: 1,797 digits
  written by 13 people, 8 × 8 pixels, ink 0–16. CC BY 4.0.
* Training: `dgsim.py` → `dg_results.json` (and `dg_results.npz`, rebuilt by the scene if missing).
* Voice: Kokoro (open-source text-to-speech, kokoro-onnx v1.0), `am_michael` at 1.05×. Script in `scene.py` (`SCRIPT`).

## What's on screen (all of it is real)

The 1,797 digits split at random: 1,257 to train on, 540 the network never trains on. Network: 64 pixels → 32 ReLU
neurons → 10 outputs with softmax (2,410 numbers), cross-entropy, Adam (step 0.01, all training digits at once), 300
steps.

* **The network:** the digit as its 64 pixels; the 32 neurons drawn as pictures of their 64 weights (red positive,
  blue negative; snapshots every 10 steps); the ten probabilities.
* **Training:** accuracy on the 540 unseen digits: 10.4% at step 0 (a guess), 90% at step 31, 96.7% at step 300
  (522 right, 18 wrong). The strip shows one unseen example of each digit with the network's answer at that step.
* **The blame on the pixels:** for an unseen 3, the gradient of log p(3) with respect to each pixel, carried back
  through both layers by backpropagation: red where more ink would make "3" more likely, blue where it would make it
  less likely (the ink drawn as white squares).
* **A mistake:** an unseen 4 with a closed top, read as a 9 (p = 0.84; p(4) = 0.11). The strip shows 10 of the 18
  mistakes, true → read as.

## Honesty notes

* Over six other random splits and starts the final accuracy was 96.5–97.8% (median 97.1%); the run shown (96.7%) is
  typical.
* The 1,797 digits come from 13 writers, and the split is by digit, not by writer: the test digits are unseen, but
  their writers are not.
* The neuron pictures are the real weights. They change a lot (on average by about twice their starting size) but
  don't look like clean pen strokes; the narration says only that each neuron now looks for its own pattern of pixels.
* 8 × 8 pixels is much coarser than the famous MNIST digits (28 × 28); this reel uses the small set only.

## Caption (paste as-is)

```
This network learned to read handwriting. Here's exactly how.

1,797 real handwritten digits, scanned at 8 × 8 pixels. Each pixel is one input: 64 numbers. They go through 32 neurons and come out as 10 probabilities, one for each digit.

Train it on 1,257 digits and test it on 540 it never sees. At the start it's guessing: 10.4%. By step 31: 90%. After 300 steps: 522 of the 540 unseen digits right, 96.7%.

Then the part people skip: backpropagation doesn't stop at the first layer. Send the blame all the way back, and every pixel finds out how it pushed the answer. Red: more ink here would make it more sure. Blue: less.

It still makes 18 mistakes, like this 4 with a closed top, which it reads as a 9 (84% sure).

64 pixels, 32 neurons, 10 probabilities, and backpropagation. That's a network learning to read.

(Every probability, picture and score on screen is computed from the real network as it trains.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #neuralnetworks #deeplearning #backpropagation #ai #handwriting #datascience
```

## YouTube

Title:

```
Watch a Neural Network Learn to Read Handwriting #shorts
```

Description:

```
A small neural network learns to read handwritten digits, trained live.

The UCI handwritten digits (1,797 digits by 13 writers, 8 × 8 pixels), split into 1,257 for training and 540 held out. Network: 64 inputs → 32 ReLU neurons → 10 softmax outputs, cross-entropy, Adam, 300 full-batch steps. Accuracy on the 540 held-out digits: 10.4% at the start, 90% by step 31, 96.7% at step 300 (522 of 540; median of six other splits and starts: 97.1%).

Backpropagation carried all the way to the input gives the gradient of log p(correct digit) with respect to every pixel: which pixels pushed the answer which way. One of the 18 mistakes: a 4 with a closed top, read as a 9 with probability 0.84.

Data: UCI Optical Recognition of Handwritten Digits (Alpaydin & Kaynak, 1998), CC BY 4.0, as shipped with scikit-learn.

Voice: Kokoro, an open-source text-to-speech model (am_michael voice).

#neuralnetwork #machinelearning #deeplearning #shorts
```

## Pinned comment

```
The pixel blame is the same backpropagation that trains the network, run one step further: instead of stopping at the first layer's weights, you ask how the answer would change if each pixel had a little more ink. It's how "saliency maps" work in image models, and it's a quick way to see what a network is actually paying attention to.
```
