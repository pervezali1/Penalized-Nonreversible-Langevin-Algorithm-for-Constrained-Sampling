# Posting kit — *Move the digit one pixel (why image networks use convolution).* (dark theme, narrated, 63 s)

## Files

* Reel: `output/convolution_reel.mp4` — 63.7 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (narration over the gentle track).
* Cover: `output/cover.png` — an unseen 0 moved one pixel: the handwriting network reads "2", the convolutional network
  still reads "0"; the accuracy by pixels moved for both (a V against a flat line).
* Runs: `cnn_results.json` (from `python cnnsim.py`, about five minutes: it trains the convolutional network six times).
* Voice: Kokoro (open-source text-to-speech, kokoro-onnx v1.0), `am_michael` at 1.05×. Script in `scene.py` (`SCRIPT`).

## What's on screen (all of it is real)

The UCI handwritten digits of reel 49 (8 × 8 pixels): the same 1,257 to train on, the same 540 unseen digits to test
on. Moving a digit sideways shifts its columns; the column that slides in is blank. (The leftmost and rightmost
columns hold 0.01% and 0.3% of all the ink, so almost nothing falls off the edge.)

* **The handwriting network** of reel 49 (64 → 32 ReLU → 10, 2,410 numbers), retrained exactly:

| Moved | 2 left | 1 left | not moved | 1 right | 2 right |
| --- | --- | --- | --- | --- | --- |
| Right, of 540 | 13.7% (74) | 39.1% (211) | 96.7% (522) | 46.5% (251) | 18.5% (100) |

* **Why:** each hidden neuron computes ReLU(Σ x_p W_pj + b_j), one weight per pixel position. Shown: neuron 12's 64
  weights with the 0's ink on top; its output 5.31, and 3.94 after the move.
* **The convolutional network** (650 numbers), trained on the same 1,257 unmoved digits the same way (Adam, step 0.01,
  full batch, 600 steps): 32 filters of 3 × 3 weights, each slid over all 64 positions (zero beyond the edge), ReLU,
  the strongest response of each filter anywhere in the digit (global max pooling), then 32 → 10 with softmax.
  32 × 9 + 32 + 32 × 10 + 10 = 650.

| Moved | 2 left | 1 left | not moved | 1 right | 2 right |
| --- | --- | --- | --- | --- | --- |
| Right, of 540 | 95.7% (517) | 95.9% (518) | 95.9% (518) | 96.5% (521) | 94.3% (509) |

* **The filter shown** (filter 19 of 32, the one whose strongest response votes most for "0" on this digit): its
  response at each of the 64 positions, before and after a one-pixel move right; the strongest response is 4.64 both
  times (one position further right). For this digit, all 32 filters' strongest responses are exactly unchanged by
  the one-pixel move.
* **The digit shown:** unseen digit #335, a 0. Of the 540, 178 are read right by the handwriting network, misread after
  one pixel either way, and read right by the convolutional network at every shift; this is the one (not a 1) the
  convolutional network is surest of. Handwriting network: 1 right → "2" (79.4%), 1 left → "4" (99.8%), 2 left → "1",
  2 right → "1".
* **The strip:** the ten unseen digits of reel 49, moved one pixel right: the handwriting network misreads 7 of them;
  the convolutional network reads all 10 right.

## Honesty notes

* Moving digits up or down hurts both: the digits fill the box from top to bottom (the top and bottom rows hold 12%
  and 13% of the ink), so a vertical move cuts ink off. One pixel up / down: handwriting network 40.9% / 41.7%,
  convolutional 59.6% / 71.7%. The reel shows only sideways moves.
* The convolutional network here is one layer with global max pooling, which throws away *where* a pattern was. Real
  image networks stack many convolution layers and pool gradually, and they are usually also trained on shifted and
  cropped copies of their images.
* Neither network saw a moved digit in training.
* Five other random starts for the convolutional network: 93.9–95.7% unmoved, and within 0.4 points of that after a
  one-pixel move either way.

## Caption (paste as-is)

```
This network reads handwriting: 96.7% right. Move every digit one pixel to the right: 46.5%.

Same digits, same network. One pixel to the left: 39.1%. This 0 becomes a 2, a 4, a 1.

Why: each of its neurons is wired to particular pixels. Move the ink by one pixel and it lands on different weights.

Image networks do something else. They use small filters, 3 by 3 pixels, and slide each one over every position in the image. Then they keep each filter's strongest response, wherever it was. Move the digit and the responses move with it; the strongest one doesn't change at all.

32 of these filters, trained on the same digits: 95.9%, and still 96.5% after a one-pixel move right (95.9% left). With about a quarter of the numbers: 650 vs 2,410.

That's convolution: a pattern means the same thing wherever it shows up.

(Every digit, probability and score on screen is computed from the two real networks.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #neuralnetworks #deeplearning #computervision #cnn #convolution #ai
```

## YouTube

Title:

```
Move the Digit One Pixel: Why Image Networks Use Convolution #shorts
```

Description:

```
Translation invariance, measured on real handwriting.

The handwriting network from the previous video (64 → 32 → 10, fully connected, 2,410 parameters) reads 96.7% of 540 held-out UCI digits (8 × 8 pixels). Shift every digit one pixel right and it drops to 46.5%; one pixel left, 39.1%; two pixels, 14–19%. Each hidden unit has one weight per pixel position, so moved ink meets different weights.

A small convolutional network trained on the same unshifted digits: 32 filters of 3 × 3, shared across all 64 positions, ReLU, global max pooling, 32 → 10 softmax (650 parameters). 95.9% unshifted; 96.5% shifted right one pixel, 95.9% left, 94.3–95.7% at two pixels. A shifted image shifts each filter's response map, and the maximum doesn't move.

Vertical shifts hurt both (the digits fill the box top to bottom, so ink is cut off): 41–42% vs 60–72%.

Voice: Kokoro, an open-source text-to-speech model (am_michael voice).

#cnn #convolution #machinelearning #shorts
```

## Pinned comment

```
Two ideas make convolution work here. Weight sharing: the same 9 weights are used at every position, so a filter that responds to one small pattern of ink responds to it anywhere in the image. And pooling: keeping only the strongest response throws away where it was. Real image networks pool more gently (a little at a time, over many layers) so they can still tell where things are, and they're also trained on shifted copies of their images. This one-layer version is the idea at its simplest.
```
