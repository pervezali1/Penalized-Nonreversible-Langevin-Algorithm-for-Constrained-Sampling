# Posting kit — *The laziest algorithm just beat my neural network.* (dark theme, narrated, 78 s)

## Files

* Reel: `output/nearest_reel.mp4` — 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (narration over the gentle track).
* Cover: `output/cover.png` — a new 3, the closest of the 1,257 training pictures (difference 270), all 1,257 lit by
  closeness, and the two scores: network 96.7%, nearest neighbor 98.7%.
* Runs: `nn_results.json` (from `python nnsim.py`, under a minute).
* Voice: Kokoro (open-source text-to-speech, kokoro-onnx v1.0), `am_michael` at 1.05×. Script in `scene.py` (`SCRIPT`).

## What's on screen (all of it is real)

The handwriting network of reel 49 (64 → 32 → 10, 300 Adam steps, retrained exactly) against 1-nearest neighbor: keep
all 1,257 training pictures; for a new digit, compute difference = Σ over the 64 pixels of (ink difference)², ink in
shades 0–16, and copy the label of the training picture with the smallest difference. Nothing is trained. Both are
tested on the same 540 unseen digits.

| | Neural network | Nearest neighbor |
| --- | --- | --- |
| Right on the 540 unseen digits | 522 (96.7%) | 533 (98.7%) |
| 6 other random splits (1,257 / 540) | 96.5–97.8% | 98.3–99.6% (wins all 6) |
| Multiplications to read one digit | 64 × 32 + 32 × 10 = 2,368 | 1,257 × 64 = 80,448 (34×) |
| Numbers it keeps | 2,410 (weights and biases) | 80,448 (every pixel of every picture) |

* The mistake shown: an unseen 9 whose closest training picture is a 3 (difference 479); the closest training 9 is at
  866. The network reads that 9 correctly.
* Of the nearest neighbor's 7 mistakes, the network also gets 3 wrong; the network's 18 mistakes include 15 that
  nearest neighbor gets right.
* Voting among the 3, 5 or 7 closest pictures does slightly worse here: 98.0%, 98.0%, 97.6%.
* Four unseen digits have two equally close training pictures; in every case both carry the same label.

## Honesty notes

* "Beat" is on this data: 8 × 8 digits, small, clean and centered, where sharing pixels really does mean the same
  digit. On photos, pixel-by-pixel nearest neighbor does far worse than a trained network.
* It's not because the network is too small: in `nnsim.py`, 128 neurons for 300 steps reads 96.7%, 128 neurons for
  1,000 steps 96.5%, 256 neurons for 1,000 steps 97.0%. (A convolutional network would do better.)
* The digits come from 13 writers, and the 540 test digits share those writers, which favours memorizing.
* "Eighty thousand multiplications" rounds 80,448; "thirty-four times" rounds 33.97. Additions are not counted, for
  either method.
* "AI search and recommendations use them": systems that search by meaning look up the nearest neighbors of a query
  among stored feature vectors (embeddings) that a network computed; usually with approximate, faster search.

## Caption (paste as-is)

```
The laziest algorithm in machine learning just beat my neural network. And it doesn't learn anything at all.

My handwriting network trained for 300 steps. It reads 96.7% of digits it has never seen.

The lazy way: keep all 1,257 training pictures. For a new digit, find the one that looks most alike, pixel by pixel, and copy its answer. That's called nearest neighbor.

533 of the 540 unseen digits right: 98.7%. On 6 other random splits of the data, it won all 6.

The catch: to read one digit, it compares it with every picture it kept. That's 80,448 multiplications, 34 times the network's work, and it grows with every picture you add. With a million pictures: 64 million, every time.

So always try the simple method first. And nearest neighbors are still everywhere: AI search and recommendations use them every day, on features a network learned instead of pixels.

(Both were tested on the same 540 unseen digits. Every number on screen is computed.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #neuralnetworks #ai #datascience #knn #algorithms #deeplearning
```

## YouTube

Title:

```
The Laziest Algorithm in Machine Learning Beat My Neural Network #shorts
```

Description:

```
Nearest neighbor vs a neural network, on the same handwritten digits.

The handwriting network (64 → 32 → 10, trained with Adam for 300 steps) reads 522 of 540 held-out UCI digits (96.7%). 1-nearest neighbor, which keeps all 1,257 training digits and copies the label of the closest one (squared pixel difference), reads 533 (98.7%) with no training at all. On 6 other random train/test splits it wins all 6 (98.3–99.6% vs 96.5–97.8%). The cost: 1,257 × 64 = 80,448 multiplications per prediction against the network's 2,368 (34×), and it grows with the dataset.

Voice: Kokoro, an open-source text-to-speech model (am_michael voice).

#machinelearning #knn #neuralnetwork #shorts
```

## Pinned comment

```
Why does the lazy way win here? These digits are tiny (8×8), clean and centered, so "shares the same pixels" really does mean "same digit". On photos, two pictures of the same dog can share almost no pixels, and pixel-by-pixel nearest neighbor falls apart. That's why modern systems let a network turn each picture into features first, then find the nearest neighbors of those.
```
