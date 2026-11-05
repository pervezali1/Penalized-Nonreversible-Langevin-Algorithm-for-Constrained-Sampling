# Posting kit — *I deleted 60% of a neural network.* (dark theme, narrated, 64 s)

## Files

* Reel: `output/prune_reel.mp4` — 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (narration over the gentle track).
* Cover: `output/cover.png` — the 32 neurons with 60% of the weights deleted (black), 947 of 2,368 left, still 96.7%;
  the three accuracy curves.
* Runs: `prune_results.json` (from `python prunesim.py`, a couple of seconds).
* Voice: Kokoro (open-source text-to-speech, kokoro-onnx v1.0), `am_michael` at 1.05×. Script in `scene.py` (`SCRIPT`).

## What's on screen (all of it is real)

The handwriting network of reel 49 (64 → 32 → 10; 64 × 32 + 32 × 10 = 2,368 weights, biases kept; 96.7% of the 540
unseen digits right), retrained exactly. A share of the weights is set to zero ("deleted"), three ways:

| Deleted | At random (median of 20) | Smallest first | Smallest first + 100 retraining steps |
| --- | --- | --- | --- |
| 0% | 96.7% | 96.7% | 96.7% |
| 30% | 63.3% | 95.6% | 95.9% |
| 50% | 40.4% (24–53%) | 90.0% | 96.5% |
| 60% | 33.1% | 86.9% | 96.7% |
| 70% | 21.4% | 84.1% | 95.6% |
| 80% | 21.6% | 55.6% | 93.1% |
| 90% | 13.8% | 16.9% | 77.6% |
| 95% | 11.1% | 16.1% | 37.0% |

* "Smallest first": the weights with the smallest |w|, both layers together. "Retraining": 100 Adam steps (step 0.01) on
  the 1,257 training digits, with the deleted weights held at zero.
* The neuron pictures are the 64 first-layer weights of each neuron (red +, blue −, black: deleted), after retraining
  where the reel says so. The random picture is one of the 20 random choices (40.2%, the closest to the median).
* The strip: what the network reads for ten unseen digits at each stage.

## Honesty notes

* "Exactly as well as before" means the same score, 96.7% (522 of 540); not necessarily the same 522 digits.
* Retraining is a second, short round of training; the deleted weights get no say in it. Without retraining, deleting
  the smallest 60% leaves 86.9%.
* Real systems prune in more careful ways (gradually, by structure, with longer retraining); this is the simplest
  version, on a small network.

## Caption (paste as-is)

```
I deleted 60% of this network's weights. It still reads 96.7% of handwriting, exactly as well as before.

It's the network from the handwriting reel: 2,368 weights, 96.7% right on digits it has never seen.

Delete half of them at random and it falls apart: 40%.

Delete the smallest half instead, the weights closest to zero, and it still reads 90%. Most of its weights were barely doing anything.

Then let the survivors retrain for 100 steps, with the deleted ones held at zero. 60% gone: 96.7%, the same as the full network. 80% gone: 93%.

But there's a limit. Delete 90% and even after retraining it reads only 78%.

That's pruning: one of the tricks that shrink big networks to fit on a phone.

(Every weight, picture and score on screen is computed from the real network.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #neuralnetworks #deeplearning #ai #pruning #datascience #modelcompression
```

## YouTube

Title:

```
I Deleted 60% of a Neural Network. It Didn't Notice. #shorts
```

Description:

```
Magnitude pruning, measured on a real handwriting network.

The network (64 → 32 → 10, 2,368 weights, 96.7% on 540 held-out UCI digits). Deleting half the weights at random leaves 40.4% (median of 20 random masks); deleting the half with the smallest magnitude leaves 90.0%. With 100 steps of fine-tuning and the pruned weights held at zero: 60% pruned recovers 96.7%, 80% pruned 93.1%, 90% pruned 77.6%, 95% pruned 37.0%.

Pruning, along with quantization and distillation, is one of the standard ways to make trained networks smaller and faster.

Voice: Kokoro, an open-source text-to-speech model (am_michael voice).

#machinelearning #neuralnetwork #pruning #shorts
```

## Pinned comment

```
Why the smallest weights can go: a weight's effect on a neuron is the weight times the pixel's ink, so a weight near zero barely moves anything, whatever the input. Random deletion throws away big, important weights just as often as tiny ones. And the short retrain lets the surviving weights adjust to cover for the missing ones, which is why 60% can go with no loss at all.
```
