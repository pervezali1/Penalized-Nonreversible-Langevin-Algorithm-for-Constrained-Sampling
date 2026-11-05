# Posting kit — *How to fool a neural network.* (dark theme, narrated, 72 s)

## Files

* Reel: `output/adversarial_reel.mp4` — 72.3 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (narration over the gentle track).
* Cover: `output/cover.png` — the unseen 3 (99.9% "3"), the two-shade nudge, and the nudged digit read as a 9 (99.2%).
* Runs: `adv_results.json` (from `python advsim.py`, about a second; it retrains the network of reel 49 exactly and
  checks it is the same network).
* Voice: Kokoro (open-source text-to-speech, kokoro-onnx v1.0), `am_michael` at 1.05×. Script in `scene.py` (`SCRIPT`).

## What's on screen (all of it is real)

The handwriting network of reel 49 (64 pixels → 32 ReLU neurons → 10 probabilities, trained on 1,257 of the UCI
digits; 522 of the 540 unseen digits right, 96.7%). A pixel's ink is a whole number from 0 to 16 ("shades").

* **The 3:** the unseen 3 of reel 49, read as a 3 with p = 0.9992.
* **The direction:** backpropagation carried to the pixels gives the gradient of L = −log p(3) for each pixel; its sign
  says which way each pixel would have to move to make "3" less likely (red: add ink, blue: take ink away). Pixels
  that can't move that way (already 0 or 16) are left dark.
* **The nudge** (the "fast gradient sign" attack, Goodfellow, Shlens & Szegedy 2014): every pixel moved k shades that
  way, kept inside 0–16. k = 1: still a 3, p(3) = 0.63. k = 2: p(3) = 0.008, p(9) = 0.992. 46 of the 64 pixels change.
* **Any answer:** toward each digit d, 40 small steps of one shade along the gradient of −log p(d), never more than k
  shades from the original 3, with the smallest k that works: the same 3 is read as each of 0–9, with k from 2 to 4.
  (For "8" it only just wins: p = 0.44.)
* **All 540 unseen digits**, each nudged along its own gradient, against random nudges of the same size (median of 20
  random draws):

| Shades | Right, along the gradient | Right, random |
| --- | --- | --- |
| 0 | 522 | 522 |
| 1 | 339 | 513 |
| 2 | 71 | 489 |
| 3 | 4 | 450 |
| 4 | 0 | 374 |

  The smallest single nudge that fools each of the 522 correctly read digits: 1 shade for 183, 2 for 268, 3 for 67,
  4 for 4.

## Honesty notes

* The attack knows the network's weights (it needs the gradient): this is the "white-box" setting.
* On 8 × 8 pixels with 16 shades, one shade is a visible step: the nudged digits look faintly hazy, but plainly still
  the same digit to a person. In large image networks the same attack works with changes too small to see (Szegedy et
  al. 2013; Goodfellow et al. 2014); this reel shows only the small network.
* Every altered digit is still a valid scan: whole shades, 0 to 16.

## Caption (paste as-is)

```
I changed this 3 by two shades of ink. The network is now 99% sure it's a 9.

This is the network from the handwriting reel: 96.7% right on digits it has never seen, and 99.9% sure this one is a 3.

Backpropagation can carry the blame all the way back to the pixels: for each one, which way would make "3" less likely. Push every pixel that way by one shade (out of 16): still a 3, but only 63% sure. Two shades: "a 9", 99.2% sure.

Pick the answer you want and push toward it instead: the same 3 can be read as every digit from 0 to 9. Never more than 4 shades.

And it's not just this 3. Push all 540 test digits 3 shades along their own gradients: 4 are still right. Push them the same amount in random directions: 450 are.

The gradient that teaches a network is also a map for fooling it. That's an adversarial example.

(Every digit, probability and count on screen is computed from the real network.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #neuralnetworks #adversarialexamples #deeplearning #ai #backpropagation #aisafety
```

## YouTube

Title:

```
How to Fool a Neural Network (Adversarial Examples) #shorts
```

Description:

```
An adversarial example, built step by step on a real handwriting network.

The network: 64 pixels → 32 ReLU neurons → 10 softmax outputs, trained on the UCI handwritten digits (8 × 8 pixels, ink 0–16), 96.7% on 540 held-out digits. For an unseen 3 (p = 0.999), backpropagation to the input gives the gradient of −log p(3) with respect to each pixel. Moving every pixel one shade along its sign (the fast gradient sign method, Goodfellow et al. 2014) drops p(3) to 0.63; two shades and the network reads a 9 with p = 0.992. Stepping toward a chosen digit instead, within 2–4 shades, makes the same 3 read as every digit 0–9.

Across all 540 held-out digits: 3 shades along each digit's own gradient leaves 4 right; 3 shades in random directions leaves 450 right. Every one of the 522 correctly read digits is fooled by a single nudge of at most 4 shades.

Voice: Kokoro, an open-source text-to-speech model (am_michael voice).

#adversarialexamples #neuralnetwork #machinelearning #shorts
```

## Pinned comment

```
Why random noise barely matters but the gradient's direction does: the network's output changes by roughly (gradient) · (change). A random nudge points every which way and mostly cancels out; a nudge along the sign of the gradient lines up with it in every pixel at once, so 64 small changes all push the same way. It's the reason Goodfellow and colleagues gave for why big images are easy to fool: more pixels, more small pushes adding up.
```
