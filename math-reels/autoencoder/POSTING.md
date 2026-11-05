# Posting kit — *Every digit, squeezed to two numbers (an autoencoder).* (dark theme, narrated, 75 s)

## Files

* Reel: `output/autoencoder_reel.mp4` — 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (narration over the gentle track).
* Cover: `output/cover.png` — the 540 unseen digits on the network's 2-number map, coloured by digit, with an unseen
  digit, its 2 numbers and the digit rebuilt from them.
* Runs: `ae_results.json` (from `python aesim.py`, about two minutes; `ae_results.npz` is rebuilt by the scene if missing).
* Voice: Kokoro (open-source text-to-speech, kokoro-onnx v1.0), `am_michael` at 1.05×. Script in `scene.py` (`SCRIPT`).

## What's on screen (all of it is real)

The UCI handwritten digits of reel 49: the same 1,257 to train on and the same 540 unseen. Neither method below ever
sees a label; the labels are used afterwards, only to colour the map and to score it.

* **The network (an autoencoder):** 64 pixels → 64 ReLU neurons → 2 numbers → 64 ReLU neurons → 64 pixels (sigmoid),
  trained with backpropagation and Adam (step 0.003, all training digits at once, 3,000 steps) only to rebuild its
  input: L = mean of (rebuilt pixel − pixel)².
* **The map:** each unseen digit's 2 numbers as a point; shown at 100 moments of training (each map rescaled to fit
  the panel), then coloured by the real digit. The 0s, 4s and 6s each take their own region; 5, 7 and 8 overlap in the
  middle (share of each unseen digit's 5 nearest neighbours among the other unseen digits that are the same digit: 4: 91%, 6: 86%, 0: 83%, 2: 73% …
  5: 45%, 8: 31%).
* **Against PCA** (reel 43; 2 directions keeping 28.5% of the spread): for each unseen digit, do its 5 nearest
  training digits on the map (a vote) name the right digit? PCA: 62.8%. The network: 72.2%. Rebuild error (mean
  squared, pixels 0–1): PCA 0.052, the network 0.040, 24% less.
* **Backwards:** a 9 × 9 grid of points across the map (the middle 96% of the unseen digits' codes), each fed to the
  second half of the network: the digit that "lives" at each point.
* **The strip:** ten unseen digits and their rebuilds from 2 numbers (by the network; by PCA during the comparison).

## Honesty notes

* The run shown is the median of 9 random starts; the 5-neighbour score over the 9 runs: 68.1–81.3%. PCA has no
  randomness (62.8%).
* PCA does a decent job too: the gap is real (about 9 points) but not a landslide, and the narration says so.
* "Found its own corner" describes the map shown: 0s at the top, 4s at the right, 6s at the bottom.

## Caption (paste as-is)

```
This network squeezes every handwritten digit into just 2 numbers. Nobody ever told it which digit is which.

64 pixels go in, 2 numbers in the middle, 64 pixels come out. Its only job: rebuild the digit it was given. That's an autoencoder.

So every digit becomes one point on a map. At the start, the map is a blob. Backpropagation trains it to rebuild better and better.

Then colour each point by its real digit. The network never saw a label, and still the 0s, 4s and 6s each found their own corner. (The 5s, 7s and 8s still crowd the middle: 2 numbers isn't much room.)

PCA, the straight-line squeeze from the flowers reel, does a decent job too. A simple test: do a digit's 5 nearest neighbours on the map name it? PCA: 63%. The network, with its bends: 72%.

And the map works backwards: pick any point, and the second half of the network draws the digit that lives there.

(Every point, score and picture on screen is computed from the real networks.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #autoencoder #neuralnetworks #deeplearning #ai #datascience #unsupervisedlearning
```

## YouTube

Title:

```
Every Digit, Squeezed to 2 Numbers (An Autoencoder Learns a Map) #shorts
```

Description:

```
An autoencoder learns a 2-D map of handwriting without labels.

The UCI handwritten digits (8 × 8 pixels; 1,257 for training, 540 held out). Network: 64 → 64 ReLU → 2 (linear bottleneck) → 64 ReLU → 64 sigmoid, trained with backpropagation and Adam to minimize reconstruction error only. Plotting the 540 held-out digits' 2-D codes and colouring them by label afterwards: 0, 4 and 6 form their own regions; 5, 7 and 8 overlap.

Compared with PCA to 2 dimensions (28.5% of the variance): 5-nearest-neighbour classification on the 2-D codes is 62.8% for PCA and 72.2% for the autoencoder (median of 9 random initializations; range 68.1–81.3%), with 24% lower reconstruction error. Decoding a 9 × 9 grid of latent points shows the digit at each place on the map.

Voice: Kokoro, an open-source text-to-speech model (am_michael voice).

#autoencoder #machinelearning #deeplearning #shorts
```

## Pinned comment

```
Why the network beats PCA here: PCA can only squeeze along straight lines (it's a projection, like the shadow in the least-squares reel). The autoencoder's ReLU neurons let it bend the map, so digits that a straight-line squeeze piles together can be pulled apart. Scaled up, with a little randomness added, the same bottleneck is a variational autoencoder, the part of image generators like Stable Diffusion that compresses pictures before the generating happens.
```
