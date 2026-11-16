# Posting kit — *What a neural network looks like in 3D.* (3D stage, no narration, 30 s)

## Files

* Reel: `output/neurons3d_reel.mp4` — 1080×1920, 60 fps, H.264 CRF 16, AAC, −14 LUFS (sound driven by the network).
* Cover: `output/cover.png` — the output surface late in training, the points riding on it, step 1,080, 95.2% right.
* Network: `../nnbuild/nnbuild_train.npz` (the 8 + 8 run of reels 31–32; `python ../nnbuild/nnsim.py`).

## What's on screen (all of it is real)

The 2 → 8 → 8 → 1 ReLU network of reels 31–32, trained with Adam (step 0.01) for 1,500 steps on 400 points in two
spiral arms. Every surface is that network's output over the input plane, drawn in 3D under a camera that turns once in
30 s.

* **Training (0–7 s):** height = the network's probability that a point belongs to the orange arm, from the 101 saved
  snapshots (one every 15 steps). The 400 points ride on the surface. Accuracy: 56.2% at step 0 → 95.5% at step 1,500
  (382 of 400).
* **Layer 1 (7–13 s):** each neuron is max(0, w₁x₁ + w₂x₂ + b): a flat tilted sheet, cut off at zero along one line
  (dashed). Drawn exactly as its two flat pieces.
* **Layer 2 (13–19 s):** each neuron is max(0, a weighted sum of the 8 layer-1 sheets + d): folds of folds.
* **The sum (19–27 s):** the 8 layer-2 surfaces, multiplied by their output weights, added one at a time (in the order
  that helps the accuracy most: m8, m2, m3, m6, m1, m5, m4, m7) and squashed to 0–1:

| Neurons added | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Accuracy | 50.0% | 56.8% | 65.3% | 70.0% | 72.0% | 70.5% | 69.3% | 88.5% | 95.5% |

## Honesty notes

* Each tile is one neuron's output scaled to its own maximum, so the shapes compare but the heights don't (the real
  maxima: 4.1 to 8.0 for layer 1, 12.6 to 54.1 for layer 2).
* The big surface is drawn on an 80 × 80 grid and smoothed over about one grid cell, because its cliffs are nearly
  vertical; the accuracy, the points right and the order are exact. During training the points' heights are read off
  the saved 96 × 96 snapshots.
* The adding order is chosen to help most at each step; in another order the in-between numbers differ, and with this
  one the accuracy dips (72.0% → 69.3%) before the last two neurons lift it to 95.5%.

## Caption (paste as-is)

```
This is a neural network learning, in 3D. The height is how sure it is that a point is orange.

400 points, two spiral arms. At the start the surface is almost flat: 56% right. 1,500 training steps later, the orange arm sits on a plateau and the blue arm in the valley: 95.5%.

Now take it apart. Layer 1: 8 neurons. In 3D, each one is just a flat sheet with a single fold.

Layer 2: 8 more. Each one takes a mix of those sheets and folds it again.

Add layer 2 back up, one neuron at a time: 56.8%, 65.3%, 70%, 72%, 70.5%, 69.3%, 88.5%, 95.5%.

No single neuron knows the spiral. The sum does.

Which layer surprised you? 👇

(A real 2 → 8 → 8 → 1 network. Every surface is computed from its actual weights.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #neuralnetworks #deeplearning #ai #datascience #relu #3d
```

## YouTube

Title:

```
What a Neural Network Looks Like in 3D #shorts
```

Description:

```
A real neural network, every neuron drawn as a 3D surface.

A 2 → 8 → 8 → 1 ReLU network trained with Adam for 1,500 steps on 400 points in two spiral arms. Height = the network's probability that a point belongs to the orange arm: 56.2% right at the start, 95.5% (382 of 400) at the end. Each layer-1 neuron, max(0, w·x + b), is a flat sheet folded once; each layer-2 neuron folds a weighted mix of those sheets again. Adding the layer-2 surfaces back one at a time (weighted, then squashed by a sigmoid): 56.8, 65.3, 70.0, 72.0, 70.5, 69.3, 88.5, 95.5%.

#machinelearning #neuralnetwork #deeplearning #shorts
```

## Pinned comment

```
Why is every first-layer neuron a flat sheet with one fold? A neuron computes w₁x₁ + w₂x₂ + b, which over the plane is a tilted flat sheet, and ReLU sets everything below zero to zero: that's the fold. The second layer adds 8 of those sheets with its own weights and folds again, so its creases can bend. The output adds 8 of those. Depth isn't magic: it's folds of folds.
```
