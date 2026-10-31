# Posting kit — *Backpropagation: how a network learns from one mistake.* (dark theme, no formulas on screen, 46 s)

## Files

* Reel: `output/backprop_reel.mp4` — 46.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (gentle track).
* Cover: `output/cover.png` — the 8 + 8 network with the blame glowing on every edge, the off neurons dark.
* Simulation archive: `backprop_sim.npz` (regenerate with `python bpsim.py backprop_sim.npz`; not committed, 125 KB).

## What's on screen (all of it is real)

Two networks from reel 31, trained exactly as there (`../nnbuild/nnsim.py`: the two-arm spiral of reel 2,
400 points, ReLU, Adam with step size 0.01, binary cross-entropy, the same seeds), stopped at a chosen
step (`bpsim.py`):

| | One neuron | 8 + 8 |
| --- | --- | --- |
| Taken at training step | 300 of 1,500 (converged: its accuracy never changes after this) | 150 of 1,500 (mid-training) |
| Right before the step | 249 of 400 (62.25%) | 293 of 400 (73.25%) |
| The example | point #396, x = (−0.87, +0.05), vermilion | point #305, x = (+0.68, −0.08), vermilion |
| Guess before (probability of vermilion) | 0.49 | 0.34 |
| Error (guess − answer) | −0.51 | −0.66 |
| Numbers in the network | 5 (3 weights, 2 biases) | 105 (88 weights, 17 biases) |
| Plain gradient step on that one example, size | 0.3 | 0.02 |
| Guess after | 0.62 | 0.72 |
| Right after the step | 255 of 400 (63.75%) | 298 of 400 (74.5%) |

For each network the reel shows the three moves of one training step:

* **Forward.** Gold pulses carry the example along the edges, layer by layer. Each hidden neuron is lit
  by its activation (any neuron that is on is visibly lit; the brightest in a layer is its largest). The
  output node fills with the guess's colour and the number appears next to it.
* **Error.** The guess minus the answer (the answer is 1: the point is vermilion). This single number is
  all the network learns from.
* **Backward.** Magenta pulses carry the blame back along the edges, layer by layer. Then every edge
  glows by the size of the gradient on its weight (how much of the error that weight is responsible for,
  on a square-root scale) and every neuron gets a magenta ring sized by the blame that reaches it. A
  neuron that was off (activation zero) passed nothing forward and so gets no blame back: its ring and
  its outgoing edges stay dark. In the 8 + 8 network 4 of the 8 first-layer neurons and 2 of the 8
  second-layer neurons are off for this example.
* **Nudge.** The weights move to their new values (weight − step size × gradient), with a white shimmer
  on each edge sized by how far it moved. The field, the fold lines, the guess and the count of points
  right are recomputed live from the tweened weights every frame, so the field on screen is never an
  interpolation of two pictures: at every frame it is the output of the network with the weights shown.

Edges are coloured by the sign of the live weight (vermilion positive, blue negative) and thickened by
its size (per layer). The example is ringed in gold on the field.

**One neuron, the actual numbers** (before → after the step): w₁ −0.004 → −0.142, w₂ 1.768 → 1.776,
b 0.474 → 0.632, output weight 1.032 → 1.119, output bias −0.627 → −0.474. Gradients: 0.460, −0.027,
−0.527, −0.290, −0.510. The neuron's fold line x₂ = 0.002·x₁ − 0.268 tilts to x₂ = 0.080·x₁ − 0.356
(it moves away from the example, which is at x₁ = −0.87, so the example lands more firmly on the "on"
side) and the output bias rises, so the whole field shifts toward vermilion: 6 more points right.

**8 + 8, the actual blame:** output delta −0.662; layer-2 deltas +0.47, 0, −1.26, +0.81, −0.56, −1.23,
+0.39, 0 (two off); layer-1 deltas 0, −1.38, −0.75, −1.93, 0, 0, 0, +4.18 (four off). Largest gradient on
a weight 2.86 (first layer), so the largest move is 0.02 × 2.86 = 0.057 on a weight of size 3.5: the
largest weight moves by under 2%; the largest move of any of the 105 numbers is 0.084 (a bias). The
guess still goes 0.34 → 0.72, because 105 small moves all push the same way.

No formulas on screen, by request; the maths lives here and in the pinned comment.

## Honesty notes

* Real training (reel 31) averaged the gradient over all 400 points and used Adam. Here one example is
  used and the step is plain gradient descent, so that the blame of *one* mistake can be followed. The step
  sizes (0.3 and 0.02) were chosen so the change is visible but small; they are stated on screen.
* The example is chosen, not random: among the vermilion points the network gets wrong (away from the
  data's edge, with at least one first-layer neuron on), the one whose blame on the input weights is
  largest. Any wrong point would show the same three moves.
* A step on one example improves that example and happens to improve the overall count too (249 → 255,
  293 → 298). That is not guaranteed in general: a step on one point can make others worse.
* The counts on screen are exact; the percentages in this file are the exact fractions of 400.

## Caption (paste as-is)

```
How does a network learn from one mistake?

Three moves. Every one of them is on screen, with the real numbers.

Forward: the example flows through the network and comes out as a guess. One neuron, one point it gets wrong: the guess is 0.49, the answer is 1.

Error: guess minus answer, −0.51. That single number is the only thing the network learns from.

Backward: the blame flows back along every edge. Each weight finds out how much of the error was its fault. That is backpropagation, and that is all it is: the error, handed back along the same wires it came out of.

Then every weight moves a little, against its blame. Five numbers, nudged once: the guess goes 0.49 → 0.62, and 6 more of the 400 points come out right.

Now the 8 + 8 network, mid-training. Same three moves. Watch the blame fan out through every path, and watch which neurons get none: the ones that were off. Nothing went through them, so nothing comes back. 105 numbers nudged once, none by more than 0.09, and the guess goes 0.34 → 0.72.

Forward: guess. Backward: blame. Nudge every weight. Repeat a few thousand times. That is all training is.

(Two networks from reel 31, stopped mid-training. One example each, one plain gradient step, step size stated on screen; real training averages over all the points and uses a smarter step. Every edge's glow is the real gradient on that weight; the field, the guess and the count of points right are recomputed from the live weights every frame.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #neuralnetworks #backpropagation #deeplearning #ai #math #explained
```

Short alternative:

```
Backpropagation in one reel. Forward: the example becomes a guess (0.49; the answer is 1). Error: −0.51. Backward: the blame flows back along every edge, each weight learning how much was its fault. Nudge every weight a little: the guess goes 0.49 → 0.62. Then the 8 + 8 network: the blame fans out through every path, skips the neurons that were off, and 105 numbers move at once: 0.34 → 0.72. Repeat a few thousand times. That is training.

Follow for more.

#machinelearning #neuralnetworks #backpropagation #ai
```

## YouTube description

```
One training step in slow motion, twice: on a one-neuron network and on the 8 + 8 network from the previous reel, both stopped mid-training on the two-arm spiral (400 points, ReLU, trained with Adam as before).

Forward: the example's two inputs flow along the edges (gold), each neuron lit by its activation, and come out as a guess, the probability that the point is vermilion. One neuron: guess 0.49, answer 1. 8 + 8: guess 0.34, answer 1.

Error: guess minus answer (−0.51 and −0.66). This is the only signal the network learns from.

Backward: the blame flows back along the edges (magenta). Every edge then glows by the size of the gradient on its weight, and every neuron gets a ring sized by the blame reaching it. A neuron that was off for this example passed nothing forward, so it gets no blame back: in the 8 + 8 network 4 of 8 first-layer neurons and 2 of 8 second-layer neurons are dark.

Nudge: every weight moves against its blame (weight − step size × gradient; step sizes 0.3 and 0.02, stated on screen). The output field, the fold lines, the guess and the count of points right are recomputed from the live weights on every frame. One neuron: 0.49 → 0.62, 249 → 255 of 400 right. 8 + 8: 0.34 → 0.72, 293 → 298 right, with none of the 105 numbers moving by more than 0.09.

Honesty notes: real training averages the gradient over all the points and uses Adam; here one example and a plain step are used so the blame of one mistake can be followed. The example is the wrong vermilion point whose blame on the input weights is largest. A step on one example is not guaranteed to help the others; here it did.

Same spiral as reels 2, 31 and 32. Code in the repo.

#machinelearning #neuralnetworks #backpropagation #deeplearning
```

## Pinned comment

```
What "blame" means, precisely: the gradient. For every weight, the question "if this weight were a little bigger, how much would the error change?" has an exact answer, and the chain rule gives it layer by layer, starting at the output and working back. That backward sweep is backpropagation. Its cost is about the same as one forward pass, which is the whole reason training big networks is possible.

Why off neurons get nothing: a ReLU neuron that is off outputs exactly 0 whatever its inputs do, so nothing downstream depends on its inputs for this example. Its incoming weights get gradient 0. They will get blame from other examples, where the neuron is on.

Honesty check: real training does this on a batch of examples at once and uses Adam, which rescales each weight's step. Here it is one example and a plain step of stated size (0.3 for the tiny net, 0.02 for the 8 + 8), so that one mistake's blame can be seen. Every number on screen is computed from the saved weights; the field is recomputed from the live weights on every frame, not faded between two pictures.
```

## Audio

Gentle register, −16 LUFS, true peak under −1 dBTP, circular over 46 s. A soft chord bed; a rising pluck as
the signal crosses each layer and a high note at the guess; a low thud at the error; a darker, falling
pluck as the blame crosses each layer back; a quick four-note shimmer when the weights move and a chime
at the end of the nudge pitched by the new accuracy; bells at the headings.
