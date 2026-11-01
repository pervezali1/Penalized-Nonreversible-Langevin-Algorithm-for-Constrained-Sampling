# Posting kit — *The vanishing gradient, measured.* (dark theme, 55 s)

## Files

* Reel: `output/vanishing_reel.mp4` — 55.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (gentle track).
* Cover: `output/cover.png` — both ladders of starting gradients, "Sigmoid: layer 1 gets 16 million times less."
* Simulation archive: `vanishing_sim.npz` (regenerate with `python vgsim.py`; not committed, 3.7 MB; the scene
  regenerates it if it is missing).

## What's on screen (all of it is real)

Two networks on the two-arm spiral of reel 2 (400 points, `neurons/train.py`): 2 inputs → 10 hidden
layers of 16 neurons → 1 output, binary cross-entropy. The only difference is the fold: sigmoid on the
left, ReLU on the right, each with its standard random start (sigmoid: weights N(0, 1/fan-in); ReLU:
N(0, 2/fan-in); biases 0). Of six seeds per activation, the run with the (lower) median final accuracy is
shown: sigmoid seed 3, ReLU seed 0.

**The ladder (5–19 s).** For each of the 11 weight layers (11 next to the output at the top, 1 next to
the input at the bottom), the size of the gradient of the loss averaged over all 400 points (RMS over the
layer's weights) at the start, filled in from the top down as the blame travels backward. Log-scale bars,
each tick 10×. Between the sigmoid rows: the factor it shrank by.

| Layer | 11 | 10 | 9 | 8 | 7 | 6 | 5 | 4 | 3 | 2 | 1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sigmoid | 4.9e-2 | 2.3e-3 | 6.6e-4 | 1.6e-4 | 2.7e-5 | 6.8e-6 | 1.5e-6 | 3.0e-7 | 6.0e-8 | 1.4e-8 | 3.1e-9 |
| ReLU | 9.4e-3 | 2.5e-3 | 1.8e-3 | 1.5e-3 | 1.6e-3 | 2.2e-3 | 1.6e-3 | 4.0e-3 | 3.2e-3 | 2.8e-3 | 3.6e-3 |

Sigmoid: layer 1 gets 6.4 × 10⁻⁸ of layer 11's blame, about **16 million times less** (geometric average
5.2× per layer). ReLU: layer 1 gets **38%** of layer 11's. The sigmoid factors are 21.6 for the first step
(from the single output neuron into layer 10: one output weight of typical size 0.25 times σ′ ≈ 0.24) and
3.5–5.8 for every step after (16 weights of size ~0.25 add up to about 1, times σ′ ≈ 0.24).

Ten-seed median of the starting gradients (not just the shown seed): sigmoid 3.3e-2 at layer 11 down to
3.3e-9 at layer 1 (10 million times less); ReLU between 2.4e-3 and 7.7e-3 at every layer.

**Why (19.6–27.6 s).** σ′(z) = σ(z)(1 − σ(z)) is never above 0.25; the measured average over every
sigmoid neuron and point at the start is 0.24 (per layer 0.225–0.241). ReLU′(z) is 1 for a neuron that
is on and 0 for one that is off; 51% of the ReLU neurons are on, on average (37–63% per layer), and the
ReLU start (variance 2/fan-in) is chosen to make up for the half that are off.

**The forward half (27.6–32.6 s).** The same shrink happens on the way up. The ladder switches to how
different the 400 points still are after each layer at the start (the spread across points, averaged over
the layer's neurons; row 11 is the output), filled from the input up:

| After layer | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | out |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sigmoid | 9.9e-2 | 2.6e-2 | 6.1e-3 | 1.8e-3 | 4.1e-4 | 1.0e-4 | 2.3e-5 | 4.2e-6 | 8.0e-7 | 1.8e-7 | 2.4e-7 |
| ReLU | 2.6e-1 | 2.1e-1 | 2.2e-1 | 1.1e-1 | 9.3e-2 | 1.1e-1 | 1.0e-1 | 7.0e-2 | 1.1e-1 | 1.1e-1 | 1.6e-1 |

(The input spread is 0.46.) By layer 10 the sigmoid network sees all 400 points as one: at the start it
gives every point the same guess, 0.594 (identical to 6 decimals). The ReLU network's guesses range from
0.50 to 0.70 at the start.

**Training (32.6–47.6 s).** Both trained with plain full-batch gradient descent, step size 0.1, 3,000
steps. The ladder switches to how far each layer's weights have moved, |W − W₀| / |W₀|, and the output
fields train live (snapshots every 25 steps), with a colour key (sure blue · 50/50 · sure vermilion).
While a field gives every point the same guess, a label on it says so ("EVERY POINT 0.594", then
"0.500", "no colour = 50/50"): the sigmoid field stays the neutral 50/50 colour because the only thing
its training can do is move that one shared guess from 0.594 to 0.500, which takes the loss from 0.711
to 0.693 within the first few hundred steps.

| | Sigmoid | ReLU |
| --- | --- | --- |
| Layer 1 moved | 0.00014% | 66% |
| Layer 11 moved | 16% | 49% |
| Final loss | 0.693 (= log 2, a coin flip) | 0.023 |
| Every guess between | 0.49999985 and 0.50000014 | 0.00001 and 0.99996 |
| Points right | 206 / 400 (meaningless: every guess is 0.500) | 396 / 400 (99%) |

All six seeds: sigmoid final accuracy 44–61% with loss 0.693 every time; ReLU 90.5–99.5% (median 99.0%).

## Honesty notes

* Re-checked (after a viewer-style question, "did the sigmoid really not move?"): the backpropagated
  gradients match finite differences to 4 digits (including the ~10⁻⁹ ones in layer 1), and the result
  does not change with more training: plain descent for 30,000 steps at step 0.1 or 1, and Adam (step
  0.01) for 30,000 steps, all end at loss 0.6931 with every guess 0.5000.
* Two halves of one problem: the forward signal fades (the points blur together) and the backward blame
  fades (the early layers can't learn to un-blur them). Both are the same multiplication by
  weights × σ′ at every layer.

* Plain gradient descent on purpose: the update is proportional to the gradient, so the vanishing
  gradient shows up directly as layers that don't move. Step sizes 0.3, 1 and 3 were also tried for the
  sigmoid net: none leaves loss 0.693 in 3,000 steps (at 3 the last layers move more, the first still
  ~0.001%). Adam rescales each weight's step and hides part of the problem: the same sigmoid net with
  Adam (step 0.01, 3,000 steps) reached only 59–64% on three seeds.
* The fix in practice is not only ReLU: careful starting scales, normalisation layers and skip
  connections (ResNets) all exist to keep the blame alive on the way down.
* Both networks start from their own standard initialisation; with the same initialisation the sigmoid
  gradients still shrink, since σ′ ≤ 0.25 at every layer.

## Caption (paste as-is)

```
Ten layers deep, the first layer can't hear the error.

Two networks with the same shape on the same spiral: 10 hidden layers of 16 neurons. The only difference is the fold each neuron uses (and the standard random start that goes with it). Left: sigmoid. Right: ReLU.

Watch the blame flow backward from the output. Every number is the real gradient each layer receives at the start.

Sigmoid: it shrinks at every layer. 0.049 at the top… 0.0000000031 at the bottom. Layer 1 gets 16 million times less than layer 11.

ReLU: layer 1 still gets 38% of what layer 11 gets.

Why? Going down one layer, the blame is multiplied by the weights and by the slope of the fold. A sigmoid's slope is never more than 0.25 (here it averages 0.24), so every layer divides the blame by about 4 to 5. Ten layers: ten divisions. ReLU's slope is exactly 1 for any neuron that is on, so the blame passes through whole.

And it happens on the way up too: each sigmoid layer shrinks the differences between points the same way. By layer 10, all 400 points look identical, and the network gives every one of them the same guess (0.594).

Then I trained both with plain gradient descent for 3,000 steps. Sigmoid: layer 1 moved 0.00014%. The one shared guess drifted to 0.500 and stayed there: no colour on its map, no opinion about any point. ReLU: every layer moved, layer 1 by 66%, and it learned the spiral: 396 of 400 right.

That's the vanishing gradient, and it's a big reason deep networks use ReLU (plus careful starts and skip connections).

(Real gradients, averaged over all 400 points, checked against brute-force nudges. Median of 6 random seeds per network; every sigmoid seed stayed at a coin flip, every ReLU seed reached 90% or more. Ten times more training, or Adam, doesn't rescue the sigmoid.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #deeplearning #neuralnetworks #vanishinggradient #relu #ai #math
```

Short alternative:

```
The vanishing gradient, measured. Two 10-layer networks, same data. Sigmoid: the blame shrinks about 5× per layer and layer 1 gets 16 million times less than layer 11; going up, the points blur until all 400 get one guess. ReLU: layer 1 still gets 38%. Train both: sigmoid's first layer moves 0.00014% and its one guess sits at 0.500. ReLU: 99% right.

Follow for more.

#machinelearning #deeplearning #relu #ai
```

## YouTube description

```
The vanishing gradient, measured on two real networks.

Two networks on the two-arm spiral (400 points): 10 hidden layers of 16 neurons each. Left: sigmoid neurons. Right: ReLU neurons. Each starts from its standard random initialisation.

The ladder shows the size of the gradient each of the 11 weight layers receives at the start (averaged over all 400 points), filled in from the output down as the blame flows backward. Sigmoid: 4.9 × 10⁻² at layer 11 down to 3.1 × 10⁻⁹ at layer 1, 16 million times less, shrinking 3.5 to 5.8× at every hidden layer. ReLU: layer 1 still gets 38% of layer 11's gradient.

Why: going down one layer multiplies the blame by the weights and by the slope of the activation. σ′(z) never exceeds 0.25 (measured average here: 0.24), so the blame divides by about 4 at every sigmoid layer. ReLU′(z) is exactly 1 for a neuron that is on, so the blame passes through.

Going up, the same shrink blurs the points: the spread between the 400 points falls from 0.46 at the input to 2 × 10⁻⁷ at sigmoid layer 10, so the sigmoid network gives every point the same guess (0.594) before training even starts. ReLU keeps them about 0.1 apart all the way up.

Training: plain gradient descent, step size 0.1, 3,000 steps. How far each layer's weights moved: sigmoid layer 1, 0.00014%; ReLU layer 1, 66%. Sigmoid ends at loss 0.693 (log 2, a coin flip) with every guess equal to 0.500; ReLU ends at loss 0.023 and 396 of 400 points right.

Median of 6 seeds per network: every sigmoid seed stayed at loss 0.693; ReLU seeds reached 90.5% to 99.5%. Plain gradient descent shows the effect most directly; Adam rescales steps and hides part of it (the same sigmoid net with Adam reached only 59–64%).

#machinelearning #deeplearning #vanishinggradient #relu
```

## Pinned comment

```
Where the 16 million comes from: the chain rule multiplies one factor per layer. Each factor is (weights) × σ′(z). σ′(z) = σ(z)(1 − σ(z)) peaks at 0.25 when z = 0, and here it averages 0.24. With the standard start, the 16 weights feeding back into each neuron add up to about 1, so every sigmoid layer divides the blame by about 4 to 5. Ten of those: 5.2¹⁰ ≈ 16 million.

ReLU′ is 1 for a neuron that is on and 0 for one that is off. About half are on, and the ReLU start (weights with variance 2/fan-in instead of 1/fan-in) doubles the signal to make up for the half that are off, so the blame keeps its size on the way down.

Fair question: is plain gradient descent rigged? It is the cleanest way to see it, because each weight moves in proportion to its gradient. Adam rescales every weight's step and partly hides the problem, but the same sigmoid network with Adam still only reached 59–64%. Real deep networks fix it with ReLU-like activations, careful starts, normalisation layers and skip connections.
```

## Audio

Gentle register, −16 LUFS, true peak under −1 dBTP, circular over 55 s. A soft chord bed; as the blame
reaches each layer, two plucks, one per network: the sigmoid's falls 3 dB for every factor of 10 its
gradient loses (it fades out as the blame vanishes), the ReLU's stays; a soft sweep as each slope curve
draws; plucks rising up the layers for the forward half, the sigmoid's fading as its points blur; a
quiet tick every few training snapshots; a chime at the end of training for each network; bells
at the headings.
