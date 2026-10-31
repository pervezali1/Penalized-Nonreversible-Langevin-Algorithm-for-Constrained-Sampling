# Posting kit — *Backpropagation: how a network learns from one mistake.* (dark theme, every number on screen, 52 s)

## Files

* Reel: `output/backprop_reel.mp4` — 52.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (gentle track).
* Cover: `output/cover.png` — the 8 + 8 network with the blame on every neuron as a number, the off neurons at 0.
* Simulation archive: `backprop_sim.npz` (regenerate with `python bpsim.py backprop_sim.npz`; not committed, 125 KB).

## What's on screen (all of it is real)

Two networks from reel 31, trained exactly as there (`../nnbuild/nnsim.py`: the two-arm spiral of reel 2,
400 points, ReLU, Adam with step size 0.01, binary cross-entropy, the same seeds), stopped at a chosen
step (`bpsim.py`):

| | One neuron | 8 + 8 |
| --- | --- | --- |
| Taken at training step | 200 of 1,500 (its fold has settled; the finished run gets one more point right) | 150 of 1,500 (mid-training) |
| Right before the step | 248 of 400 (62%) | 293 of 400 (73.25%) |
| The example | point #396, x = (−0.87, +0.05), vermilion | point #305, x = (+0.68, −0.08), vermilion |
| Guess before (probability of vermilion) | 0.49 | 0.34 |
| Error (guess − answer) | −0.51 | −0.66 |
| Numbers in the network | 5 (3 weights, 2 biases) | 105 (88 weights, 17 biases) |
| Plain gradient step on that one example, size | 0.2 | 0.02 |
| Guess after | 0.57 | 0.72 |
| Right after the step | 252 of 400 (63%) | 298 of 400 (74.5%) |

A strip under the heading names the four moves (FORWARD · ERROR · BACKWARD · NUDGE) and lights the one
under way. For each network:

* **Forward.** Gold pulses carry the example along the edges. For the one-neuron net every weight sits
  on its edge as a number pill (w₁, w₂, the bias b, the output weight a, the output bias c) and a ledger
  under the diagram does the arithmetic as the signal arrives: the sum (each input × its weight, added,
  plus the bias), the fold (below zero would become 0; here 0.55 passes), the output (× the output weight,
  plus its bias, squashed to a probability). For the 8 + 8 net each neuron shows the number it passed on,
  0 for the ones that are off.
* **Error.** Guess minus answer (the answer is 1: the point is vermilion), in the ledger and in the readouts.
* **Backward.** Magenta pulses carry the blame back. For the one-neuron net a BLAME pill appears under
  every weight and the ledger shows where each number comes from: the blame on a weight is the blame
  arriving at it times the number that weight multiplied (so the bias, which multiplies 1, gets the blame
  itself); the blame passes back through the neuron because it was on. For the 8 + 8 net every neuron
  shows the blame that reached it, 0 for the ones that were off (4 of 8 in layer 1, 2 of 8 in layer 2),
  and every edge glows by the size of the gradient on its weight (square-root scale).
* **Nudge.** New = old − step size × blame, for every number at once. The pills tick from their old
  values to their new ones and the ledger shows the five updates; for the 8 + 8 net the neurons' numbers
  tick too. The field, the fold lines, the guess and the count of points right are recomputed from the
  tweened weights on every frame, so nothing on screen is faded between two pictures.

Edges are coloured by the sign of the live weight (vermilion positive, blue negative) and thickened by
its size (per layer). The example is ringed in gold on the field.

**One neuron, every number** (before → after): w₁ +0.011 → −0.082, w₂ +1.778 → +1.784, b +0.467 → +0.574,
a +1.040 → +1.096, c −0.624 → −0.522. Forward: sum 0.548, fold 0.548, logit −0.054, guess 0.486. Error
−0.514. Blame: on c −0.514, on a −0.282, through the neuron −0.534, on w₁ +0.467, on w₂ −0.027, on b
−0.534. The fold line x₂ = −0.006·x₁ − 0.263 tilts to x₂ = 0.046·x₁ − 0.322, and the output bias rises,
so the field shifts toward vermilion: the example crosses to the right side (0.49 → 0.57) and 4 more
points are right (248 → 252).

**8 + 8, the blame:** output −0.662; layer-2 deltas +0.47, 0, −1.26, +0.81, −0.56, −1.23, +0.39, 0; layer-1
deltas 0, −1.38, −0.75, −1.93, 0, 0, 0, +4.18. Largest gradient on a weight 2.86 (first layer); the largest
move of any of the 105 numbers is 0.084 (a bias), under 2% of the largest weight. The guess still goes
0.34 → 0.72, because 105 small moves all push the same way.

## Honesty notes

* Real training (reel 31) averaged the gradient over all 400 points and used Adam. Here one example is
  used and the step is plain gradient descent, so that the blame of *one* mistake can be followed. The step
  sizes (0.2 and 0.02) were chosen so the change is visible but small; they are stated on screen.
* Every number on screen is rounded to 2 decimals (the footer says so). The arithmetic is done with the
  full values, so a reader multiplying the rounded numbers can land one unit off in the last digit (for
  example −0.53 × −0.87 = 0.461 on screen reads +0.47, the full-precision −0.534 × −0.874 = +0.467).
* The example is chosen, not random: among the vermilion points the network gets wrong (away from the
  data's edge, with at least one first-layer neuron on), the one whose blame on the input weights is
  largest. Any wrong point would show the same four moves.
* A step on one example improves that example and happens to improve the overall count too (248 → 252,
  293 → 298). That is not guaranteed in general: a step on one point can make others worse. During the
  nudge the live count briefly dips (to 245 and 291) before the new weights settle; that is real too.
* The one-neuron net is shown at step 200 rather than at the end of training because its weights are
  easier to read there (at the end the weight on x₁ is −0.004, which rounds to −0.00 on screen); its
  count of points right is 248 at step 200 and 249 at the end.

## Caption (paste as-is)

```
How does a network learn from one mistake? Every number, on screen.

Four moves.

Forward: multiply by the weights, add, fold, squash. One neuron, one point it gets wrong: +0.01 × −0.87 + 1.78 × +0.05 + 0.47 = 0.55, which passes the fold; 1.04 × 0.55 − 0.62 = −0.05, squashed to a guess of 0.49. The answer is 1.

Error: 0.49 − 1 = −0.51. That single number is the only thing the network learns from.

Backward: the blame flows back along every edge, and the rule is one line. The blame on a weight = the blame arriving × the number that weight multiplied. The output bias multiplied 1, so it gets −0.51. The output weight multiplied 0.55, so it gets −0.28. Through the neuron (it was on, so the blame passes): −0.51 × 1.04 = −0.53. Then w₁ multiplied x₁ = −0.87, so it gets +0.47; w₂ multiplied +0.05, so it gets −0.03. That is backpropagation. All of it.

Nudge: new = old − 0.2 × blame, for all five numbers at once. The guess goes 0.49 → 0.57 and 4 more of the 400 points come out right.

Then the 8 + 8 network, mid-training, 105 numbers. Same four moves. Watch the blame fan out, and watch which neurons get 0: the ones that passed 0 forward. Nothing went through them, so nothing comes back. 105 numbers nudged once, none by more than 0.09, and the guess goes 0.34 → 0.72.

Forward, error, backward, nudge. Repeat a few thousand times. That is all training is.

(Two networks from reel 31, stopped mid-training. One example each, one plain gradient step, step size on screen; real training averages over all the points and uses a smarter step. Every number is computed from the saved weights and rounded to 2 decimals; the field, the guess and the count of points right are recomputed from the live weights every frame.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #neuralnetworks #backpropagation #deeplearning #ai #math #explained
```

Short alternative:

```
Backpropagation with every number on screen. Forward: +0.01 × −0.87 + 1.78 × +0.05 + 0.47 = 0.55, fold, then 1.04 × 0.55 − 0.62 squashed to a guess of 0.49. Answer 1, error −0.51. Backward: the blame on each weight = the blame arriving × the number it multiplied. Nudge: new = old − 0.2 × blame, and the guess goes 0.49 → 0.57. Then 105 numbers at once in the 8 + 8 network: 0.34 → 0.72. Repeat a few thousand times. That is training.

Follow for more.

#machinelearning #neuralnetworks #backpropagation #ai
```

## YouTube description

```
One training step in slow motion, with every number on screen: first on a one-neuron network, then on the 8 + 8 network from the previous reel, both stopped mid-training on the two-arm spiral (400 points, ReLU, trained with Adam as before).

Forward: the example's two inputs flow along the edges (gold). The one-neuron net shows each weight on its edge and a ledger does the arithmetic as the signal arrives: sum = +0.01 × −0.87 + 1.78 × +0.05 + 0.47 = 0.55; the fold passes it (below zero would become 0); out = 1.04 × 0.55 − 0.62 = −0.05, squashed to a guess of 0.49. The 8 + 8 net shows the number each neuron passed on (0 for the ones that are off).

Error: guess minus answer (−0.51; for the 8 + 8 net 0.34 − 1 = −0.66). This is the only signal the network learns from.

Backward: the blame flows back along the edges (magenta). The blame on a weight is the blame arriving at it times the number that weight multiplied: the output bias gets −0.51, the output weight −0.51 × 0.55 = −0.28, the neuron passes −0.51 × 1.04 = −0.53 back because it was on, and then w₁ gets −0.53 × −0.87 = +0.47, w₂ gets −0.53 × +0.05 = −0.03, b gets −0.53. In the 8 + 8 net every neuron shows the blame that reached it; the 4 first-layer and 2 second-layer neurons that passed 0 get 0.

Nudge: new = old − step size × blame, for every number at once (step sizes 0.2 and 0.02, on screen). The pills tick to their new values; the output field, the fold lines, the guess and the count of points right are recomputed from the live weights on every frame. One neuron: 0.49 → 0.57, 248 → 252 of 400 right. 8 + 8: 0.34 → 0.72, 293 → 298, with none of the 105 numbers moving by more than 0.09.

Honesty notes: real training averages the gradient over all the points and uses Adam; here one example and a plain step are used so the blame of one mistake can be followed. Numbers are rounded to 2 decimals and the arithmetic uses the full values, so a product of the rounded numbers can be one unit off in the last digit. The example is the wrong vermilion point whose blame on the input weights is largest. A step on one example is not guaranteed to help the others; here it did.

Same spiral as reels 2, 31 and 32. Code in the repo.

#machinelearning #neuralnetworks #backpropagation #deeplearning
```

## Pinned comment

```
The whole of backpropagation is the one rule on screen: the blame on a weight = the blame arriving at it × the number that weight multiplied. "Blame" is the gradient: how much the error would change if that number were a little bigger. The chain rule hands it back layer by layer, starting from the error at the output, and the backward sweep costs about as much as one forward pass. That is why training big networks is possible at all.

Why off neurons get 0: a ReLU neuron that is off outputs exactly 0 whatever its inputs do, so for this example nothing downstream depends on its incoming weights. They get blame 0 now; they will get blame from other examples where the neuron is on.

Honesty check: real training does this on a batch of examples at once and uses Adam, which rescales each weight's step. Here it is one example and a plain step of stated size (0.2 for the tiny net, 0.02 for the 8 + 8), so that one mistake's blame can be seen. Numbers are rounded to 2 decimals and the arithmetic uses the full values (so −0.53 × −0.87 reads +0.47 on screen: the full numbers give 0.467). The field is recomputed from the live weights on every frame, not faded between two pictures.
```

## Audio

Gentle register, −16 LUFS, true peak under −1 dBTP, circular over 52 s. A soft chord bed; a rising pluck as
the signal crosses each layer and a high note at the guess; a low thud at the error; a darker, falling
pluck as the blame crosses each layer back; a quick four-note shimmer when the weights move and a chime
at the end of the nudge pitched by the new accuracy; bells at the headings.
