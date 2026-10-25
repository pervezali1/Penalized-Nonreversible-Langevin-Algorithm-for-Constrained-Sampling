# Posting kit — *What if every weight starts at zero?*

## Files

* Reel: `output/init_reel.mp4` — 23.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −14 LUFS.
* Cover: `output/cover.png` — step 1,000: 100% / 50% / 62%, and the three neuron boards: 32 different tiles, 32 dark tiles, 32 identical tiles.

## What's on screen (all of it is real)

Three copies of the reel-18 network (2 → 32 → 32 → 1, ReLU) on the same spiral (seed 14), same
Adam (lr 0.01), same 1,000 full-batch steps. The only difference is the starting weights:

| Network | Start | Accuracy at step 1,000 | First-layer neurons that differ from each other (of 32, start → end) |
| --- | --- | --- | --- |
| RANDOM | reel 18's random init (seed 2) | **100%** (first reached at step 470) | 32 → 32 |
| ALL ZERO | every weight 0, every bias 0 | **50%** | 1 → 1 |
| ALL 0.1 | every weight 0.1, every bias 0 | **62%** | 1 → 1 |

Under each decision field is a board of 32 tiles, one per first-layer neuron, showing that neuron's
response over the plane (each tile scaled to its own maximum; dark = the neuron never fires).
RANDOM's tiles are 32 different half-planes turning as they train. ALL ZERO's tiles are dark for the
whole run. ALL 0.1's tiles are 32 identical pictures that move in perfect lockstep. All three networks have exactly 32 neurons in this layer (and 64 in
total); the on-screen count is how many of those 32 are different from each other, computed from the
actual weights at every step (unique rows of weights + bias). It is a result, not a setting.

**Why.** Two neurons with identical weights compute identical outputs, so backpropagation hands
them identical gradients, so after the update they are still identical. Nothing in training can
break a tie that was there at the start. A layer of 32 identical neurons is one neuron; the whole
64-neuron network collapses to 2 → 1 → 1 → 1, which can only draw a single straight line through
the plane: hence 62%. With zeros it is worse: every hidden activation is 0, so every weight's
gradient is 0 and only the output bias moves, which is why that panel never changes at all (50%).

Random initialisation is not about being random for its own sake. It breaks the symmetry so that
the 64 neurons can become 64 different feature detectors. (Modern schemes like He and Xavier
initialisation are "random with the right spread", which is the next refinement.)

## Caption (paste as-is)

```
Why don't we start a neural network with all weights at zero? I tried it.

Same 64 neurons, same spiral, same optimiser, same 1,000 steps. Three starts: random, every weight 0, every weight 0.1.

Random: 100% at step 470. All zeros: 50%, and the picture never changes, not one pixel. All 0.1: 62%, stuck on a single straight line.

The row of 32 tiles under each network is the real answer. Each tile is one first-layer neuron, what it responds to. The random network has 32 different tiles that keep turning as it learns. The zero network's tiles are dark the whole time. The 0.1 network's tiles are 32 identical copies, moving in perfect lockstep.

Here's the rule: two neurons that start identical get identical gradients, so they stay identical forever. Training cannot break a tie. Start 32 neurons the same and you have built a network with one neuron per layer. One neuron draws one straight line, and a spiral is not a straight line. That's the 62%. With zeros it's worse: nothing fires, so nothing even moves.

Randomness isn't noise. It's what makes 64 neurons 64 different neurons.

(numpy, 2 → 32 → 32 → 1 ReLU, Adam 0.01, 1,000 steps. The "distinct neurons" counter is computed from the actual weights at every step.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #neuralnetworks #deeplearning #ai #math #backpropagation
```

Short alternative:

```
Start a neural network with every weight at zero and it never learns. Every weight the same number: 62%, one straight line. Random: 100%. Identical neurons get identical gradients, so they stay identical forever. Randomness is what makes 64 neurons 64 different neurons.

Follow for more.

#machinelearning #neuralnetworks #deeplearning #ai
```

## Pinned comment

```
The maths, two lines. Forward: neuron j computes a_j = f(w_j·x + b_j). If w_j = w_k and b_j = b_k then a_j = a_k for every input. Backward: the gradient for w_j is (error signal to neuron j) × x, and the error signal to j is Σ_i W_ij·δ_i·f′, which is also identical for j and k when their outgoing weights match (they do: all 0 or all 0.1). Identical gradients, identical updates, identical forever: it's an invariant. That's "symmetry breaking", and it's why every framework's default initialiser is random (He/Kaiming for ReLU, Xavier/Glorot for tanh: random with a spread chosen so the signal neither explodes nor vanishes through the layers). Bonus: the zero network can't even move because ReLU(0) = 0 everywhere, so every weight gradient is 0; only the output bias trains. Next: what the right AMOUNT of randomness is (too small vs too large vs He init). Want it?
```

## Audio

Keep the built-in track: three voices. The random net's voice is a chord of eight detuned partials
that pull into tune as its accuracy rises; the all-zero net's voice is one flat tone that never
moves; the all-same net's voice is eight partials locked to exactly the same pitch, one sound
pretending to be many. A soft tick per training step, a ding when the random net hits 100%, bells
at the reveals.

## Replies to expect

* "The random one has 32 neurons and the others have 1, of course it wins" → all three have 32 neurons per layer; count the tiles. In the zero and 0.1 networks the 32 are perfect copies of each other, so they act as one. That is the result, not the setup.
* "Why does 0.1 get 62% and not 50%?" → it's a one-neuron-per-layer network, so it can learn one straight line, and the best straight line through the spiral gets 62% of the points.
* "Would biases random and weights zero work?" → partly: random biases break the tie in the first layer, but with zero weights ReLU(0 + b) is still a constant per neuron, so the first update is tiny and it trains very slowly. Random weights are the standard for a reason.
* "Why does zero not move at all?" → ReLU(0) = 0, so every hidden activation is 0, so every weight gradient (activation × error) is 0. Only the final bias has a non-zero gradient.
* "What about tanh instead of ReLU?" → same symmetry argument, same 1 distinct neuron; I ran it, 62% and 50% again.
* "Isn't 0.1 just a bad constant?" → any constant gives the same result; the problem is sameness, not the value.
* "So more randomness is better?" → no: too large and the signal explodes, too small and it vanishes. He/Xavier initialisation picks the spread. That's the next reel.
