# Posting kit — *What is each neuron looking at?* (dark theme, no formulas on screen, 45 s)

## Files

* Reel: `output/nnfields_reel.mp4` — 45.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (gentle track).
* Cover: `output/cover.png` — layer 2's eight fields under layer 1's eight, the curved pieces under the straight folds.
* Weights: read from `../nnbuild/nnbuild_train.npz` (the 8 + 8 run of reel 31; regenerate with `python ../nnbuild/nnsim.py`).

## What's on screen (all of it is real)

The trained 8 + 8 network of reel 31 (ReLU, Adam, 1,500 steps, 95.5% on the two-arm spiral), taken
apart neuron by neuron. Every tile is that neuron's actual activation over the input plane, computed from
the trained weights, bright where the neuron is on and dark where it is off (each tile scaled to its own
maximum). Nothing is trained in this reel; nothing is drawn by hand.

**Layer 1 (2.8–13 s).** Eight tiles, n_j = max(0, w_j·x + b_j). Each is a straight fold: off on one side
of its line w_j·x + b_j = 0 (dashed on the tile), a ramp on the other. A highlight walks the eight tiles
while the posting notes record that neuron's trained formula, for example n₁ = max(0, -4.35·x₁ +3.00·x₂ -1.40).

**Layer 2 (13.3–24.8 s).** Layer 1 shrinks to a strip; eight new tiles, m_k = max(0, v_k·n + d_k), each
a fold of a weighted sum of the eight fields above. The highlight walks them. The pieces curve: a fold
of an already folded field bends. No formulas appear on screen, by request; they live in this file.

**The output (24.8–38.8 s).** Layer 2's tiles move to a column with their output weights a_k (vermilion
positive, blue negative), and the big field shows σ(Σ a_k m_k + c) assembled one field at a time, in the
order that helps the accuracy most (greedy), with the decision boundary and the live accuracy:

| Fields added | 1 (m₈) | 2 (m₂) | 3 (m₃) | 4 (m₆) | 5 (m₁) | 6 (m₅) | 7 (m₄) | 8 (m₇) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Accuracy | 57% | 65% | 70% | 72% | 70% | 69% | 89% | 96% |

Output weights a₁…a₈: −1.92, +5.41, +7.87, −5.46, +1.67, +4.87, −2.32, −4.93; c = +0.12. The partial
sums are not meant to be good classifiers; the point is that the spiral exists only in the full sum.
Note the dip at fields 5 and 6: adding a correct piece can lower the accuracy until the piece that
balances it arrives. "Take any one away and it drops" is checked: removing a single layer-2 field
leaves 85, 80, 85, 75, 59, 89, 89 or 74% (for m₁ to m₈), never 96%.

## Caption (paste as-is)

```
What is each neuron looking at?

Last reel a network of 8 + 8 neurons learned the spiral. This is the same network, trained, taken apart one neuron at a time. Every tile is what one neuron actually computes over the plane: bright where it's on, dark where it's off.

Layer 1: eight neurons, eight straight folds. Each one is off on one side of a line and a ramp on the other, and that is all a single neuron can be. None of them looks remotely like a spiral.

Layer 2: eight more neurons, each fed by all eight above. Each is a fold of a weighted sum of those folds. Now the pieces curve, because a fold of an already folded field bends. Still no spiral: each neuron sees one piece.

The output: the eight layer-2 fields, weighted and added. Watch it assemble. One field: 57%. Four: 72%. Seven: 89%. All eight: 96%. Take any one away and it drops. The spiral doesn't live in any neuron. It lives in the sum.

That's the answer to the question people keep asking under these reels. No neuron "knows" the pattern. Each one draws one simple fold; the layer adds folds; the next layer folds the sum; and the shape appears only when every piece is weighed in.

(Every tile is computed from the trained weights of reel 31's network; the dashed lines are the real fold lines; the weights and the accuracies are exact. The fields are added in the order that helps the accuracy most. Remove any single field from the finished sum and the accuracy falls, to between 59% and 89%.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #neuralnetworks #deeplearning #ai #math #interpretability #explained
```

Short alternative:

```
The 8 + 8 network from last time, taken apart. Layer 1: eight straight folds. Layer 2: eight curved pieces, each a fold of the folds above. The output: the pieces weighted and added, 57% with one, 72% with four, 96% with all eight. No neuron knows the spiral. The sum does.

Follow for more.

#machinelearning #neuralnetworks #ai #deeplearning
```

## YouTube description

```
The trained 8 + 8 network from the previous reel (ReLU, 95.5% on the two-arm spiral), taken apart one neuron at a time. Every tile is one neuron's actual activation over the input plane, computed from the trained weights: bright where it is on, dark where it is off.

Layer 1: eight neurons, n = max(0, w·x + b). Each is a straight fold, off on one side of the line w·x + b = 0 and a ramp on the other. The dashed line on each tile is that fold. A highlight walks the eight and the card shows each neuron's exact trained formula.

Layer 2: eight neurons, m = max(0, v·n + d), each a fold of a weighted sum of the eight fields above. The card shows each one's three largest inputs. The pieces curve: a fold of an already folded field bends.

The output: σ(a₁m₁ + … + a₈m₈ + c). The layer-2 tiles line up with their output weights, and the big field assembles as the fields are added one at a time, in the order that helps the accuracy most: 57%, 65%, 70%, 72%, 70%, 69%, 89%, 96%. The dip at five and six is real: a correct piece can lower the accuracy until the piece that balances it is added. The spiral exists only in the full sum: remove any one field from it and the accuracy falls to between 59% and 89%.

This is the whole answer to "what does each neuron do": nothing recognisable on its own. A neuron is one simple fold, a layer adds folds, the next layer folds the sum, and the shape is in how the pieces are weighed together.

Every number on screen is computed, not drawn: the tiles, the fold lines, the weights and the accuracies all come from the saved weights of reel 31's run.

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #neuralnetworks #deeplearning #ai #math #interpretability
```

## Pinned comment

```
The maths behind the pictures: with ReLU, each layer-1 neuron is linear on each side of its line and zero on one side, so its field is a straight crease. A layer-2 neuron takes a weighted sum of those creased fields (a surface made of flat pieces whose creases are straight lines) and folds it where the sum is zero; that new crease follows the old ones and so bends. The output is one more weighted sum, squashed to a probability. Sixteen simple fields, one shape.

Honesty check: tiles are scaled to each neuron's own maximum (the raw activations range from 4 to 54), so brightness is relative, not comparable between tiles. The fields are added in the greedy order that helps the accuracy most; in the order of weight size the partial accuracies are 50, 50, 43, 49, 63, 70, 64, 59, 96, which makes the same point less watchably. Real networks with thousands of neurons are harder to read than this, but the mechanism is identical.

Next: backpropagation, one neuron at a time. How the blame for an error flows back to each weight.
```

## Audio

Keep the built-in track: the soft chord bed; a pluck for every tile that appears (an arpeggio per layer); a
soft tick as the highlight moves from neuron to neuron; while the output assembles, one pluck per field
added, pitched by the accuracy it reaches, and a chime at the full sum; a low tone at the close; bells at
the headings. −16 LUFS, 45 s loop.

## Replies to expect

* "Why don't any of the tiles look like a spiral?" → because no neuron computes one. Each is one fold; the spiral is in the weighted sum of all sixteen, which is the last picture.
* "Why does the accuracy drop at fields 5 and 6?" → adding a correct piece can tip the balance until its counterpart is added; the pieces only make sense together. In weight-size order it is even less monotone.
* "What are the bright and dark?" → the neuron's output, max(0, w·x + b), scaled to its own maximum; dark means exactly zero.
* "Does this work for big networks?" → the mechanism is the same; reading thousands of tiles is not. That's what interpretability research does with better tools.
* "Which neuron matters most?" → by output weight, m₃ (+7.87) and m₄ (−5.46); by what adding it first buys you, m₈ (57% on its own). Different questions, different answers.
```
