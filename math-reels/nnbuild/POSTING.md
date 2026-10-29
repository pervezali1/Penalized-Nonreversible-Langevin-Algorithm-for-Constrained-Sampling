# Posting kit — *A neural network, one neuron at a time.* (dark theme, 48 s)

## Files

* Reel: `output/nnbuild_reel.mp4` — 48.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (gentle track).
* Cover: `output/cover.png` — the two-layer network trained: the curved boundary wrapped around both spiral arms, 96%.
* Training archive: `nnbuild_train.npz` (regenerate with `python nnsim.py nnbuild_train.npz`; not committed).

## What's on screen (all of it is real)

The same data every time: the two-arm spiral of reel 2 (400 points, `neurons/train.py`). Five networks
are trained from scratch on it (`nnsim.py`): 1, 2, 3 and 8 ReLU neurons in one layer, then two layers
of 8. Adam, step size 0.01, 1,500 steps, binary cross-entropy. For each network the reel shows:

* the dark night-blue background of reels 1–22, with the light ink to match;
* the diagram on the left, growing as neurons appear; input-to-neuron edges coloured by the sign of the
  live weight (vermilion positive, blue negative) and thickened by its size;
* the output on the right: the probability of the vermilion class over the input plane, trained live
  from the random start (101 snapshots, every 15 steps), with the decision boundary (output = 0.5) as an
  ink line and each first-layer neuron's fold line (w·x + b = 0) dashed over it;
* readouts: neurons, accuracy (the live value at the current snapshot), training step.

No formulas on screen, by request; the maths lives in this file and in the pinned comment.

Of six random seeds per network, the run with the median final accuracy is shown (the typical run).

| Network | Final accuracy | All six seeds |
| --- | --- | --- |
| 1 neuron | **62%** | 62, 60, 62, 62, 62, 60 |
| 2 neurons | **62%** (62.5) | 65, 63, 62, 63, 62, 62 |
| 3 neurons | **67%** | 63, 62, 67, 65, 71, 68 |
| 8 neurons | **81%** | 83, 78, 76, 78, 81, 83 |
| 8 + 8 (two layers) | **96%** (95.5) | 98, 96, 94, 100, 95, 93 |

The story the cues tell: a neuron is max(0, w·x + b), a weighted sum then a fold at zero, so one neuron
is one straight fold in the input plane; a layer adds folds, so its edges are straight lines; a second
layer folds the already-folded field, so its edges curve. The 1-neuron field is a straight ramp; 2 and 3
make corners and wedges; 8 makes a polygon of straight edges that cannot follow a spiral; 8 + 8 wraps
around both arms.

## Caption (paste as-is)

```
A neural network, one neuron at a time.

Same data every time: two spiral arms, 400 points. Same training every time. The only thing that changes is how many neurons there are, and how they're connected.

One neuron is a weighted sum followed by a fold at zero: n = max(0, w₁x₁ + w₂x₂ + b). Flat on one side of a line, a ramp on the other. It can draw one straight edge, and nothing else. Trained on the spiral: 62%.

Two neurons, added. Two folds. The edge can bend once. 62%. Three: corners, wedges, a band. 67%. Every edge is still a straight line, because every neuron is still a straight fold.

A layer of eight. Eight folds, added: eight straight edges in the plane, and the field is flat between them. 81%. Better, but a spiral curves everywhere and these edges can't.

Now the chain again: eight more neurons, fed by the first eight. A fold of an already folded field bends. Folds of folds. The edges curve, they wrap around both arms, and the accuracy goes to 96%.

That's the whole trick. Neuron: one fold. Layer: many folds, added. Depth: folds of folds. Everything else is scale.

(Every network is real and trained from scratch with ReLU neurons and Adam; the fields are the networks' actual outputs over the plane and the dashed lines are the real fold lines of the first layer. The run shown for each size is the median of six random seeds.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #neuralnetworks #deeplearning #ai #math #relu #explained
```

Short alternative:

```
One neuron: one straight fold. 62% on the spiral. Two: 62%. Three: 67%. A layer of eight: 81%, eight straight edges. Then the chain again, eight more fed by the first eight: folds of folds, the edges curve, 96%. That's the whole trick.

Follow for more.

#machinelearning #neuralnetworks #ai #deeplearning
```

## YouTube description

```
A neural network built up one neuron at a time, on the same data every time: two spiral arms, 400 points.

A neuron is a weighted sum followed by a fold at zero, n = max(0, w₁x₁ + w₂x₂ + b), and the output is σ(a·n + c), a probability. Over the input plane one neuron is one straight fold: flat on one side of a line, a ramp on the other. Trained on the spiral it reaches 62%.

Two neurons added together: two folds, the edge bends once, 62%. Three: 67%. A layer of eight: eight folds, eight straight edges in the plane, 81%. Every edge is a straight line because every neuron is a straight fold, and a spiral curves everywhere.

Then the chain again: a second layer of eight, fed by the first eight. A fold of an already folded field bends, so the edges can curve. The boundary wraps around both arms: 96%.

Each stage trains live on screen from its random start (ReLU neurons, Adam, step size 0.01, 1,500 steps), with the diagram growing on the left and the decision boundary and the first layer's fold lines on the field. Of six random seeds per network the median run is shown; all six are in the posting notes.

Neuron: one fold. Layer: many folds, added. Depth: folds of folds. That is the whole trick; everything else is scale.

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #neuralnetworks #deeplearning #ai #math #relu
```

## Pinned comment

```
Why "folds": with ReLU, max(0, z), every neuron is linear on each side of the line w·x + b = 0 and bends there. Add several and you get a continuous surface made of flat pieces whose creases are those lines: a single layer can only ever make straight creases. Feed the result into another layer and each new neuron folds the already-creased surface, so the new creases follow the old ones and bend. That is why depth buys curves that width alone cannot, on a budget of 16 neurons here.

Honesty check: 1,500 steps and the median of six seeds per size; the best seed for 8 + 8 reached 100%, the worst 93%. With tanh instead of ReLU the edges are smooth from the start (reel 18) but the story is the same. A wide enough single layer can also approximate the spiral, with far more neurons than 16; depth is the cheaper way to curve.

Next, same white style: what each hidden neuron is actually looking at, one field per neuron.
```

## Audio

Keep the built-in track: the soft chord bed; at each stage one pluck per neuron that appears (a rising
arpeggio); during training a quiet ratchet tick every few snapshots; a chime at the end of each stage,
pitched by its accuracy; a low tone at the close; bells at the headings. −16 LUFS, 48 s loop.

## Replies to expect

* "Why is 2 no better than 1?" → two straight folds still can't follow a spiral; the best two-fold field just picks a slightly better wedge. The jump comes with enough folds (8) and then with depth.
* "Would 64 neurons in one layer do it?" → much better than 8 (reel 2 got 100% with 64 tanh neurons), but with straight creases it needs many of them. Two layers of 8 do it with 16.
* "Why ReLU here and tanh in reel 2?" → ReLU makes the fold story literal: the creases are straight lines you can draw. The same build-up works with tanh, with smooth bends instead.
* "Are the dashed lines real?" → yes: w·x + b = 0 for each first-layer neuron, from the live weights at every snapshot.
* "Is 96% the best?" → the best of six seeds reached 100%, the worst 93%; the median run is shown so the number is typical, not lucky.
```
