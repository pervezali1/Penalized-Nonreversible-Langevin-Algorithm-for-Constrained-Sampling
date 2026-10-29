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

The story the cues tell: a neuron is max(0, w·x + b), a weighted sum then a fold at zero, and the fold is
the activation, the only source of bending; a layer adds folds, so its creases are straight lines; a
second layer folds the already-folded field, so its creases bend (a ReLU net stays piecewise linear at
any depth, so "curve" means many short straight pieces). The 1-neuron field is a straight ramp; 2 and 3
make corners and wedges; 8 makes a polygon of straight edges that cannot follow a spiral; 8 + 8 wraps
around both arms.

## Caption (paste as-is)

```
A neural network, one neuron at a time.

Same data every time: two spiral arms, 400 points. Same training every time. The only thing that changes is how many neurons there are, and how they're connected.

One neuron is a weighted sum followed by a fold at zero: output = max(0, w₁x₁ + w₂x₂ + b). The fold is the activation, and it is the only reason anything in a network ever bends. Without it, every layer is just more straight lines, and any number of them collapses into one. With it, one neuron draws one straight crease: flat on one side of a line, a ramp on the other. Trained on the spiral: 62%.

Two neurons, added: two creases. 62%. Three: 67%. A layer of eight: eight creases, eight straight lines in the plane. 81%. Everything is still straight lines, because adding folds together only ever gives you more straight creases.

Now the chain again: eight more neurons, fed by the first eight. Each new neuron folds a surface that is already creased, so its own crease follows the old ones and bends. Folds of folds. Now the boundary can zigzag along the spiral: 96%.

So the honest version is: the activation makes the fold, depth makes folds of folds. And a ReLU network never draws a true curve. Its boundary is always short straight pieces; depth just gives it many more of them, in the right places.

(Every network is real and trained from scratch with ReLU neurons and Adam; the fields are the networks' actual outputs over the plane and the dashed lines are the real fold lines of the first layer. The run shown for each size is the median of six random seeds.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #neuralnetworks #deeplearning #ai #math #relu #activation #explained
```

Short alternative:

```
One neuron: a sum and a fold. The fold is the activation; without it nothing ever bends. One layer: folds added, all straight creases: 62%, 62%, 67%, 81%. Another layer: folds of folds, the creases bend: 96%. Same data, same activation, one more layer.

Follow for more.

#machinelearning #neuralnetworks #ai #deeplearning
```

## YouTube description

```
A neural network built up one neuron at a time, on the same data every time: two spiral arms, 400 points.

A neuron is a weighted sum followed by an activation, here max(0, z). That fold is the only source of bending in the whole network: with no activation, every layer is linear and any stack of them collapses into a single straight line. With it, one neuron is one straight crease in the input plane: flat on one side of the line w·x + b = 0, a ramp on the other. Trained on the spiral it reaches 62%.

Two neurons added: two creases, 62%. Three: 67%. A layer of eight: eight creases, 81%. Adding folds together gives a surface made of flat pieces whose creases are straight lines, so a single ReLU layer can only ever draw straight edges, and a spiral curves everywhere.

Then the chain again: a second layer of eight, fed by the first eight. Each new neuron folds a surface that is already creased, so its crease follows the old ones and bends. Folds of folds. The boundary can now zigzag closely along both arms: 96%.

Two honest notes. First, it is the activation that bends, not the layer count; depth lets the folds compose. Second, a ReLU network never draws a true curve: its boundary is always made of straight pieces, and depth gives it many more of them in the right places. With a smooth activation such as tanh (reel 2), even one layer bends smoothly, and the build-up looks the same with gentler edges.

Each stage trains live on screen from its random start (ReLU, Adam, step size 0.01, 1,500 steps), with the diagram growing on the left and the decision boundary and the first layer's fold lines on the field. Of six random seeds per network the median run is shown; all six are in the posting notes.

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #neuralnetworks #deeplearning #ai #math #relu #activation
```

## Pinned comment

```
Where the bending comes from: the activation, max(0, z). A weighted sum is a flat sheet; the activation folds it along the line where the sum is zero. That is the only nonlinear thing in the network. Take it away and every layer is a linear map, and a stack of linear maps is one linear map: a single straight boundary, no matter how deep.

What depth adds: a second layer takes a weighted sum of already-folded sheets (still a surface of flat pieces with straight creases) and folds it again along the line where that sum is zero. On a creased surface that line is not straight, it bends at every old crease. So layer 2's creases are bent lines, and the boundary can follow a curve with many short straight pieces. A single ReLU layer can only make straight creases, however wide it is.

Honesty check: a ReLU net is piecewise linear at any depth, so "the edges curve" on screen means "many short straight pieces"; the smooth look comes from the probability shading. With tanh or a sigmoid even one layer gives genuinely curved boundaries, which is what reel 2 showed. Median of six seeds per size; the 8 + 8 runs ranged from 93% to 100%.

Next: the same network taken apart, one field per neuron. What each of the sixteen is actually looking at.
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
