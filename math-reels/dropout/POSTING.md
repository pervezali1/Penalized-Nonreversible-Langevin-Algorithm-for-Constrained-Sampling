# Posting kit — *I flipped 83 of 400 labels. Which network believes them?*

## Files

* Reel: `output/dropout_reel.mp4` — 23.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −14.5 LUFS.
* Cover: `output/cover.png` — step 3,000: the plain field fractured into islands, the other two smooth; 76 / 32 / 28 lies believed.

## What's on screen (all of it is real)

The reel-02 spiral (400 points) with 83 labels flipped at random (20%, seed 1). Three copies of a
wide ReLU network (2 → 256 → 256 → 1), identical starting weights (seed 2), same Adam (lr 0.01),
same 3,000 full-batch steps, differing only in regularisation:

| Network | What differs | Lies believed (of 83) | Train acc. on the noisy labels | Accuracy on a clean held-out spiral |
| --- | --- | --- | --- | --- |
| PLAIN | nothing | **76** | 97% | **77%** (peaked at 83% around step 260, then fell) |
| DROPOUT | 50% of hidden neurons zeroed at random every step (inverted dropout, fresh mask per step; off at evaluation) | **32** | 83% | **83%** (peak 88%) |
| WEIGHT DECAY | L2 penalty 1e-3 on the weights | **28** | 82% | **82%** |

Flipped points are drawn as diamonds in the colour of the lie they were given; a diamond fills when
the network's prediction agrees with the lie. "Lies believed" counts exactly that at every step. All
three start at 46 of 83 because an untrained network agrees with half of everything by chance; the
plain one climbs to 76, the other two fall to the low 30s and 20s.

Why the plain field fractures: a 256-wide network has enough neurons for one to carve a small
island of the wrong colour around a single flipped point, and nothing stops it. Dropout removes any
given neuron half the time, so an island that depends on one neuron is wrong half the time during
training and never pays off; the network is pushed toward decisions many neurons agree on. Weight
decay penalises the large weights a sharp island needs. Both end up smoother, believe fewer lies,
and do better on clean data, while scoring *worse* on the training set they were given.

## Caption (paste as-is)

```
I took the spiral from reel 2 and flipped 83 of its 400 labels. Lies, planted in the training data. Then I trained the same network three ways.

Plain: nothing held back. Dropout: half the neurons switched off at random, every single step. Weight decay: weights kept small.

The diamonds are the lies. A filled diamond means the network believes it.

The plain network believed 76 of the 83. Its training accuracy hit 97%, which sounds great until you look at the field: an island of the wrong colour carved around every lie. On clean data it scored 77%, and it had peaked at 83% a tenth of the way through, before it started memorising.

Dropout believed 32. Weight decay believed 28. Their training scores were lower, 83% and 82%, and their clean scores were higher, 83% and 82%. They were worse at the exam they were given and better at the one that matters.

Why: memorising one point takes one neuron that owns that point. Dropout kills that neuron half the time, so owning a point never pays. Weight decay shrinks the big weights a sharp island needs. Both force the network to only believe things many neurons agree on.

A network that can memorise anything needs a reason not to. That's all regularisation is.

(numpy, 2 → 256 → 256 → 1 ReLU, same init, Adam 0.01, 3,000 steps; dropout p = 0.5 inverted with a fresh mask per step; L2 1e-3. Clean test = a fresh spiral of 400 points. Every count is computed from the real predictions at every step.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #neuralnetworks #deeplearning #ai #overfitting #dropout
```

Short alternative:

```
I flipped 83 of 400 training labels. A plain network memorised 76 of the lies and scored 77% on clean data. With dropout: 32 lies, 83%. With weight decay: 28 lies, 82%. Worse on the exam they were given, better on the one that matters. A network that can memorise anything needs a reason not to.

Follow for more.

#machinelearning #deeplearning #ai #overfitting #dropout
```

## Pinned comment

```
A network that can memorise anything needs a reason not to.

The maths: with 256 neurons per layer and 400 points the plain network has far more capacity than it needs, so it can fit every label, including the 83 wrong ones, by carving a tiny region around each. That's the 97% training accuracy. Dropout (Srivastava et al., 2014) multiplies each hidden activation by 0 with probability p = 0.5 during training (and by 1/(1−p) otherwise so the scale stays right), with a new mask every step. A feature that depends on one specific neuron is absent half the time, so the gradient pushes toward redundant features that many neurons share, and a one-point island is exactly the kind of feature that doesn't survive. Weight decay adds λ·w to every gradient, shrinking weights each step; a sharp island needs large weights, so it's taxed away.

Honesty check: (1) 20% label noise is a lot; the point is to make memorisation visible, not to claim typical numbers. (2) The clean-test gap here is 77% vs 83%, meaningful but not enormous; with less noise the gap shrinks, with a smaller network the plain one can't memorise in the first place (reel 10's 64-neuron net only reached 90% train on this data). (3) Early stopping would also have worked: plain peaked at 83% clean accuracy at step 261. (4) Numbers move a few points with the seed; the ordering (plain believes most lies, scores lowest on clean data) held on every seed I ran.

Next: the double descent curve, where making the network even bigger makes overfitting go away again. Want it?
```

## Audio

Keep the built-in track: three chords, one per network, each climbing with that network's clean-test
accuracy and drifting out of tune as it believes more lies, so the plain network's chord sours while
the other two settle. A short clash each time the plain network adopts another lie, a soft tick per
training step, bells at the reveals.

## Replies to expect

* "Why do all three start at 46 lies?" → an untrained network agrees with half of everything by chance; 83/2 ≈ 41, and this seed's random start happened to agree with 46. What matters is the direction: plain goes up to 76, the others go down.
* "Isn't 97% train accuracy good?" → on a dataset where 20% of the labels are wrong, 97% means you have memorised the wrong ones. The clean score is the one that counts.
* "Just stop early" → yes, that also works here: plain peaked at 83% clean accuracy at step 261. Early stopping is regularisation by clock instead of by architecture.
* "Dropout at 0.5 is too aggressive for a 2-layer net" → 0.3 gives 81–82% clean and believes slightly more lies; 0.7 gives 85% but trains slowly. 0.5 is the classic value and sits in the middle.
* "Which is better, dropout or weight decay?" → here, a tie within noise. Dropout peaked higher (88%), weight decay was steadier. Modern practice uses both, and often batch norm instead of dropout.
* "Does this prove deep nets overfit?" → it shows a wide net memorises noise when nothing stops it, and that the standard tools stop it. Reel 10 showed the extreme: 100% train accuracy on completely shuffled labels.
