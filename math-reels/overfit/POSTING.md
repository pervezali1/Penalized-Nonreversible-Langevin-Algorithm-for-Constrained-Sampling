# Posting kit — *Can a neural network learn pure noise?*

## Files

* Reel: `output/overfit_reel.mp4` — 20.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −14 LUFS.
* Cover: `output/cover.png` — the coin-flip test reveal.

## What's on screen (all of it is real)

Reel 2's 64-neuron network (32 + 32, tanh, Adam lr 0.02, 3,000 steps) trained twice on the
same 400 spiral points:

| Run | Labels | Train accuracy | Test accuracy (400 unseen points) |
| --- | --- | --- | --- |
| 1 | real | 100% | 92% |
| 2 | the same labels, randomly shuffled | 97.8% | 53.5% (chance) |

The background is the network's live decision field; dots are training points (hollow
ring = currently wrong); diamonds are the held-out test points (hollow = wrong). In run 2
the test labels are random too, because with shuffled labels there is no pattern to carry
over; the honest test score is a coin flip. The visible shuffle (dots flipping colour) is
the actual permutation used.

## Caption (paste as-is)

```
Reel 2's network, same 400 points, one change: I shuffled the labels. Every dot's colour is now random. There is nothing to learn.

It learned anyway. Training accuracy climbed to 97.8%; the decision boundary became a patchwork wrapped around every single dot. Then I showed it 400 points it had never seen: 53%. A coin flip.

That's overfitting: the network didn't find a pattern, it memorised the answer key. The only number that tells you which happened is accuracy on data the model has never seen. Training accuracy can't.

(With the real labels the same network gets 100% train / 92% test. Both runs: 64 neurons, Adam, lr 0.02, 3,000 steps, same seed. Every sound follows the training accuracy.)

Ever been fooled by a training score?

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #overfitting #neuralnetwork #deeplearning #datascience
```

Short alternative:

```
Shuffled the labels so there was nothing to learn. The network still hit 97.8% on training. On unseen points: 53%, a coin flip. That's overfitting, and only test data can catch it.

Follow for more.

#machinelearning #overfitting #neuralnetwork #deeplearning #datascience
```

## Pinned comment

```
This is a tiny version of the experiment in "Understanding deep learning requires rethinking generalization" (Zhang et al., 2017): big networks can fit completely random labels, so a good training score on its own proves nothing. 97.8% rather than 100% here because 64 neurons is just barely enough capacity for 400 random labels in 3,000 steps; give it more neurons or more steps and it goes all the way. Fix in practice: hold out data, early-stop on it, regularise. Next: watching early stopping catch it in real time. Want it?
```

## Audio

Keep the built-in track, **"the sound of memorising"**: the chord resolves with training
accuracy in both runs (it sounds like success both times, which is the point), static that
clears as the loss falls, a rattle under the shuffle, a bright bell for the real test and a
dissonant stinger when the coin-flip test lands.

## Replies to expect

* "Why not 100% on shuffled?" → capacity: 64 neurons, 3,000 steps; it's climbing when the run ends (pinned comment).
* "Test labels are random, of course it's 50%" → yes: the point is that no test set exists on which this network could score well, because it learned nothing transferable; with real spiral labels as the test it scores 49%.
* "This is why you use a validation set" → exactly; that's the next reel.
