# Posting kit — *Three classes and softmax.* (dark theme, narrated, 58 s)

## Files

* Reel: `output/softmax_reel.mp4` — 58.3 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (narration over the gentle track).
* Cover: `output/cover.png` — the trained three-colour map with the ringed point's blame p − y on the card.
* Simulation archive: `softmax_sim.npz` (regenerate with `python smsim.py`; not committed; the scene regenerates it).
* Voice: Kokoro (open-source text-to-speech, kokoro-onnx v1.0), `am_michael` at 1.05×. Script in `scene.py` (`SCRIPT`).

## What's on screen (all of it is real)

A three-arm spiral, 450 points (3 × 150, 1.7 turns, noise 0.06), and a network 2 → 16 → 16 → 3 with ReLU, a softmax output
and cross-entropy loss, trained with Adam (step 0.005) for 1,500 steps on all the points (`smsim.py`). Six seeds; the
median final accuracy is shown (seed 1: 96.7%, 435 of 450; all six: 96.4–98.0%).

* **The map:** the three probabilities over the plane, mixed: each colour weighted by its probability, and the
  stronger the colour, the surer the network (50/50/50 would be the dark base); white lines are the decision boundaries
  (where the winning class changes). Trained live from snapshots every 15 steps.
* **The ringed point** (#245, a blue point on its arm at (−0.61, 0.06)): its three scores z, e^z and probabilities p
  live on the card during the softmax explanation; its probabilities as three bars; in the blame segments the bars show
  p − y with y = (0, 1, 0).
* Its story: at step 0 the three probabilities are nearly equal (0.28, 0.38, 0.34); early in training it leans to the
  wrong colours (vermilion 0.53 at step 75); from step 615 on, blue stays on top; at the end p = (0.01, 0.75, 0.24), so
  p − y = (0.01, −0.25, 0.24).

## Honesty notes

* The ringed point was chosen as the blue point (away from the centre) that the network gets right latest while still
  ending right, so its trajectory shows the learning; the network as a whole passes 90% of points right by step 420.
* "The blame is p − y" is the gradient of the cross-entropy loss with respect to the scores z, for softmax outputs:
  ∂L/∂z = p − y. It is the same formula as for two classes with a sigmoid (reel 34).

## Caption (paste as-is)

```
Two classes need one probability. Three need softmax.

Three spiral arms, three colours. For every point the network gives three scores, one per colour. Softmax turns them into probabilities: raise e to each score, divide by the total. Three numbers that always add up to 1.

The map shows those probabilities mixed: the stronger the colour, the surer the network. Watch it train.

Now the ringed point. Its blame at the output is guess minus answer, just like with two classes, but now it's three numbers: positive for the wrong colours (push them down), negative for the right one (push it up). Early on it leans the wrong way. From step 615, blue stays on top, and the blame keeps shrinking.

After 1,500 steps: 435 of 450 points right (96.7%).

Softmax plus cross-entropy: however many classes, the blame is still just p − y.

(Every probability, score and blame on screen is computed from the real network as it trains.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #softmax #neuralnetworks #deeplearning #classification #ai #math
```

## YouTube

Title:

```
Softmax: How a Network Picks Between Three Classes #shorts
```

Description:

```
Softmax and cross-entropy on a three-arm spiral, trained live.

A network with three outputs gives each point three scores z. Softmax turns them into probabilities p_i = e^{z_i} / (e^{z_1} + e^{z_2} + e^{z_3}), which are positive and add up to 1. The map shows the probabilities mixed by colour; the white lines are the decision boundaries.

For one ringed point the blame at the output, ∂L/∂z for cross-entropy, is p − y: three numbers, negative for the right class and positive for the others. Early in training it leans towards the wrong colours; from step 615 blue stays on top, and the blame shrinks to (0.01, −0.25, 0.24) by the end.

Network: 2 → 16 → 16 → 3, ReLU, softmax, Adam, 1,500 steps, median of 6 seeds: 435 of 450 points right (96.7%).

Voice: Kokoro, an open-source text-to-speech model (am_michael voice).

#machinelearning #softmax #deeplearning #shorts
```

## Pinned comment

```
Why the blame is just p − y: the loss is L = −log p_right, and p comes from softmax. Differentiate through both and almost everything cancels: ∂L/∂z_i = p_i − y_i. For two classes with a sigmoid it's the same formula (the backpropagation-with-maths reel shows the one-output version). That clean cancellation is why every classifier pairs softmax with cross-entropy.
```
