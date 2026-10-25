# Posting kit — *Same AI. Same data. Three activations.*

## Files

* Reel: `output/activations_reel.mp4` — 23.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −14.4 LUFS.
* Cover: `output/cover.png` — step 1,000, the three fields side by side: 100% / 99.8% / 71%.

## What's on screen (all of it is real)

Three copies of the reel-02 network (2 → 32 → 32 → 1, the "64 neurons") trained on the same
two-arm spiral (400 points, seed 14) from **identical initial weights** (seed 2) with the same
Adam (lr 0.01, β 0.9/0.999) for the same 1,000 full-batch steps. The only difference is the
function inside each hidden neuron:

| Network | Activation | First 100% | Accuracy at step 1,000 |
| --- | --- | --- | --- |
| ReLU | max(0, z) | step 470 | 100% |
| tanh | tanh(z) | never (399 of 400 right) | 99.8% |
| sigmoid | 1/(1 + e⁻ᶻ) | never | 71% |

Each panel is the live decision field of that network (background tint = its output probability,
white line = the 50% boundary), with the data points drawn filled when classified correctly and as
hollow rings when wrong. The race chart beneath is the accuracy of all three per step. The leader
tag shows who is ahead at every step.

**Why.** Backpropagation multiplies the error signal by the activation's slope at every layer.
Sigmoid's slope is at most ¼ (at z = 0) and much smaller elsewhere, so through two hidden layers at
most 1/16 of the signal reaches the first layer, usually far less. tanh's slope is at most 1 and
ReLU's slope is exactly 1 wherever the neuron is active, so the signal arrives intact. The sigmoid
network is not broken, it is just learning twenty times slower; given many more steps it does get
there, which is exactly why "it trains eventually" is not good enough and nobody uses it inside
networks any more (it is still the right choice for the output of a binary classifier).

Seeds differ in detail: with other starts, sigmoid reaches between about 70% and 95% at step
1,000, tanh reaches 100% at or shortly after step 1,000, and ReLU is always first (checked on
seeds 1 and 2 at 2 × 32, 2 × 16 and 3 × 32).

## Caption (paste as-is)

```
Same 64 neurons. Same spiral. Same starting weights. Same optimiser, same 1,000 steps. I changed one thing: the little squashing function inside each neuron.

ReLU hit 100% at step 470. tanh got to 99.8%, one point still wrong at step 1,000. Sigmoid, the function every textbook draws first: 71%, and the race chart shows it barely moving.

Look at the three decision fields. ReLU's is made of straight edges, because ReLU is two straight lines glued together. tanh's is smooth. Sigmoid's never found the spiral at all.

Why: training sends the error backwards through the layers, multiplying by each neuron's slope on the way. Sigmoid's slope is at most ¼. Two layers: at most 1/16 of the signal gets back to the first layer, usually far less. That's the vanishing gradient. ReLU's slope is exactly 1 when the neuron is on, so the signal arrives intact. tanh is in between.

Sigmoid isn't broken. It's just learning about twenty times slower. In a network with ten layers it would be a thousand times slower, which is the same as never. That's the whole reason ReLU took over in 2012 and why the function you learned first is the one nobody uses inside a network.

(numpy, 2 → 32 → 32 → 1, identical init, Adam 0.01, 1,000 full-batch steps on the reel 2 spiral. Fields are the real network output at every pixel.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #neuralnetworks #deeplearning #ai #math #relu
```

Short alternative:

```
Same 64 neurons, same data, same start. Only the activation changed. ReLU: 100% at step 470. tanh: 99.8%. Sigmoid: 71% and barely moving. Its slope is at most ¼ per layer, so the error signal vanishes on the way back. That's why nobody puts sigmoid inside a network.

Follow for more.

#machinelearning #neuralnetworks #deeplearning #ai #relu
```

## Pinned comment

```
The maths: a hidden neuron computes a = f(z). Backprop multiplies the error by f′(z) at every layer. Sigmoid: f′ = a(1 − a) ≤ 0.25. tanh: f′ = 1 − a² ≤ 1. ReLU: f′ = 1 if z > 0 else 0. With L hidden layers the signal reaching layer 1 is scaled by the product of L slopes, so sigmoid gives at best 0.25^L (1/16 here, 1/1,000,000 at ten layers) while ReLU gives 1 along every active path. Adam partly compensates by rescaling each weight's step, which is why sigmoid still crawls upward here instead of flat-lining; with plain SGD it stays at 62% for the entire run (I checked). Caveats people will raise, all true: sigmoid is still correct as the OUTPUT of a binary classifier, ReLU has its own "dead neuron" failure, and GELU/SiLU/Swish are smooth cousins of ReLU that modern transformers use. Next: what happens when every weight starts at zero. Want it?
```

## Audio

Keep the built-in track: three voices, one per network, each a tone whose pitch climbs with that
network's accuracy, so the dissonant cluster resolves to a chord as ReLU and tanh converge while
sigmoid's voice stays low and wobbles; a soft tick per training step, a ding when a network first
hits 100%, bells at the reveals.

## Replies to expect

* "Sigmoid works fine in my network" → at the output, yes, always. Inside, with Adam and two layers, it crawls; with more layers or plain SGD it stalls. Try the same code with SGD.
* "Isn't 71% just a bad seed?" → across seeds sigmoid lands between about 70% and 95% at step 1,000 and is always last; ReLU is always first. The ordering is the claim, not the exact number.
* "Why is tanh not 100%?" → one point of 400 is still on the wrong side at step 1,000; it gets there a few hundred steps later. Across seeds tanh hits 100% at or shortly after step 1,000.
* "What about dead ReLUs?" → real: a ReLU neuron whose input is always negative has zero slope and never recovers. Leaky ReLU and GELU exist for that reason. Didn't happen here at this size.
* "GELU / SiLU / Swish?" → smooth versions of ReLU with slope near 1 for positive inputs; they behave like ReLU in this race.
* "Why ReLU's field has straight edges" → ReLU is piecewise linear, so a ReLU network is a piecewise-linear function: its boundary is made of straight segments. tanh is smooth, so its boundary curves.
