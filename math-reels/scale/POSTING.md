# Posting kit — *How random should the start be?*

## Files

* Reel: `output/scale_reel.mp4` — 23.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −15 LUFS.
* Cover: `output/cover.png` — step 1,000: 57% / 98% / 50%, with the three signal boards: a staircase down to 10⁻¹³, a flat row, a row of ∞.

## What's on screen (all of it is real)

Three copies of an 8-hidden-layer ReLU network (2 → 32 × 8 → 1) on the reel-02 spiral, trained with
**plain gradient descent** (lr 0.1, full batch) for 1,000 steps from random weights drawn with the
same seed, differing only in the spread of those random weights:

| Network | Weight std | Signal at layer 8 before training | Accuracy at step 1,000 |
| --- | --- | --- | --- |
| TOO SMALL | 0.01 | 7 × 10⁻¹³ (input was 0.46) | **57%** |
| HE INIT | 0.25 = √(2/32) | 0.4 (same as the input) | **98%** (90% first reached at step 369) |
| TOO BIG | 1.0 | 7 × 10³; 6 × 10²⁰ after one step; overflow at step 2 | **50%** |

Under each decision field is a bar per hidden layer: the standard deviation of that layer's
activations on the data, on a log scale from 10⁻¹⁴ to 10⁴, with a dashed line at the input's level.
Too small: each layer shrinks the signal about 25×, so by layer 8 it is 10⁻¹³. He: each layer passes
it on at the same size. Too big: each layer multiplies it by about 4, so by layer 8 it is 7,000; the
first gradient step is enormous, the second overflows, and from then on the first layer is dead
(all zeros) with infinities further in, which the bars show as empty bars and ∞ bars.

**Why He's number.** A ReLU neuron with n inputs of variance v, weights of variance σ², outputs
variance ≈ ½·n·σ²·v (the ½ because ReLU zeroes half of its inputs). Setting σ² = 2/n makes the
output variance equal the input variance, layer after layer. That is He (Kaiming) initialisation,
from "Delving Deep into Rectifiers" (He, Zhang, Ren, Sun, 2015), and it is the default in every
framework for ReLU networks. Xavier/Glorot (σ² = 1/n) is the same idea for tanh, which has no ½.

Caveats stated in the kit: Adam partially rescues bad scales (with Adam at 0.01 the too-small net
stays at 50% but the too-big one trains to 100%), which is why this reel uses plain gradient descent; batch normalisation and residual connections also make
deep networks far less sensitive to the starting scale. The lesson is about what the starting
weights do to the signal, which those tools were invented to manage.

## Caption (paste as-is)

```
Last reel: the weights must start random. This one: how random?

Same 8-layer network, same spiral, same plain gradient descent, same 1,000 steps. Three sizes of random start: tiny weights (std 0.01), He initialisation (std 0.25), big weights (std 1.0).

The bars under each network are the key. Each bar is one layer, showing how loud the signal is as it passes through. Tiny weights: every layer shrinks the signal 25×, and by layer 8 it is 10⁻¹³. The network is whispering to itself. Big weights: every layer multiplies it by 4, so by layer 8 it is 7,000 before a single step of training. One step later it is 10²⁰. On step 2 it overflows and the network is wreckage.

He initialisation sits exactly in between: each layer hands the signal on at the same size it received it. That network reaches 90% at step 369 and 98% by the end. The other two: 57% and 50%.

The formula is one line: weight std = √(2 ÷ number of inputs to the neuron). The 2 is there because ReLU throws away half of every signal. Kaiming He published it in 2015 and it is now the default start for every ReLU network you have ever used. Nobody thinks about it. Everything depends on it.

Honesty check: this uses plain gradient descent on purpose. With Adam the too-small net still sits at 50% after 1,000 steps, but the too-big one trains to 100%, because Adam shrinks the giant steps that overflow here. Batch norm and residual connections also make deep networks far less sensitive to the starting scale. The reel shows the raw problem those tools were built to manage.

Not too quiet. Not too loud.

(numpy, 2 → 32 × 8 → 1 ReLU, plain gradient descent lr 0.1, 1,000 full-batch steps, same seed for all three. The bars are the real activation std at every layer at every step.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #neuralnetworks #deeplearning #ai #math #backpropagation
```

Short alternative:

```
How big should a neural network's random starting weights be? Too small and the signal fades to 10⁻¹³ by layer 8. Too big and it hits 7,000, then overflows. He initialisation, std = √(2/inputs), keeps it the same size at every layer: 98% vs 57% vs 50%. Not too quiet, not too loud.

Follow for more.

#machinelearning #neuralnetworks #deeplearning #ai
```

## Pinned comment

```
Not too quiet, not too loud: that's the whole theory of initialisation.

The maths: a ReLU neuron with n inputs of variance v and weights of variance σ² outputs variance ≈ ½·n·σ²·v (the ½ because ReLU zeroes half its inputs). So each layer multiplies the SIZE of the signal by √(½·n·σ²). With n = 32: σ = 0.01 gives 0.04× per layer, about 10⁻¹¹ over 8 layers (the screen shows 10⁻¹³ because the first layer has only 2 inputs). σ = 1.0 gives 4× per layer, so 7,000 by layer 8. σ = √(2/n) = 0.25 gives exactly 1×, which is He/Kaiming initialisation (He et al., 2015). Xavier/Glorot uses 1/n for tanh, which doesn't throw half away.

Honesty check, three things:
1. Optimiser. This is plain gradient descent. Adam divides each step by the gradient's running scale, so it partly rescues bad starts: in my runs the too-small net still stayed at 50% for 1,000 steps, but the too-big net trained to 100%. I used plain descent so you see what the weights alone do.
2. Depth. With 2 layers all three starts learn eventually. The factor compounds per layer, which is why it takes 8 layers to make the gap 10⁻¹³ vs 1 vs 7,000.
3. Seeds. The exact percentages move with the seed. He always wins and the other two always lose, but 57% and 98% are this seed's numbers, not universal constants.

Next: why a network can reach 100% on shuffled labels, with dropout vs without. Want it?
```

## Audio

Keep the built-in track: three voices. The He net's chord pulls into tune as it learns; the too-small
net plays the same chord at a whisper whose level follows its layer-8 signal; the too-big net opens
with a two-step overload blast and then dies to a dull hum. A tick per training step, a ding when the
He net passes 90%, bells at the reveals.

## Replies to expect

* "Adam would fix this" → partly, yes: Adam divides each step by the gradient's running scale, so tiny gradients still move. In my tests with Adam the too-small net still sits at 50% after 1,000 steps, but the too-big one trains to 100%, because Adam shrinks the giant steps that overflow here. The reel uses plain gradient descent to show what the weights alone do.
* "Why 8 layers?" → the per-layer factor compounds: with 2 layers the three starts all learn eventually; with 8 the difference is 10⁻¹³ vs 1 vs 10³ and nothing can hide it.
* "Why does too-big show 0 and ∞ at the same time?" → the first update blew the weights up, the second overflowed to infinity; after that the first layer outputs exactly 0 for every input (dead ReLUs), and 0 × ∞ further in is undefined, so the network is broken in both directions at once.
* "Isn't 57% for too-small just slow?" → yes, it would get there eventually; gradients of 10⁻¹³ mean "eventually" is a very long time. Same lesson as sigmoid in reel 18.
* "What about Xavier?" → same idea without the 2: σ² = 1/n, right for tanh/sigmoid because they don't zero half the signal.
* "Does this matter with batch norm?" → much less, which is partly why batch norm was such a big deal in 2015. Residual connections too. The reel shows the raw problem those tools solve.
