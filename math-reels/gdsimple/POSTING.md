# Posting kit — *Gradient descent, in the simplest way I can.*

## Files

* Reel: `output/gdsimple_reel.mp4` — 30.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (quieter, gentler track).
* Cover: `output/cover.png` — the cutaway bowl, the ball a few steps down, one vermilion arrow pointing downhill.

## What's on screen (all of it is real)

White background, one picture, one rule. The hill is the bowl f(w₁, w₂) = 0.5u² + 1.8v² on axes
rotated 32°, rendered as a shaded ceramic cutaway: the bowl is cut open along a plane so its
profile shows as a pale section face, the surface is lit from the upper left with a soft ground
shadow, and iso-height contour lines ring the valley. A real sphere (radial shading, specular dot,
contact shadow that follows the slope) starts near the rim at height 4.80. Every 0.62 s:

1. a vermilion arrow grows from the ball, pointing in the direction of steepest descent (−∇f), as
   long as the step the ball will take (learning rate 0.09 × the gradient), with a dashed
   ghost circle at the landing spot;
2. the ball slides along it; the walked arrow stays behind at lower alpha and a dot marks the
   landing, so the path builds as a chain of shrinking arrows.

Leader-line labels name the parts the first time they appear ("slope 3.66", "each dot: one step").
From step 3 a magnifier loupe follows the ball (×4 rising to ×12) so the arrow stays readable as
the real steps shrink to a few pixels; nothing inside it ever grows while the hill flattens. The
first steps are long and drop fast into the valley, then the steps shrink and the path curves along
the valley floor: the classic descent path, from real numbers. Readouts: STEP (a dot per step),
HEIGHT (ERROR) and SLOPE with progress bars. At the end the axes are labelled WEIGHT 1, WEIGHT 2
and ERROR for the closing line.

| Step | Height | Slope |
| --- | --- | --- |
| 0 | 4.80 | 3.66 |
| 1 | 3.70 | 3.03 |
| 5 | 1.60 | 1.81 |
| 10 | 0.62 | 1.11 |
| 24 | 0.044 | 0.30 |

## Caption (paste as-is)

```
Gradient descent, explained as simply as I can manage.

A hill. Height is how wrong you are. A ball dropped somewhere on it. The ball wants the lowest point, but it can't see the hill, only the ground under its feet.

So it follows one rule: feel the slope, take a step downhill. That's the arrow. Where the hill is steep the arrow is long and the step is big. Near the bottom the slope fades, the arrows shrink, the steps shrink, and the ball settles.

That is the whole algorithm. Gradient means the slope. Descent means walk down it.

In a neural network, the two directions on the ground are two of the network's weights, and the height is the error on the training data. A real network has millions of weights, so the hill has millions of directions, and the same rule works unchanged: feel the slope, step downhill, repeat.

Everything else you've heard of, learning rates, momentum, Adam, batches, is a refinement of how big the step is and which way it points. The reels before this one are those refinements. This one is the rule.

(The surface is a real function, 0.5u² + 1.8v² on rotated axes; every arrow is its actual negative gradient and every step is exactly learning rate 0.09 times that arrow. 24 steps: height 4.80 → 0.044. The magnifier shows the real arrow enlarged, never a redrawn one.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#machinelearning #gradientdescent #neuralnetworks #ai #math #deeplearning
```

Short alternative:

```
Gradient descent in one picture: a hill, a ball, one rule. Feel the slope under your feet, step downhill. Long arrows where it's steep, short arrows near the bottom, and it settles. In a neural network the ground is the weights and the height is the error. Millions of directions, same rule.

Follow for more.

#machinelearning #gradientdescent #ai #math
```

## Pinned comment

```
The one line of maths, for anyone who wants it: w ← w − η·∇E(w). Here w is the position on the ground (the weights), E is the height (the error), ∇E is the slope, the arrow, and η is the learning rate, how far along the arrow you step (0.09 in the reel). That's the entire algorithm. Why the path curves: the bowl is steep one way and shallow the other, so the first steps drop fast into the valley and later steps crawl along it; that's also why momentum exists (reel 4) and why the step size matters so much (reel 5). Why the arrow points straight downhill: of all directions you could step, the negative gradient is the one where the height falls fastest per unit of distance, which is what "steepest descent" means.

Honesty check: this hill has one bottom, so the rule always finds it. Real error surfaces have many valleys, flat plateaus and saddles; the rule still only ever sees the slope under its feet, which is the whole difficulty of training and the subject of reels 4, 5, 7 and 8.

Next, same white style: what the learning rate does, three balls on the same hill with three step sizes. Want it?
```

## Audio

Keep the built-in track, deliberately gentler than the dark reels: a soft sustained chord that swells
in with the surface, a quiet brush as each arrow grows, one plucked note each time the ball lands,
walking down a scale as the height falls, a low tone when it settles, and a bell at each heading.
Mixed at −16 LUFS.

## Replies to expect

* "What if there are two valleys?" → then the ball finds whichever one it rolls into, and that depends on where it started. Reel 4 is 10,000 balls on exactly that hill.
* "How does it know the slope without seeing the hill?" → calculus: the slope at a point can be computed from the function's formula without looking anywhere else. In a network that computation is backpropagation.
* "Why not just jump to the bottom?" → it doesn't know where the bottom is, only which way is down from here. That's the whole constraint.
* "What decides the step size?" → the learning rate, a number you choose. Too small and it crawls, too big and it overshoots the valley. Reel 5.
* "Is this a real loss surface?" → it's a real function with the two features that matter, a steep direction and a shallow one. Real loss surfaces are the same idea in millions of dimensions, and far bumpier.
