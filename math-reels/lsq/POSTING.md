# Posting kit — *The line of best fit is a shadow (least squares in 3D).* (3D stage, narrated, 60 s)

## Files

* Reel: `output/lsq_reel.mp4` — 60.2 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (narration over the gentle track).
* Cover: `output/cover.png` — y, the plane of all lines, its shadow ŷ and the miss at a right angle; the best line in 2D below.
* Voice: Kokoro (open-source text-to-speech, kokoro-onnx v1.0), `am_michael` at 1.05×. Script in `scene.py` (`SCRIPT`).

## What's on screen (all of it is real)

Three points, (−1, 1), (0, 3), (1, 2), and the straight line y = a + b x that misses them least (in the sense of
least squares: the smallest sum of squared vertical misses).

| Segment | 3D | 2D panel and readouts |
| --- | --- | --- |
| y as an arrow | the three heights as one vector, y = (1, 3, 2); each axis is the height of one point | the points, labelled y₁, y₂, y₃ |
| The plane | (1, 1, 1) and x = (−1, 0, 1): every line a + b x gives the heights a (1, 1, 1) + b (−1, 0, 1), so all lines form the plane they span | |
| Tries | a line = a point on the plane; its miss = the distance to the tip of y | (a, b) = (2, 0), (1.2, 1), (2.6, −0.5), (1.6, 0.9): error² 2.00, 3.92, 4.58, 2.30 (live in between) |
| The shadow | the foot of the perpendicular from y: ŷ = (1.5, 2, 2.5), miss y − ŷ = (−0.5, 1, −0.5) at a right angle to the plane (camera swings to show it) | y = 2 + 0.5x, error² = 0.25 + 1 + 0.25 = 1.5, the smallest |
| Normal equations | Xᵀ(y − Xβ̂) = 0 ⇒ XᵀX β̂ = Xᵀy: [[3, 0], [0, 2]] (a, b) = (6, 1) ⇒ a = 2, b = 0.5 | |

The error² readout is the sum of squared misses, which is exactly the squared distance in 3D; the "miss" label on
the dashed line is its square root.

## Honesty notes

* The x-values −1, 0, 1 are centred, which makes (1, 1, 1) and x perpendicular and the normal equations diagonal.
  For uncentred x (say 1, 2, 3) the picture is the same but the plane is spanned by two arrows at an angle, and the
  two equations are coupled.
* With n points the arrow y lives in n dimensions and the lines still form a plane (2-dimensional); the shadow
  picture is identical, just not drawable beyond 3 points.

## Caption (paste as-is)

```
The line of best fit is just a shadow.

Three points: (−1, 1), (0, 3), (1, 2). We want the straight line a + b·x that misses them the least.

Stack the three heights into one arrow in 3D: y = (1, 3, 2).

Any line gives three heights too: a·(1, 1, 1) + b·(−1, 0, 1). All the possible lines together make a plane. Pick a line, and it's a point on that plane; how badly it misses is the distance from that point to the tip of y.

So the best line is the point on the plane right under y: its shadow. The leftover miss stands at a right angle to the plane. That right angle is two equations, Xᵀ(y − Xβ) = 0, the normal equations, and they give a = 2, b = 0.5. Sum of squared misses: 1.5, the smallest possible.

Least squares is a projection.

(Every arrow, miss and error on screen is computed live.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#linearalgebra #leastsquares #regression #statistics #datascience #math #visualmath #machinelearning
```

## YouTube

Title:

```
The Line of Best Fit Is a Shadow (Least Squares in 3D) #shorts
```

Description:

```
Least squares, seen as a projection.

Three points (−1, 1), (0, 3), (1, 2) and the line y = a + b x. The three heights form one vector y = (1, 3, 2) in 3D. Every line gives the heights a·(1, 1, 1) + b·(−1, 0, 1) = Xβ, so all lines together form the plane spanned by the columns of X. The sum of squared misses of a line is the squared distance from its point Xβ to y.

The closest point is the orthogonal projection of y onto the plane, ŷ = (1.5, 2, 2.5): the miss y − ŷ = (−0.5, 1, −0.5) is perpendicular to the plane. Perpendicular to both columns means Xᵀ(y − Xβ̂) = 0, the normal equations XᵀXβ̂ = Xᵀy: [[3, 0], [0, 2]] β̂ = (6, 1), so a = 2, b = 0.5, with sum of squared misses 1.5.

With n points the picture lives in n dimensions, but it's the same projection.

Voice: Kokoro, an open-source text-to-speech model (am_michael voice).

#leastsquares #linearalgebra #regression #shorts
```

## Pinned comment

```
Why squared misses? Because the sum of squares is the squared length of the miss vector, so "smallest sum of squares" means "closest point", and the closest point on a plane is always where you drop a perpendicular. That's the whole reason least squares has a clean formula: β̂ = (XᵀX)⁻¹Xᵀy.

The matrix that makes the shadow, X(XᵀX)⁻¹Xᵀ, is called the projection (or "hat") matrix: it puts the hat on y.
```
