# Posting kit — *Eigenvectors: the directions a matrix only stretches.* (3D stage, narrated, 68 s)

## Files

* Reel: `output/eigen_reel.mp4` — 68.0 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (narration over the gentle track).
* Cover: `output/cover.png` — the egg with its three eigenvectors (vermilion, green, gold) and λ = 2, 1.2, 0.5 on the card.
* Voice: Kokoro (open-source text-to-speech, kokoro-onnx v1.0), `am_michael` at 1.05×. Script in `scene.py`
  (`SCRIPT`), voiced by `narration.py` through `../voicekit.py`; timings in `narration_timing.json`.

## What's on screen (all of it is real)

The 3D stage of reel 38 (`../stage3d.py`). A symmetric matrix A built from three eigenvalues, 2, 1.2 and 0.5, on tilted
axes (a rotation by 35°, −30°, 20°): A = [[1.58, 0.21, 0.39], [0.21, 1.19, 0.51], [0.39, 0.51, 0.94]], det A = 1.2.

| Segment | What happens | The numbers |
| --- | --- | --- |
| Ball | 80 unit arrows, nearly evenly spread (a Fibonacci sphere), inside a glass ball | |
| Apply | the matrix eases in: the ball becomes an egg and every arrow turns; one is followed with its original line dashed and its angle live | every one of the 80 turns, by 5.5° to 36.3°; the followed one by 29.5° |
| Eigenvectors | three directions drawn with their lines dashed through the centre: they stay on their lines and only stretch | |
| Eigenvalues | A v = λ v; each eigenvector's stretch measured live | λ₁ = 2, λ₂ = 1.2, λ₃ = 0.5 |
| Determinant | λ₁λ₂λ₃ = 2 × 1.2 × 0.5 = 1.2 = det A (reel 38) | the egg holds 1.2 balls of volume |
| Power steps | Aᵏ applied continuously (k from 0 to 12; real powers are fine because A is symmetric with positive eigenvalues), every arrow rescaled to length 1: they swing onto the strongest eigenvector | mean angle to v₁: 57.4° at k = 0, 0.53° at k = 12; within 1°: 0 → 72 of 80 |

Why the arrows gather: in the eigenvector basis, Aᵏ multiplies the v₁ part by 2ᵏ and the v₂ part by 1.2ᵏ, so the
v₂ share shrinks by 1.2 / 2 = 0.6 per step (the narration says the strongest direction "wins by a factor of 1.7",
which is 2 / 1.2). The 8 arrows still more than a degree off after 12 steps started almost at right angles to v₁.

## Honesty notes

* A is symmetric on purpose: its eigenvectors are perpendicular and real. A general matrix can have
  eigenvectors that aren't perpendicular, or complex eigenvalues (a rotation has no real eigenvector).
* "Every arrow turns" is about these 80; any arrow lying exactly along an eigenvector would not. None of the 80 does
  (the least turn is 5.5°).
* The power steps use continuous k to animate smoothly; the counts quoted are at whole k.

## Caption (paste as-is)

```
A matrix knocks almost every arrow off its line. Almost.

80 arrows, all of length 1, pointing every way. Apply a matrix: the ball becomes an egg, and every single arrow turns, by 5° to 36°.

Except three directions. They never turn. They only stretch. Those are the eigenvectors, and how much they stretch are the eigenvalues: here 2, 1.2 and 0.5. That's all A v = λ v says: v stays on its own line.

Multiply the three: 2 × 1.2 × 0.5 = 1.2. That's the determinant from the last reel: the egg holds 1.2 balls of volume.

Then the trick behind PageRank and a lot of machine learning: apply the matrix again and again, rescaling each arrow back to length 1. The strongest direction wins every time. After 12 steps, 72 of the 80 arrows lie within a degree of the top eigenvector.

(Every angle, stretch and count on screen is computed live from the real matrix.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#linearalgebra #eigenvectors #eigenvalues #matrix #math #visualmath #machinelearning #pca
```

## YouTube

Title:

```
Eigenvectors: The Directions a Matrix Only Stretches #shorts
```

Description:

```
Eigenvectors, visually: 80 unit arrows and a 3 × 3 symmetric matrix.

Apply the matrix and the ball of arrows becomes an egg. Every one of the 80 arrows is knocked off its line (by 5.5° to 36.3°), except three special directions that only stretch: the eigenvectors. Their stretch factors are the eigenvalues, here 2, 1.2 and 0.5. That is exactly what A v = λ v means.

Multiply the eigenvalues: 2 × 1.2 × 0.5 = 1.2 = det A, the volume scale factor from the determinant video.

Then power iteration: apply A again and again, rescaling each arrow to length 1. The component along the top eigenvector grows fastest (each step the rest shrinks by λ₂/λ₁ = 0.6), so the arrows swing onto it: mean angle 57.4° at the start, 0.53° after 12 steps, 72 of 80 within a degree. Power iteration is how PageRank was computed and how the top principal component can be found.

The matrix: A = [[1.58, 0.21, 0.39], [0.21, 1.19, 0.51], [0.39, 0.51, 0.94]] (symmetric, so its eigenvectors are perpendicular). Everything on screen is computed live.

Voice: Kokoro, an open-source text-to-speech model (am_michael voice).

#linearalgebra #eigenvectors #eigenvalues #math #shorts
```

## Pinned comment

```
Why repeated multiplication finds the top eigenvector: write any arrow as a mix of the three eigenvectors. Each time you apply A, the first part is multiplied by 2, the second by 1.2, the third by 0.5. After k steps the first part has grown by 2ᵏ and the others by much less, so after rescaling, almost all that's left is the first. That's power iteration.

Where it shows up: PageRank (the top eigenvector of the web's link matrix), PCA (the top eigenvector of the covariance matrix), and the stability of training dynamics (the largest eigenvalue of the Hessian, from the Hessian reel).
```
