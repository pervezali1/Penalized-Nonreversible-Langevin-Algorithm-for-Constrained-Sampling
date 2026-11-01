# Posting kit — *Every matrix is a turn, a stretch and a turn (the SVD).* (3D stage, narrated, 73 s)

## Files

* Reel: `output/svd_reel.mp4` — 73.1 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (narration over the gentle track).
* Cover: `output/cover.png` — the stretch step: the egg along the axes, σ₁ = 2.4, σ₂ = 1.25, σ₃ = 0.5, and Σ on the card.
* Voice: Kokoro (open-source text-to-speech, kokoro-onnx v1.0), `am_michael` at 1.05×. Script in `scene.py` (`SCRIPT`).

## What's on screen (all of it is real)

A matrix M built from its decomposition, M = U Σ Vᵀ, with U and V rotations (Euler angles 40°, −25°, 30° and −55°, 35°,
−20°) and Σ = diag(2.4, 1.25, 0.5): M = [[0.17, −1.80, 0.25], [0.99, −0.52, −0.84], [1.03, −1.09, 0.14]] (a
random-looking matrix; `numpy.linalg.svd` recovers exactly these singular values). det M = 1.5.

| Segment | What happens |
| --- | --- |
| All at once | M applied to a unit ball with three painted arrows (v₁, v₂, v₃, the right singular vectors): it becomes a tilted egg |
| Turn (Vᵀ) | back to the ball, then the rotation Vᵀ by slerp: the ball looks the same, the arrows line up with x, y, z; the first egg stays as a dashed ghost |
| Stretch (Σ) | the stretch along the axes, live: the arrows become σ₁ = 2.4, σ₂ = 1.25, σ₃ = 0.5 |
| Turn (U) | the rotation U: the egg lands exactly on the ghost |
| M = U Σ Vᵀ | the three matrices on the card |
| det | σ₁σ₂σ₃ = 2.4 × 1.25 × 0.5 = 1.5 = det M (reel 38) |
| Rank 2 | σ₃ squashed (to 0.02, so the solid stays drawable): the egg goes flat; the camera swings to see its edge; kept σ₁² + σ₂² over σ₁² + σ₂² + σ₃² = 96.7% |
| Rank 1 | σ₂ squashed too: a needle along u₁; kept σ₁² / (σ₁² + σ₂² + σ₃²) = 76.1% |

## Honesty notes

* "Percent of the matrix" means the share of the sum of squared singular values (the squared Frobenius norm), which
  is the standard measure of how much of a matrix a low-rank approximation keeps; the formula is on the card.
* This M has a positive determinant, so U and V can both be taken as rotations. For a matrix with a negative
  determinant one of them would include a reflection.
* The flat and needle shapes are drawn with the dropped stretch at 0.02 instead of exactly 0, so the solid still has
  faces to shade; the percentages use exactly 0.

## Caption (paste as-is)

```
Every matrix does just three things: turn, stretch, turn.

Watch a matrix act on a ball, all at once: it becomes a tilted egg. Now in slow motion:

1️⃣ A turn. The ball looks the same, but its three painted arrows swing onto the x, y and z axes.
2️⃣ A stretch along those axes: by 2.4, 1.25 and 0.5.
3️⃣ One more turn, and it lands exactly on the same egg.

M = U Σ Vᵀ. Every matrix in the world can be written like that. It's the singular value decomposition, and the stretches are the singular values.

Multiply them: 2.4 × 1.25 × 0.5 = 1.5, the determinant again.

Now the useful part. Drop the smallest stretch: the egg goes flat, and still keeps 97% of the matrix. Keep only the biggest: a single line, still 76%. That's the idea behind PCA and image compression: keep the big stretches, throw away the rest.

(The ball, the egg, every rotation and every stretch are computed from the real decomposition.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#linearalgebra #svd #matrix #pca #math #visualmath #machinelearning #datascience
```

## YouTube

Title:

```
Every Matrix Is a Turn, a Stretch and a Turn (SVD) #shorts
```

Description:

```
The singular value decomposition, visually: M = U Σ Vᵀ.

A 3 × 3 matrix acts on a ball and turns it into a tilted egg. In slow motion the same thing happens in three steps: a rotation Vᵀ (the ball looks the same, but its painted arrows line up with the axes), a stretch Σ along the axes (the singular values 2.4, 1.25 and 0.5), and a rotation U. The result lands exactly on the original egg. Every real matrix can be written this way.

The product of the singular values is |det M|: 2.4 × 1.25 × 0.5 = 1.5.

Low rank: setting the smallest singular value to zero flattens the egg into a disc and keeps 96.7% of the sum of squared singular values; keeping only the largest gives a line and keeps 76.1%. Truncating the SVD like this is the best low-rank approximation of a matrix (Eckart–Young), which is what PCA and SVD image compression use.

The matrix: M = [[0.17, −1.80, 0.25], [0.99, −0.52, −0.84], [1.03, −1.09, 0.14]]. Everything on screen is computed.

Voice: Kokoro, an open-source text-to-speech model (am_michael voice).

#linearalgebra #svd #pca #math #shorts
```

## Pinned comment

```
How the SVD relates to the eigenvector reel: the singular values of M are the square roots of the eigenvalues of MᵀM, and V holds its eigenvectors. For a symmetric matrix with positive eigenvalues the SVD and the eigendecomposition are the same thing.

Why "keep the biggest stretches" works: the Eckart–Young theorem says that cutting the SVD after the k largest singular values gives the closest rank-k matrix there is. PCA does exactly this to a data matrix: the directions with the biggest singular values are the principal components.
```
