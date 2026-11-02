# Posting kit — *PCA on real data: 150 flowers, squashed flat.* (3D stage, narrated, 68 s)

## Files

* Reel: `output/pca_reel.mp4` — 68.1 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (narration over the gentle track).
* Cover: `output/cover.png` — the cloud turned to face us and coloured by species: setosa apart, a 1.19 cm gap.
* Data: `iris.csv` — Fisher's iris data (Anderson 1935, Fisher 1936), as shipped with scikit-learn (same as R's `iris`).
* Voice: Kokoro (open-source text-to-speech, kokoro-onnx v1.0), `am_michael` at 1.05×. Script in `scene.py` (`SCRIPT`).

## What's on screen (all of it is real)

150 iris flowers, three measurements each: sepal length, petal length and petal width (cm). One point per flower,
drawn around the average flower (5.84, 3.76, 1.20 cm). PCA = the SVD of the centred data matrix (each measurement
minus its average): `numpy.linalg.svd`.

| Segment | What happens | The numbers |
| --- | --- | --- |
| The cloud | 150 points in 3D, the three measurement axes through the average flower | |
| Directions | the three principal directions (the columns of V), two spreads long each way | spreads (standard deviations): 2.05, 0.39, 0.18 cm |
| Shares | each direction's share of the total spread, σᵢ² / (σ₁² + σ₂² + σ₃²) | 95.8%, 3.4%, 0.8% |
| Flatten | every point slides along the third direction onto the best plane | kept 99.2%; points move 0.14 cm on average, 0.58 cm at most |
| Face us | the plane turns to face the camera: each flower is now 2 numbers | |
| One line | squashed onto the first direction | kept 95.8% |
| Species | coloured by species afterwards (PCA never used them) | setosa: PC1 ≤ −2.14; the others: PC1 ≥ −0.96; a 1.19 cm gap |

## Honesty notes

* Three of iris's four measurements are used (sepal width is left out), so the cloud can be drawn in 3D. With
  all four (unscaled), the first direction holds 92.5% of the spread.
* The data are in centimetres and not rescaled, so the spread is dominated by petal length; "spread" here is the
  variance, the standard measure PCA keeps.
* Five pairs of flowers have identical measurements, so 145 distinct points are visible.
* "Best plane": the plane through the average flower that minimises the sum of squared distances to the points.

## Caption (paste as-is)

```
150 flowers, 3 measurements each. Squash them flat, and lose almost nothing.

This is Fisher's iris data: sepal length, petal length and petal width for 150 real flowers. Every flower is one point in 3D.

PCA asks: which directions does this cloud spread along the most? The answer comes from the SVD of the data. One direction holds 95.8% of the spread. The other two: 3.4% and 0.8%.

Drop the smallest one. Every point slides onto the best plane, moving just 0.14 cm on average, and 99.2% of the spread is still there. Three measurements just became two. Even a single line keeps 95.8%.

The best part: PCA never saw which species each flower is. Colour them in afterwards, and setosa sits on its own, 1.19 cm from the rest.

That's PCA: the SVD of your data. Keep the directions with the most spread, drop the rest.

(Every direction, percentage and distance on screen is computed from the real data.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#pca #datascience #machinelearning #linearalgebra #svd #statistics #math #visualmath
```

## YouTube

Title:

```
PCA on Real Data: 150 Flowers Squashed Flat #shorts
```

Description:

```
Principal component analysis, on Fisher's iris data: 150 flowers, three measurements each (sepal length, petal length, petal width, in cm).

Centre the data and take its SVD: the right singular vectors are the principal directions, and the squared singular values say how much of the spread (variance) lies along each: 95.8%, 3.4% and 0.8%. Projecting onto the first two directions (the best-fitting plane) keeps 99.2% of the variance, moving each point 0.14 cm on average; onto the first direction alone, 95.8%.

PCA never uses the species labels. Coloured afterwards, setosa separates cleanly along the first direction: every setosa flower scores at most −2.14, every other flower at least −0.96, a 1.19 cm gap.

Everything on screen is computed from the data with numpy.

Voice: Kokoro, an open-source text-to-speech model (am_michael voice).

#pca #datascience #svd #shorts
```

## Pinned comment

```
How this connects to the SVD reel: PCA is the SVD of the centred data matrix X = U Σ Vᵀ. The columns of V are the principal directions, the singular values σᵢ measure the spread along them, and "keeping the top k" is the best rank-k approximation of the data (Eckart–Young).

Why setosa separates: its petals are far smaller than the other two species', and petal length is most of the first direction. PCA finds the direction of biggest spread; here, that happens to be the direction that tells setosa apart. It doesn't always work out that way: PCA looks for spread, not for classes.
```
