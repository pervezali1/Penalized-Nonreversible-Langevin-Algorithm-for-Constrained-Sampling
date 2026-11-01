# Posting kit — *The determinant is a volume.* (3D stage, maths card in Computer Modern, narrated, 59 s)

## Files

* Reel: `output/determinant_reel.mp4` — 58.617 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −16 LUFS (narration over the gentle track).
* Cover: `output/cover.png` — the general matrix: the slanted solid with its three coloured edges, and
  det = 2.23 with the million-point check on the card.

## The look

A new style for the series, made for linear algebra: a black stage with a soft navy pool of light on the
floor, a glossy cyan solid (flat-shaded faces with a light gradient, glowing edges, hidden edges dashed, a
soft shadow) seen by a slowly orbiting camera, and a maths card set in Computer Modern, the font of
textbook maths (rendered with matplotlib's mathtext). Everything is drawn from the live matrix every frame;
nothing is a pre-made model.

## The voice

Narrated by Kokoro, an open-source text-to-speech model (kokoro-onnx v1.0, Apache-2.0), with its
`am_michael` voice (American English, male) at 1.05× speed. The script is in `narration.py`, one line per
segment, read straight from the on-screen story with the numbers spelled out; the reel's timeline is sized
to the lines (each segment lasts at least as long as its line plus a short lead and tail), and the music
ducks 10 dB under the voice. The on-screen headlines stay, so the reel still works with the sound off.

| Segment | Line |
| --- | --- |
| 0 | The determinant is a volume. |
| 1 | A three by three matrix moves the three edges of a cube. Its columns are where those edges land. |
| 2 | Stretch it by two, one, and one and a half. The volume grows by exactly that: three. |
| 3 | Now slant it. Same base, same height, so the volume stays three. |
| 4 | Any matrix at all: its volume is the determinant. Here, two point two three. A million random points measure the same. |
| 5 | Tip the third edge into the plane of the other two, and the volume drains away. |
| 6 | Determinant zero. Crushed flat, with no way back. No inverse. |
| 7 | Keep going, and it comes out the other side: the mirror image. Same volume, but the determinant is minus two point two three. |
| 8 | So the determinant is how much a matrix scales volume. Zero means flattened. Negative means flipped. |

To rebuild the voice: download the two model files (see `narration.py`), set `KOKORO_DIR`, and run
`build.py` (it calls `narration.py`, which also rewrites `narration_timing.json`).

## What's on screen (all of it is real)

A 3 × 3 matrix M moves the unit cube. Its three columns c₁, c₂, c₃ are where the cube's three edges from
the corner land; they are drawn as coloured arrows (vermilion, green, gold) and the matrix on the card
uses the same colours for its columns. The solid is the image of the cube (a parallelepiped), the dashed
wireframe is the original unit cube, and the card shows the matrix and its determinant, recomputed every
frame as the matrix changes.

| Time | Matrix | det |
| --- | --- | --- |
| 0–10 s | the identity: the unit cube | 1 |
| 10–16 s | stretch: diag(2, 1, 1.5) | 3 (= 2 × 1 × 1.5) |
| 16–23 s | shear: [[2, 0.9, 0], [0, 1, 0], [0, 0, 1.5]] | 3 (same base, same height) |
| 23–31 s | a general matrix G = [[2, 0.9, 0.4], [0.3, 1, 0.5], [−0.2, 0.4, 1.5]] | 2.233 |
| 31–42 s | c₃ pushed straight into the plane of c₁ and c₂ | 0 (to 10⁻¹⁶) |
| 42–50 s | c₃ pushed straight through: its mirror image in that plane | −2.233 |
| 50–58 s | back to the identity | 1 |

**The check.** For G, a million random points are thrown into the box around the solid and the fraction
that lands inside the solid (tested by solving M·q = p and checking 0 ≤ q ≤ 1) times the box's volume
gives 2.229 ± 0.005, against det G = 2.233.

**The flat and the mirror.** c₃ is moved along the normal of the plane spanned by c₁ and c₂ (projected
into it for det = 0, reflected through it for the mirror), so the mirror image has exactly the same volume
with the opposite sign: −2.233. The mirrored solid is drawn in magenta; for that part the camera swings
round to the side it grows towards, so its depth stays visible.

## Honesty notes

* "Volume" means the signed volume of the image of the unit cube. A negative determinant means the
  solid is mirrored (its three edges have the opposite handedness); its actual volume is |det|.
* The camera moves (orbit, and a swing for the mirror segment); the solid's shape on screen depends on
  the view, the numbers don't.
* Numbers on the card are rounded to 2 decimals; the volume line on the general matrix is the Monte Carlo
  estimate to 2 decimals (2.23), not the determinant restated.

## Caption (paste as-is)

```
The determinant is a volume.

A 3 × 3 matrix is a machine that moves space. Watch what it does to a cube: the three edges from the corner land on the matrix's three columns, and the cube becomes a slanted box.

The determinant is how much that box's volume changed.

Stretch by 2, 1 and 1.5: the volume is 3, and so is the determinant.
Slant it: same base, same height, still 3.
A general matrix: det = 2.233. I checked it the hard way: a million random points thrown at the box measure a volume of 2.23.

Now tip the third edge into the plane of the other two. The volume drains away: det = 0. The box is crushed flat, and nothing can un-flatten it. That's why a matrix with determinant 0 has no inverse: whole directions of space get squashed to nothing, and there's no way to know where anything came from.

Keep pushing and it comes out the other side: the mirror image. Same volume, but det = −2.233. A negative determinant means space got flipped, like a reflection.

Size, flattening and flipping: that's all a determinant measures.

(Every number on the card is computed from the live matrix, every frame. The box is the real image of the cube; the volume check is a real Monte Carlo count.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#linearalgebra #determinant #matrix #math #maths #mathematics #visualmath #machinelearning
```

Short alternative:

```
A matrix moves a cube; its determinant is the new volume. Stretch 2 × 1 × 1.5: det 3. Slant it: still 3. A general matrix: 2.233, and a million random points agree. Tip one edge flat: det 0, no inverse. Push through: the mirror image, det −2.233.

Follow for more.

#linearalgebra #determinant #math #visualmath
```

## YouTube description

```
The determinant of a 3 × 3 matrix is the volume of what it does to a unit cube.

A matrix's three columns are where the cube's three edges from the corner land; the cube becomes a parallelepiped, and its signed volume is the determinant. In this video the matrix changes live and the determinant is recomputed every frame:

- the identity: det 1 (the unit cube)
- a stretch by 2, 1, 1.5: det 3
- a shear of that: det still 3 (same base, same height)
- a general matrix [[2, 0.9, 0.4], [0.3, 1, 0.5], [−0.2, 0.4, 1.5]]: det 2.233, checked by a million random points (Monte Carlo volume 2.229 ± 0.005)
- the third column pushed into the plane of the other two: det 0, the box is flat and the matrix has no inverse
- pushed through to its mirror image: det −2.233, the same volume with the orientation flipped

Why it matters: det = 0 means the matrix squashes some direction to nothing (it is singular, not invertible); |det| is how volumes scale, which is why it shows up in changes of variables, in the Jacobian, and in normalizing flows in machine learning; the sign says whether space is flipped.

#linearalgebra #determinant #matrix #math
```

## Pinned comment

```
Why "no inverse" when det = 0: the matrix flattened 3D space into a plane. Every point of a whole line got sent to the same place, so there's no way to undo it: an inverse would have to send one point back to many.

Why a negative determinant is a mirror: the three edges c₁, c₂, c₃ have a handedness (like your thumb, index and middle finger). A positive determinant keeps it; a negative one turns a right hand into a left hand. Here c₃ was reflected through the plane of c₁ and c₂, so the volume stayed 2.233 and only the sign flipped.

Where it shows up in ML: normalizing flows track how a transformation stretches probability using exactly this, log |det J|.
```

## Audio

Narration (Kokoro, am_michael, 1.05×) over the gentle register, −16 LUFS, true peak under −1 dBTP, circular over 58.6 s; the music ducks 10 dB under each line. A soft chord bed; under each
morph a soft voice whose pitch follows the live volume |det| on a log scale (rising as the solid grows,
sinking into a low hum as it flattens, back up through the mirror); a low thud at det = 0; a bright
three-note chime at the million-point check; bells at the headings.
