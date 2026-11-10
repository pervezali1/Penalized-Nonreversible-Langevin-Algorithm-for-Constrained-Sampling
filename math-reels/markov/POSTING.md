# Posting kit — *How long does snakes and ladders last? One matrix knows exactly.* (dark theme, narrated, 80 s)

## Files

* Reel: `output/markov_reel.mp4` — 1080×1920, 60 fps, H.264 CRF 16, AAC, −16.5 LUFS (narration over the gentle track;
  the −1 dBTP peak limit holds it half a unit under the usual −16).
* Cover: `output/cover.png` — the board with the expected visits to each square, the lengths of 100,000 simulated games
  against the matrix's prediction, and both averages.
* Runs: `board_results.json` (from `python boardsim.py`, about 2 seconds; seed 64).
* Voice: Kokoro (open-source text-to-speech, kokoro-onnx v1.0), `am_michael` at 1.05×. Script in `scene.py` (`SCRIPT`).

## What's on screen (all of it is real)

A 36-square board made up for this reel: ladders 3→16, 8→26, 20→29; snakes 13→2, 27→10, 32→6, 34→23. One player
starts off the board (square 0), rolls one die each turn, and the game ends on reaching or passing 36.

* Q is the 36 × 36 matrix of one-turn chances between square 0 and squares 1–35 (each 1/6; a ladder or snake moves its
  1/6 to the column of its other end; rolls that reach 36 leave the matrix).
* Row 0 of (I − Q)⁻¹: the expected visits to each square (square 29 most, 1.02). Its sum, the expected length of a
  game: 16.2469… turns, exactly 3943478451868282690841/242721583302259117800 (checked again by value iteration).
* 100,000 simulated games: mean 16.24, shortest 4, longest 142. Their length histogram matches the distribution from
  the powers of Q to within 0.002 per bar. Most likely length 9, median 13.
* With no snakes or ladders: 10.76 turns.
* Taking away one at a time and inverting again (the change in the average game):

| Removed | 32→6 | 34→23 | 27→10 | 13→2 | 20→29 | 8→26 | 3→16 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| It was worth | +4.65 | +3.71 | +2.55 | +0.58 | −2.45 | −2.15 | −0.84 |

## Honesty notes

* The board is invented for this reel, not a commercial one; on a real 100-square board the same method applies but
  the numbers differ. No exact roll is needed to finish, and a 6 gives no extra turn.
* "The snake from 32 to 6 adds 4.65 turns" compares this board with the same board minus that one snake. The seven
  effects don't simply add up: together they come to +6.05, but the full board is 5.49 turns longer than the empty one.

## Caption (paste as-is)

```
One of these 100,000 games of snakes and ladders lasted 142 turns. How long is a typical game? You don't have to play a single one. One matrix answers it exactly.

The board: 36 squares, 3 ladders, 4 snakes. Each turn you roll a die, so from every square there are 6 places to land, each with a chance of 1 in 6.

Write those chances in a matrix, Q: one row for where you are, one column for where you land. A ladder or a snake just moves its 1/6 to a different column.

Now invert one matrix: I minus Q. Its first row is how often you'll visit each square, on average. Add it up: 16.25 turns.

The check: 100,000 simulated games average 16.24, and their lengths follow the matrix's prediction. Without the snakes and ladders, a game would take 10.76.

It even tells you which snake hurts most: the one from 32 down to 6 adds 4.65 turns to the average game.

That's an absorbing Markov chain: one inverse matrix, for anything that wanders until it stops.

(The board is made up for this reel. Every number on screen is computed.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#math #markovchain #linearalgebra #probability #matrix #boardgames #datascience
```

## YouTube

Title:

```
How Long Does Snakes and Ladders Last? One Matrix Knows #shorts
```

Description:

```
How long is a game of snakes and ladders? An absorbing Markov chain answers it exactly.

A 36-square board made up for this reel (ladders 3→16, 8→26, 20→29; snakes 13→2, 27→10, 32→6, 34→23; one die, reaching or passing 36 ends the game). Q is the 36 × 36 matrix of one-turn chances between the unfinished positions; row 0 of (I − Q)⁻¹ gives the expected visits to each square, and its sum is the expected length: 16.2469 turns (exactly 3943478451868282690841/242721583302259117800). 100,000 simulated games average 16.24 (longest 142), and their length histogram matches the distribution from the powers of Q. Without snakes or ladders: 10.76. Removing one at a time: the snake 32→6 adds 4.65 turns, the ladder 20→29 saves 2.45.

Voice: Kokoro, an open-source text-to-speech model (am_michael voice).

#markovchain #linearalgebra #probability #shorts
```

## Pinned comment

```
Why does inverting I − Q count turns? (I − Q)⁻¹ = I + Q + Q² + Q³ + …, and row 0 of Qᵏ says where you could be after k turns, with what chance. Adding those up over every k gives the average number of turns you spend on each square before the game ends, and adding over the squares gives the average number of turns. The series adds up to a finite number because the game is certain to end eventually.
```
