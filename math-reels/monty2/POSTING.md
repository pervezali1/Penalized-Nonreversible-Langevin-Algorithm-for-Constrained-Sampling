# Posting kit — *3 doors. 1 car. 2 goats.* (v2, the explanatory cut)

## Files

* Reel: `output/monty2_reel.mp4` — 25.000 s, 1080×1920, 60 fps, H.264 CRF 16, AAC, −14 LUFS.
* Cover: `output/cover.png` — the three worlds with every door open: one STAY WINS, two SWITCH WINS.

## What's on screen (all of it is real)

Four beats, one argument.

1. **Hook (0–3.4 s).** Three doors. You pick door 1 (amber). The host opens door 3: a goat. "Switch?"
2. **The proof (3.6–10.8 s).** The car is behind door 1, 2 or 3, each equally likely: three worlds in
   three rows. In every world you pick door 1. The host opens a goat door (door 3 in worlds 1 and 2,
   door 2 in world 3, where door 3 has the car). Then every door opens. Staying wins only in world 1;
   switching wins in worlds 2 and 3. Tally: STAY 1/3, SWITCH 2/3. This is the whole proof, with no
   probability notation.
3. **The scoreboard (11.0–18.2 s).** 10,000 simulated games (numpy, seed 8): car and first pick uniform,
   host opens a goat the player did not pick, each game scored both ways. Live bars and percentages
   with 1/3 and 2/3 reference lines. Final: **stay 33.7%, switch 66.3%** (3,365 and 6,635 wins).
4. **The kicker (18.5–23.4 s).** 100 doors. You pick door 23. The host opens 98 goats one by one,
   accelerating, and skips door 61. "Your door: 1%. The one he skipped: 99%."

Closing line: "The host knows where the car is. His choice is information. Switch."

The rule that makes it work, and the one commenters will dispute: the host always opens a door you
did not pick and that has a goat behind it. If the host opened a random door and happened to show a
goat, switching would be 50/50. The reel states "the host always opens a goat" on screen.

## Caption (paste as-is)

```
Three doors. One car, two goats. You pick door 1. The host, who knows where the car is, opens door 3: goat. He offers you door 2. Switch or stay?

Most people say it doesn't matter, 50/50. It matters. Switching wins twice as often, and here is the proof in nine seconds with no formulas.

The car is behind door 1, 2 or 3, each equally likely. Three worlds. In every world you picked door 1 and the host opened a goat. World 1: the car is behind your door, staying wins. World 2: the host had to open door 3, switching to door 2 wins. World 3: the host had to open door 2, switching to door 3 wins. Stay wins in one world. Switch wins in two. That's it. 1/3 versus 2/3.

Don't trust the argument? I played 10,000 games both ways. Stay: 33.7%. Switch: 66.3%.

Still feels like 50/50? Make it 100 doors. You pick one. The host opens 98 goats and skips exactly one door. Your door had a 1% chance when you picked it. It still does. The one he carefully skipped carries the other 99%.

The host's choice is information. When he opens a door, he is telling you something about the doors he didn't open. Switch.

(Simulation: numpy, 10,000 games, car and pick uniform, host opens a goat you didn't pick. The host must know where the car is; a host opening at random and happening to show a goat gives 50/50.)

Follow for more: every reel is a real simulation and every number on screen is computed, not drawn.

#montyhall #probability #math #puzzle #mathfacts #statistics
```

Short alternative:

```
Monty Hall in nine seconds: the car is behind door 1, 2 or 3. You always pick 1, the host always opens a goat. Stay wins in one of the three worlds. Switch wins in two. 10,000 games: 33.7% vs 66.3%. The host's choice is information.

Follow for more.

#montyhall #probability #math #puzzle
```

## Pinned comment

```
The host knows. That one fact is the whole puzzle.

Why it's 2/3: your first pick is right 1/3 of the time and wrong 2/3 of the time. Nothing the host does changes that, because he can always open a goat door no matter where the car is. If your pick was wrong (2/3), the host has exactly one goat door he can open, and the door he leaves closed is the car. So switching wins every time your first pick was wrong: 2/3.

Why it feels like 50/50: after a door opens there are two closed doors, and our brains assign equal odds to two options. But the two doors have different histories. Yours was picked blind. His was left closed by someone who knows where the car is and was forbidden from opening it.

Honesty check: the 2/3 answer needs the host to (1) know where the car is, (2) always open a door you didn't pick, (3) always show a goat, (4) always offer the switch. Change any of those and the answer changes. A host who opens a random door and happens to reveal a goat gives exactly 50/50. In the original Let's Make a Deal, Monty did not always offer the switch, which is why the puzzle as usually stated is a rule set, not a description of the show.

Marilyn vos Savant published the 2/3 answer in 1990 and received about 10,000 letters telling her she was wrong, roughly 1,000 of them from people with PhDs. Paul Erdős didn't believe it until he saw a simulation. Beat 4 of this reel is that simulation.

Next: the Monty Hall variant where the host doesn't know, and why that flips the answer. Want it?
```

## Audio

Keep the built-in track: a door creak for every door that opens, a goat bleat for each goat, a
rising chord of dings when the three cars are revealed, a rain that follows games per frame during
the 10,000-game race with two tones that settle on 1/3 and 2/3, 98 accelerating clicks for the
100-door kicker, bells at the reveals.

## Replies to expect

* "It's 50/50, two doors left" → two doors, two different histories: yours was picked blind, his was left closed by someone who knows. The three-worlds section is the proof; count the worlds where switching wins.
* "What if the host doesn't know?" → then the times he accidentally reveals the car are thrown out, and among the remaining games switching is 50/50. The knowledge is what makes it 2/3.
* "Your simulation is rigged" → car uniform, pick uniform, host opens a goat you didn't pick, score both strategies on the same game. Twenty lines of numpy; I'll paste them on request.
* "Why 100 doors?" → it's the same puzzle with the odds stretched until intuition agrees with the maths: 1% vs 99%.
* "What did Monty actually do?" → he didn't always offer the switch; the puzzle fixes the rules so he must. With the rules fixed, 2/3.
* "Erdős got it wrong?" → yes, until Vázsonyi showed him a Monte Carlo simulation. Smart people get this wrong because the 50/50 instinct is strong, not because the maths is hard.
