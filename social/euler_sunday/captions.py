"""Captions, first comments (YOUR TURN answers), alt text and the posting schedule.

    python3 captions.py <commit-sha>   # writes posts.json with image URLs pinned to that commit

Every number below was checked in solutions.py.
"""
import json
import sys
from pathlib import Path

REPO_RAW = "https://raw.githubusercontent.com/pervezali1/Project-Euler"
TZ = "America/New_York"
IG_TIME, FB_TIME = "14:00:00", "18:00:00"   # Sundays; Metricool best-time data, clear of the reels

TAGS = "#projecteuler #math #python #coding #algorithms #mathinmotion"
SIGN = ("Euler Sunday: one Project Euler problem every week, solved two ways and drawn exactly. "
        "Every number on these slides is computed.")

POSTS = [
    {
        "problem": 1, "date": "2026-11-15",
        "caption": f"""Add up every multiple of 3 or 5 below 1000. It's Project Euler's very first problem, and there are two ways to solve it. 🧮

The loop checks all 999 numbers and keeps 466 of them. (The elif in it quietly stops 15, 30, 45, … from being counted twice.)

The shortcut uses Gauss's staircase, 1 + 2 + … + n = n(n + 1)/2, three times. The multiples of 3 add up to 166,833 and the multiples of 5 to 99,500. The multiples of 15 are in both piles, so take their 33,165 out once. That's 233,168.

Three calculations, whatever the limit. Below one million it's still three: 233,333,166,668.

✏️ YOUR TURN is on the last slide. The answer is in the first comment.

{SIGN}

{TAGS} #numbertheory""",
        "first_comment": """✏️ YOUR TURN: the multiples of 4 or 6 below 1000 add up to 165,834.

• Multiples of 4: 4 × (249 × 250 ÷ 2) = 124,500
• Multiples of 6: 6 × (166 × 167 ÷ 2) = 83,166
• Counted twice: the multiples of 12, the smallest number both 4 and 6 divide (not 24!): 12 × (83 × 84 ÷ 2) = 41,832

124,500 + 83,166 − 41,832 = 165,834. Take out the multiples of 24 instead and you get 187,002, which is too big. Did you fall for it? 👇""",
        "alt": [
            "A 10 by 10 grid of the numbers 1 to 100. Multiples of 3 are blue, multiples of 5 are orange, and multiples of both (15, 30, 45, 60, 75, 90) are green.",
            "Python code that loops from 1 to 999 and keeps numbers divisible by 3, or else by 5. 999 numbers checked, 466 hits.",
            "A staircase of blue squares inside an 8 by 9 rectangle, showing 1 + 2 + ... + 8 is half of 72. Then 3 + 6 + ... + 999 = 3 × 333 × 334 ÷ 2 = 166,833.",
            "A Venn diagram: multiples of 3 (333 numbers, sum 166,833), multiples of 5 (199 numbers, sum 99,500), overlap multiples of 15 (66 numbers, sum 33,165). 166,833 + 99,500 − 33,165.",
            "The answer, 233,168. Below one million it's 233,333,166,668. Your turn: add up every multiple of 4 or 6 below 1000.",
        ],
    },
    {
        "problem": 2, "date": "2026-11-22",
        "caption": f"""Add up the even Fibonacci numbers that don't go over four million. 🌀

The loop builds all 32 terms (1, 2, 3, 5, 8, …) and keeps the even ones.

But look at the pattern: odd + odd = even, and an even plus an odd is odd. So after every even term come exactly two odd ones, and every third Fibonacci number is even: 2, 8, 34, 144, …

Even better, the even ones make their own sequence: each is 4 times the last plus the one before. 34 = 4·8 + 2, 144 = 4·34 + 8. That's 11 terms instead of 32, and the total is 4,613,732.

✏️ YOUR TURN is on the last slide. The answer is in the first comment.

{SIGN}

{TAGS} #fibonacci""",
        "first_comment": """✏️ YOUR TURN: the 100th term is odd.

In 1, 2, 3, 5, 8, 13, … the even terms sit at positions 2, 5, 8, 11, …: two more than a multiple of 3. 100 = 3 × 33 + 1, so the 100th term is odd. The even ones nearest to it are the 98th and the 101st. No need to compute a 21-digit number 🙂""",
        "alt": [
            "Fibonacci squares of sides 1, 1, 2, 3, 5, 8, 13, 21 and 34 tiled into a rectangle with a spiral through them. The even squares, 2, 8 and 34, are blue.",
            "Python code that builds the Fibonacci list up to four million and adds the even terms. 32 terms, 11 of them even.",
            "The first 12 terms, 1 2 3 5 8 13 21 34 55 89 144 233, with every third one (2, 8, 34, 144) marked even. odd + even = odd, even + odd = odd, odd + odd = even.",
            "The 11 even Fibonacci numbers up to four million, from 2 to 3,524,578. Each is 4 times the last plus the one before: 34 = 4·8 + 2, 144 = 4·34 + 8.",
            "The answer, 4,613,732. Your turn: in 1, 2, 3, 5, 8, 13, ... is the 100th term even or odd?",
        ],
    },
    {
        "problem": 3, "date": "2026-11-29",
        "caption": f"""What is the largest prime factor of 600,851,475,143? 🔍

Testing every number up to it would take 600 billion tries. Two tricks cut that to about 1,500.

1. Divide each factor out as soon as you find it, so the number shrinks: 600,851,475,143 → 8,462,696,833 → 10,086,647 → 6857.
2. Only test divisors up to the square root of what's left. If n = a × b, the smaller one is at most √n.

After 1471 comes out, 6857 is left, and 1471² is already bigger than 6857. So 6857 has no smaller factor: it's prime, and it's the answer.

600,851,475,143 = 71 × 839 × 1471 × 6857.

✏️ YOUR TURN is on the last slide. The answer is in the first comment.

{SIGN}

{TAGS} #primenumbers""",
        "first_comment": """✏️ YOUR TURN: 2026 → 1013, and 2027 → 2027.

• 2026 = 2 × 1013, and 1013 is prime: nothing from 2 to 31 divides it (31² = 961, 32² = 1024). So its largest prime factor is 1013.
• 2027 is prime itself: nothing from 2 to 45 divides it (45² = 2025, just under 2027). So its largest prime factor is 2027.

Next year is a prime year 🎆""",
        "alt": [
            "A factor tree for the warm-up: 13,195 splits into 5 and 2,639, then 7 and 377, then 13 and 29. Primes are blue.",
            "Python trial division code: while d * d <= n, divide n by d if it divides, otherwise increase d.",
            "The factor tree of 600,851,475,143: 71 times 8,462,696,833, then 839 times 10,086,647, then 1471 times 6857.",
            "If n = a × b, the smaller factor is at most √n. 1471² = 2,163,841 is bigger than 6857, so the loop stops: 1,472 trial divisions instead of 600 billion.",
            "The answer, 6857. 600,851,475,143 = 71 × 839 × 1471 × 6857. Your turn: the largest prime factor of 2026, and of 2027.",
        ],
    },
    {
        "problem": 4, "date": "2026-12-06",
        "caption": f"""9009 = 91 × 99 reads the same forwards and backwards. What's the largest palindrome you can make by multiplying two 3-digit numbers? 🔁

Brute force multiplies all 405,450 pairs. Only 1,239 of them give a palindrome.

The trick: every 6-digit palindrome is a multiple of 11, because
abccba = 100001a + 10010b + 1100c = 11 × (9091a + 910b + 100c).
11 is prime, so one of the two numbers has to be a multiple of 11.

Count down from 999, let one number run only over multiples of 11, and stop as soon as no smaller pair can win. That's 522 pairs instead of 405,450.

The answer: 906,609 = 913 × 993, and 913 = 11 × 83.

✏️ YOUR TURN is on the last slide. The answer is in the first comment.

{SIGN}

{TAGS} #palindrome""",
        "first_comment": """✏️ YOUR TURN: yes, every 4-digit palindrome is a multiple of 11.

abba = 1000a + 100b + 10b + a = 1001a + 110b = 11 × (91a + 10b)

Example: 9009 (a = 9, b = 0) = 11 × 819. In fact every palindrome with an even number of digits is a multiple of 11. Can you see why? 👇""",
        "alt": [
            "A scatter plot of every pair of 3-digit numbers whose product is a palindrome, 1,239 dots, with the largest one circled near the top right.",
            "Python code that multiplies every pair of 3-digit numbers and keeps the largest palindrome. 405,450 pairs, 1,239 palindromes.",
            "abccba = 100001a + 10010b + 1100c = 11 × (9091a + 910b + 100c). Example: 888,888 = 924 × 962, and 924 = 11 × 84.",
            "A large square for 405,450 pairs next to a tiny square for 522 pairs, drawn with areas to scale.",
            "The answer, 906,609 = 913 × 993, and 913 = 11 × 83. Your turn: is every 4-digit palindrome a multiple of 11?",
        ],
    },
    {
        "problem": 5, "date": "2026-12-13",
        "caption": f"""2520 is the smallest number that every number from 1 to 10 divides. What's the smallest one for 1 to 20? ➗

Counting up 20, 40, 60, … works, but it takes 11,639,628 tries.

The better way: factor 2 to 20 into primes and keep only the biggest power of each prime that shows up. 16 = 2⁴, 9 = 3², and then 5, 7, 11, 13, 17 and 19 on their own.

2⁴ × 3² × 5 × 7 × 11 × 13 × 17 × 19 = 232,792,560

That's the least common multiple, lcm(1, 2, …, 20). In Python it's one line: math.lcm(*range(1, 21)). And no, 20! doesn't work. It's divisible too, but 10,450,944,000 times too big.

✏️ YOUR TURN is on the last slide. The answer is in the first comment.

{SIGN}

{TAGS} #primefactorization""",
        "first_comment": """✏️ YOUR TURN: 26,771,144,400.

From 21 to 25, only two numbers bring something new:
• 21 = 3 × 7, 22 = 2 × 11 and 24 = 2³ × 3 are already covered
• 23 is a new prime
• 25 = 5² beats the single 5

So multiply by 23 and by one more 5: 232,792,560 × 115 = 26,771,144,400 ✅""",
        "alt": [
            "The numbers 1 to 20 in a ring around an orange question mark: the smallest number all twenty divide.",
            "Python code that counts up in steps of 20 until every number from 1 to 20 divides n. 11,639,628 numbers tried.",
            "A table of the prime factors of 2 to 20 in columns 2, 3, 5, 7, 11, 13, 17, 19. The highest powers are orange: 2⁴ at 16, 3² at 9, and 5, 7, 11, 13, 17, 19.",
            "2⁴ × 3² × 5 × 7 × 11 × 13 × 17 × 19, and the Python one-liner lcm(*range(1, 21)). 20! is 10,450,944,000 times too big.",
            "The answer, 232,792,560. Your turn: the smallest number that 1 to 25 all divide.",
        ],
    },
    {
        "problem": 6, "date": "2026-12-20",
        "caption": f"""Square the sum of 1 to 100, then subtract the sum of the squares. What's left? 🟦

For 1 to 10 you can see it: a 55 × 55 square (that's (1 + … + 10)² = 3025) minus the squares along the diagonal (1² + … + 10² = 385) leaves 2640, every rectangle off the diagonal.

For 1 to 100, both sums have a formula:
1 + 2 + … + n = n(n + 1)/2 = 5050, and 5050² = 25,502,500
1² + 2² + … + n² = n(n + 1)(2n + 1)/6 = 338,350

25,502,500 − 338,350 = 25,164,150. Or in one go: (n − 1)n(n + 1)(3n + 2)/12.

✏️ YOUR TURN is on the last slide. The answer is in the first comment.

{SIGN}

{TAGS} #algebra""",
        "first_comment": """✏️ YOUR TURN: 41,230.

• 1 + … + 20 = 20 × 21 ÷ 2 = 210, and 210² = 44,100
• 1² + … + 20² = 20 × 21 × 41 ÷ 6 = 2,870
• 44,100 − 2,870 = 41,230

Check with the one-line formula: 19 × 20 × 21 × 62 ÷ 12 = 41,230 ✅""",
        "alt": [
            "A 55 by 55 square cut into rectangles i by j for i and j from 1 to 10. The diagonal squares 1² to 10² are orange, all other rectangles blue.",
            "Python code that adds k and k squared for k from 1 to 100, then prints total squared minus the sum of squares.",
            "Two formulas: n(n + 1) ÷ 2 = 5,050, squared 25,502,500; and n(n + 1)(2n + 1) ÷ 6 = 338,350.",
            "The difference is (n − 1) n (n + 1)(3n + 2) ÷ 12. For n = 10: 9 × 10 × 11 × 32 ÷ 12 = 2,640.",
            "The answer, 25,164,150 = 25,502,500 − 338,350. Your turn: do the same for 1 to 20.",
        ],
    },
    {
        "problem": 7, "date": "2026-12-27",
        "caption": f"""The first six primes are 2, 3, 5, 7, 11 and 13. What is the 10,001st? 🧱

The Sieve of Eratosthenes: write the numbers down, cross out the multiples of 2, 3, 5, 7, … and whatever survives is prime.

But how far do you write? The prime number theorem answers that: for n ≥ 6 the n-th prime is below n(ln n + ln ln n). For n = 10,001 that's about 114,319, so a sieve up to 114,320 is guaranteed to be long enough.

It finds 10,818 primes, and the 10,001st is 104,743.

✏️ YOUR TURN is on the last slide. The answer is in the first comment.

{SIGN}

{TAGS} #primenumbers #sieveoferatosthenes""",
        "first_comment": """✏️ YOUR TURN: the 100th prime is 541.

The same bound tells you how far to sieve: 100 × (ln 100 + ln ln 100) ≈ 613. Sieve up to 613 and you get 112 primes, the 100th of which is 541. (The grid on slide 1 stops at 113, the 30th prime.)""",
        "alt": [
            "A grid of the numbers 1 to 120 with the 30 primes highlighted in blue.",
            "The Sieve of Eratosthenes on 1 to 120: every composite number is crossed out with a line colored by its smallest factor, 2, 3, 5 or 7, and the primes stay blue.",
            "A line chart of the number of primes up to x for x up to 120,000, crossing 10,001 primes at 104,743, before the dashed bound at 114,319.",
            "Python sieve code up to 114,320 that prints primes[10000]. 10,818 primes found; the outer loop only needs i up to 338.",
            "The answer, 104,743. Your turn: what is the 100th prime?",
        ],
    },
]


def build(sha):
    out = []
    for p in POSTS:
        n = p["problem"]
        media = [f"{REPO_RAW}/{sha}/social/euler_sunday/images/p{n:02d}_s{k}.jpg" for k in range(1, 6)]
        assert len(p["alt"]) == 5 and len(p["caption"]) < 2200, n
        out.append({**p, "media": media,
                    "instagram": {"dateTime": f"{p['date']}T{IG_TIME}", "timezone": TZ},
                    "facebook": {"dateTime": f"{p['date']}T{FB_TIME}", "timezone": TZ}})
    return out


if __name__ == "__main__":
    sha = sys.argv[1] if len(sys.argv) > 1 else "main"
    data = build(sha)
    Path(__file__).with_name("posts.json").write_text(json.dumps(data, indent=1, ensure_ascii=False))
    for p in data:
        print(p["problem"], p["date"], len(p["caption"]), "chars")
