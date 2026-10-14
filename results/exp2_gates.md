G1: learnability (n = 64, shift 0) vs the mean predictor

| task | mean predictor | S | S vs mean predictor |
|---|---|---|---|
| A | 0.4150 | 0.3341 | -19.1% [-23.0, -15.3]* better on 81% of bodies |
| B | 0.8987 | 0.4183 | -53.4% [-56.8, -50.0]* better on 100% of bodies |

G2: zero prompt vs aligned prompt (n = 64, 2 fine-tune seeds); positive = zero prompt is worse

| task | variant | aligned | zero prompt | zero vs aligned |
|---|---|---|---|---|
| A | S | 0.3341 | 0.3494 | +4.6% [+3.0, +6.2]* better on 25% of bodies |
| A | L | 0.1591 | 0.3534 | +133.3% [+117.6, +150.1]* better on 0% of bodies |
| A | Lc | 0.1661 | 0.3515 | +119.7% [+105.4, +134.5]* better on 0% of bodies |
| B | S | 0.4183 | 0.4456 | +9.2% [+5.9, +12.5]* better on 29% of bodies |
| B | L | 0.1941 | 0.4793 | +158.9% [+141.9, +176.6]* better on 1% of bodies |
| B | Lc | 0.1782 | 0.4674 | +176.9% [+155.5, +199.1]* better on 3% of bodies |

G3: Lc vs L at shift 0 (negative = Lc better)

| task | L | Lc | Lc vs L |
|---|---|---|---|
| A | 0.1591 | 0.1661 | +6.8% [+3.5, +10.2]* better on 41% of bodies |
| B | 0.1941 | 0.1782 | -5.2% [-9.4, -0.8]* better on 68% of bodies |

All variants at shift 0 vs scratch (negative = better than scratch)

| task | variant | error | vs S |
|---|---|---|---|
| A | L | 0.1591 | -50.8% [-53.7, -47.7]* better on 99% of bodies |
| A | Lc | 0.1661 | -48.3% [-51.2, -45.4]* better on 100% of bodies |
| B | L | 0.1941 | -50.1% [-53.6, -46.6]* better on 98% of bodies |
| B | Lc | 0.1782 | -54.1% [-57.2, -51.0]* better on 100% of bodies |
