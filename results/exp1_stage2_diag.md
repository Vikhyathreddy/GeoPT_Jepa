L and LJ against native geometry N (same pretraining seed; n = 64; mean over fine-tune seeds 0, 1; 95% CI over bodies)

| task | variant | condition | pretrain seed | error | N | vs N | share of clean advantage over N retained |
|---|---|---|---|---|---|---|---|
| A | L | clean | s0 | 0.1154 | 0.1450 | -20.3% [-22.9, -17.6]* better on 94% of bodies | 100% [100, 100] |
| A | L | lblfrz05 | s0 | 0.1176 | 0.1450 | -19.0% [-21.9, -16.1]* better on 91% of bodies | 93% [86, 99] |
| A | L | lblfrz2 | s0 | 0.1272 | 0.1450 | -11.2% [-14.4, -7.9]* better on 80% of bodies | 60% [47, 72] |
| A | L | lblfrz4 | s0 | 0.1360 | 0.1450 | -4.5% [-7.5, -1.5]* better on 67% of bodies | 30% [18, 42] |
| A | L | lblfrz8 | s0 | 0.1561 | 0.1450 | +9.9% [+5.8, +14.0]* better on 30% of bodies | -37% [-62, -18] |
| A | L | lblfrs4 | s0 | 0.1373 | 0.1450 | -4.2% [-7.5, -0.8]* better on 68% of bodies | 26% [11, 39] |
| A | L | pos015 | s0 | 0.1156 | 0.1450 | -19.9% [-22.7, -17.1]* better on 91% of bodies | 99% [94, 105] |
| A | L | pos030 | s0 | 0.1183 | 0.1450 | -17.9% [-20.7, -15.2]* better on 88% of bodies | 90% [80, 101] |
| A | L | clean | s1 | 0.1200 | 0.1386 | -13.5% [-16.2, -10.7]* better on 85% of bodies | 100% [100, 100] |
| A | L | lblfrz2 | s1 | 0.1309 | 0.1386 | -4.8% [-7.8, -1.6]* better on 71% of bodies | 41% [20, 58] |
| A | LJ | clean | s0 | 0.1110 | 0.1450 | -23.3% [-25.9, -20.6]* better on 96% of bodies | 100% [100, 100] |
| A | LJ | lblfrz2 | s0 | 0.1249 | 0.1450 | -12.8% [-15.5, -10.1]* better on 83% of bodies | 59% [50, 67] |
| A | LJ | lblfrz4 | s0 | 0.1403 | 0.1450 | -2.2% [-5.6, +1.6]  better on 58% of bodies | 14% [-0, 27] |
| B | L | clean | s0 | 0.1135 | 0.1203 | -5.2% [-7.6, -2.6]* better on 67% of bodies | 100% [100, 100] |
| B | L | lblfrz05 | s0 | 0.1127 | 0.1203 | -5.5% [-8.0, -2.9]* better on 65% of bodies | 111% [92, 144] |
| B | L | lblfrz2 | s0 | 0.1177 | 0.1203 | -1.1% [-3.5, +1.5]  better on 49% of bodies | 37% [-16, 72] |
| B | L | lblfrz4 | s0 | 0.1220 | 0.1203 | +2.1% [-0.6, +4.9]  better on 43% of bodies | -25% [-117, 20] |
| B | L | lblfrz8 | s0 | 0.1426 | 0.1203 | +19.3% [+15.4, +23.1]* better on 16% of bodies | -329% [-752, -189] |
| B | L | lblfrs4 | s0 | 0.1211 | 0.1203 | +1.7% [-1.1, +4.3]  better on 42% of bodies | -12% [-101, 31] |
| B | L | pos015 | s0 | 0.1133 | 0.1203 | -5.0% [-7.8, -2.1]* better on 63% of bodies | 102% [72, 136] |
| B | L | pos030 | s0 | 0.1151 | 0.1203 | -3.3% [-6.2, -0.5]* better on 57% of bodies | 75% [25, 122] |
| B | L | clean | s1 | 0.0992 | 0.1205 | -16.4% [-18.9, -13.8]* better on 87% of bodies | 100% [100, 100] |
| B | L | lblfrz2 | s1 | 0.1110 | 0.1205 | -5.7% [-8.7, -2.7]* better on 65% of bodies | 44% [28, 58] |
| B | LJ | clean | s0 | 0.1127 | 0.1203 | -5.6% [-8.3, -2.9]* better on 66% of bodies | 100% [100, 100] |
| B | LJ | lblfrz2 | s0 | 0.1160 | 0.1203 | -2.5% [-5.1, +0.2]  better on 54% of bodies | 56% [19, 84] |
| B | LJ | lblfrz4 | s0 | 0.1211 | 0.1203 | +1.4% [-1.4, +4.1]  better on 48% of bodies | -11% [-90, 28] |

Pretext error against clean targets, split by particle type

Validation samples: 800; wall-hitting particles 17.8%, near-body (|vd0| < 0.15) 25.6%; sticking signal = 1.44% of the t = 1,2 target variance.

| checkpoint | MSE (all 6 outputs) | hit | free | hit share of error | near | far | hit / clean hit (same seed) | free / clean free (same seed) |
|---|---|---|---|---|---|---|---|---|
| L clean s0 | 0.00175 | 0.00321 | 0.00143 | 33% | 0.00294 | 0.00134 | 1.00 | 1.00 |
| LJ clean s0 | 0.00173 | 0.00321 | 0.00141 | 33% | 0.00293 | 0.00132 | 1.00 | 1.00 |
| L lblfrs4 s0 | 0.00323 | 0.00571 | 0.00269 | 32% | 0.00498 | 0.00262 | 1.78 | 1.88 |
| L lblfrz05 s0 | 0.00187 | 0.00338 | 0.00155 | 32% | 0.00308 | 0.00146 | 1.05 | 1.08 |
| L lblfrz2 s0 | 0.00250 | 0.00427 | 0.00212 | 30% | 0.00386 | 0.00204 | 1.33 | 1.48 |
| L lblfrz4 s0 | 0.00355 | 0.00607 | 0.00300 | 30% | 0.00531 | 0.00294 | 1.89 | 2.10 |
| L lblfrz8 s0 | 0.00580 | 0.00970 | 0.00496 | 30% | 0.00825 | 0.00496 | 3.02 | 3.47 |
| L pos015 s0 | 0.00193 | 0.00380 | 0.00153 | 35% | 0.00323 | 0.00149 | 1.18 | 1.07 |
| L pos030 s0 | 0.00303 | 0.00756 | 0.00205 | 44% | 0.00533 | 0.00225 | 2.35 | 1.44 |
| L clean s1 | 0.00181 | 0.00421 | 0.00129 | 41% | 0.00353 | 0.00122 | 1.00 | 1.00 |
| L lblfrz2 s1 | 0.00250 | 0.00533 | 0.00188 | 38% | 0.00436 | 0.00186 | 1.26 | 1.46 |
| LJ lblfrz2 s0 | 0.00249 | 0.00415 | 0.00213 | 30% | 0.00381 | 0.00204 | 1.29 | 1.51 |
| LJ lblfrz4 s0 | 0.00353 | 0.00590 | 0.00301 | 30% | 0.00522 | 0.00295 | 1.84 | 2.14 |
