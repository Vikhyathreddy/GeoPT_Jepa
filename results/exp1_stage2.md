
label sweep: L vs clean L

| task | a | b | error a | error b | a vs b |
|---|---|---|---|---|---|
| A | L lblfrz05 s0 | L clean s0 | 0.1176 | 0.1154 | +1.7% [+0.1, +3.4]* better on 42% of bodies |
| A | L lblfrz2 s0 | L clean s0 | 0.1272 | 0.1154 | +12.3% [+9.2, +15.4]* better on 20% of bodies |
| A | L lblfrz4 s0 | L clean s0 | 0.1360 | 0.1154 | +21.6% [+17.7, +25.3]* better on 15% of bodies |
| A | L lblfrz8 s0 | L clean s0 | 0.1561 | 0.1154 | +39.7% [+34.5, +44.8]* better on 4% of bodies |
| A | L lblfrs4 s0 | L clean s0 | 0.1373 | 0.1154 | +21.7% [+17.7, +25.8]* better on 13% of bodies |
| B | L lblfrz05 s0 | L clean s0 | 0.1127 | 0.1135 | -0.2% [-1.4, +0.8]  better on 52% of bodies |
| B | L lblfrz2 s0 | L clean s0 | 0.1177 | 0.1135 | +4.9% [+2.8, +7.1]* better on 33% of bodies |
| B | L lblfrz4 s0 | L clean s0 | 0.1220 | 0.1135 | +8.7% [+5.8, +11.7]* better on 32% of bodies |
| B | L lblfrz8 s0 | L clean s0 | 0.1426 | 0.1135 | +26.7% [+23.0, +30.4]* better on 8% of bodies |
| B | L lblfrs4 s0 | L clean s0 | 0.1211 | 0.1135 | +8.0% [+5.4, +10.6]* better on 31% of bodies |

fresh vs frozen noise at sigma = 4 (L)

| task | a | b | error a | error b | a vs b |
|---|---|---|---|---|---|
| A | L lblfrs4 s0 | L lblfrz4 s0 | 0.1373 | 0.1360 | +0.9% [-1.7, +3.5]  better on 52% of bodies |
| B | L lblfrs4 s0 | L lblfrz4 s0 | 0.1211 | 0.1220 | +0.0% [-2.0, +2.1]  better on 54% of bodies |

LT (clean, reused) vs L under label noise; positive = L better

| task | a | b | error a | error b | a vs b |
|---|---|---|---|---|---|
| A | LT clean s0 | L clean s0 | 0.1480 | 0.1154 | +31.5% [+27.0, +35.9]* better on 6% of bodies |
| A | LT lblfrz2 s0 | L lblfrz2 s0 | 0.1480 | 0.1272 | +19.0% [+14.4, +23.8]* better on 23% of bodies |
| A | LT lblfrz4 s0 | L lblfrz4 s0 | 0.1480 | 0.1360 | +9.6% [+5.9, +13.4]* better on 28% of bodies |
| A | LT lblfrz8 s0 | L lblfrz8 s0 | 0.1480 | 0.1561 | -4.3% [-7.5, -0.9]* better on 67% of bodies |
| A | LT lblfrs4 s0 | L lblfrs4 s0 | 0.1480 | 0.1373 | +10.1% [+5.9, +14.5]* better on 34% of bodies |
| B | LT clean s0 | L clean s0 | 0.1203 | 0.1135 | +7.3% [+4.6, +10.0]* better on 29% of bodies |
| B | LT lblfrz2 s0 | L lblfrz2 s0 | 0.1203 | 0.1177 | +2.8% [+0.2, +5.4]* better on 48% of bodies |
| B | LT lblfrz4 s0 | L lblfrz4 s0 | 0.1203 | 0.1220 | -0.3% [-2.8, +2.3]  better on 50% of bodies |
| B | LT lblfrz8 s0 | L lblfrz8 s0 | 0.1203 | 0.1426 | -13.8% [-16.7, -10.9]* better on 79% of bodies |
| B | LT lblfrs4 s0 | L lblfrs4 s0 | 0.1203 | 0.1211 | +0.1% [-2.4, +2.7]  better on 55% of bodies |

LJ vs its clean model

| task | a | b | error a | error b | a vs b |
|---|---|---|---|---|---|
| A | LJ lblfrz2 s0 | LJ clean s0 | 0.1249 | 0.1110 | +14.8% [+11.9, +17.7]* better on 19% of bodies |
| A | LJ lblfrz4 s0 | LJ clean s0 | 0.1403 | 0.1110 | +29.4% [+24.7, +34.1]* better on 11% of bodies |
| B | LJ lblfrz2 s0 | LJ clean s0 | 0.1160 | 0.1127 | +4.1% [+1.8, +6.2]* better on 41% of bodies |
| B | LJ lblfrz4 s0 | LJ clean s0 | 0.1211 | 0.1127 | +8.4% [+5.8, +11.2]* better on 25% of bodies |

LJ vs L at the same condition; negative = LJ better

| task | a | b | error a | error b | a vs b |
|---|---|---|---|---|---|
| A | LJ clean s0 | L clean s0 | 0.1110 | 0.1154 | -3.3% [-5.2, -1.4]* better on 63% of bodies |
| A | LJ lblfrz2 s0 | L lblfrz2 s0 | 0.1249 | 0.1272 | -0.5% [-3.3, +2.2]  better on 53% of bodies |
| A | LJ lblfrz4 s0 | L lblfrz4 s0 | 0.1403 | 0.1360 | +2.9% [+0.3, +5.6]* better on 40% of bodies |
| B | LJ clean s0 | L clean s0 | 0.1127 | 0.1135 | -0.4% [-1.7, +0.9]  better on 54% of bodies |
| B | LJ lblfrz2 s0 | L lblfrz2 s0 | 0.1160 | 0.1177 | -1.2% [-2.6, +0.2]  better on 58% of bodies |
| B | LJ lblfrz4 s0 | L lblfrz4 s0 | 0.1211 | 0.1220 | -0.4% [-2.0, +1.2]  better on 57% of bodies |

second pretraining seed

| task | a | b | error a | error b | a vs b |
|---|---|---|---|---|---|
| A | L clean s1 | L clean s0 | 0.1200 | 0.1154 | +5.8% [+2.2, +9.5]* better on 41% of bodies |
| A | L lblfrz2 s1 | L clean s1 | 0.1309 | 0.1200 | +11.1% [+8.1, +14.1]* better on 25% of bodies |
| A | L lblfrz2 s0 | L clean s0 | 0.1272 | 0.1154 | +12.3% [+9.2, +15.4]* better on 20% of bodies |
| A | N clean s1 | N clean s0 | 0.1386 | 0.1450 | -2.7% [-6.0, +0.5]  better on 60% of bodies |
| A | LT clean s1 | LT clean s0 | 0.1274 | 0.1480 | -12.6% [-15.6, -9.5]* better on 78% of bodies |
| A | LT clean s1 | N clean s1 | 0.1274 | 0.1386 | -7.7% [-10.1, -5.1]* better on 80% of bodies |
| A | LT clean s0 | N clean s0 | 0.1480 | 0.1450 | +3.2% [+0.0, +6.4]* better on 44% of bodies |
| A | L clean s1 | N clean s1 | 0.1200 | 0.1386 | -13.5% [-16.2, -10.7]* better on 85% of bodies |
| B | L clean s1 | L clean s0 | 0.0992 | 0.1135 | -11.5% [-14.1, -9.0]* better on 79% of bodies |
| B | L lblfrz2 s1 | L clean s1 | 0.1110 | 0.0992 | +13.2% [+10.8, +15.6]* better on 15% of bodies |
| B | L lblfrz2 s0 | L clean s0 | 0.1177 | 0.1135 | +4.9% [+2.8, +7.1]* better on 33% of bodies |
| B | N clean s1 | N clean s0 | 0.1205 | 0.1203 | +0.8% [-2.3, +4.2]  better on 61% of bodies |
| B | LT clean s1 | LT clean s0 | 0.1132 | 0.1203 | -5.1% [-8.1, -2.1]* better on 64% of bodies |
| B | LT clean s1 | N clean s1 | 0.1132 | 0.1205 | -5.2% [-7.4, -3.1]* better on 67% of bodies |
| B | LT clean s0 | N clean s0 | 0.1203 | 0.1203 | +0.7% [-1.6, +3.0]  better on 51% of bodies |
| B | L clean s1 | N clean s1 | 0.0992 | 0.1205 | -16.4% [-18.9, -13.8]* better on 87% of bodies |
