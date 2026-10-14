| task | condition | target noise | L | LT | LT vs L | L vs clean L | LT vs clean LT |
|---|---|---|---|---|---|---|---|
| A | clean | 0.00 | 0.1154 | 0.1480 | +31.5% [+27.0, +35.9]* better on 6% of bodies | - | - |
| A | lblfrz05 | 0.50 | 0.1176 | 0.1480 | +30.0% [+25.2, +34.7]* better on 7% of bodies | +1.7% [+0.1, +3.4]* better on 42% of bodies | +0.0% [+0.0, +0.0]  better on 0% of bodies |
| A | lblfrz2 | 2.00 | 0.1272 | 0.1480 | +19.0% [+14.4, +23.8]* better on 23% of bodies | +12.3% [+9.2, +15.4]* better on 20% of bodies | +0.0% [+0.0, +0.0]  better on 0% of bodies |
| A | pos015 | 0.25 | 0.1156 | 0.1463 | +30.0% [+25.5, +34.5]* better on 5% of bodies | +0.7% [-0.9, +2.2]  better on 45% of bodies | -0.5% [-2.2, +1.1]  better on 59% of bodies |
| A | pos030 | 0.50 | 0.1183 | 0.1502 | +30.2% [+25.8, +34.9]* better on 9% of bodies | +3.8% [+1.0, +6.5]* better on 43% of bodies | +2.2% [+0.4, +4.1]* better on 40% of bodies |
| B | clean | 0.00 | 0.1135 | 0.1203 | +7.3% [+4.6, +10.0]* better on 29% of bodies | - | - |
| B | lblfrz05 | 0.50 | 0.1127 | 0.1203 | +7.8% [+4.9, +10.8]* better on 31% of bodies | -0.2% [-1.4, +0.8]  better on 52% of bodies | +0.0% [+0.0, +0.0]  better on 0% of bodies |
| B | lblfrz2 | 2.00 | 0.1177 | 0.1203 | +2.8% [+0.2, +5.4]* better on 48% of bodies | +4.9% [+2.8, +7.1]* better on 33% of bodies | +0.0% [+0.0, +0.0]  better on 0% of bodies |
| B | pos015 | 0.25 | 0.1133 | 0.1199 | +7.1% [+4.1, +10.2]* better on 30% of bodies | +0.2% [-1.2, +1.6]  better on 47% of bodies | -0.4% [-1.0, +0.2]  better on 53% of bodies |
| B | pos030 | 0.50 | 0.1151 | 0.1217 | +6.5% [+3.6, +9.5]* better on 36% of bodies | +2.4% [+0.1, +4.7]* better on 46% of bodies | +1.0% [-0.1, +2.0]  better on 44% of bodies |

scratch / native-geometry references (n=64): {"A": {"S": 0.19283322244882584, "N": 0.1450372040271759}, "B": {"S": 0.1576971411705017, "N": 0.12026942521333694}}

| condition | family | measured target noise (t=1 / t=2 / pooled) | targets changed | L val (noisy targets) | L val vs clean targets | LT val_tgt_std |
|---|---|---|---|---|---|---|
| clean | - | 0 | 0 | 0.00177 | 0.00177 | 0.2484 |
| lblfrz05 | label | 0.587 / 0.443 / 0.500 | 100.0% | 0.04055 | 0.00191 | 0.2484 (clean LT) |
| lblfrz2 | label | 2.346 / 1.771 / 1.999 | 100.0% | 0.62064 | 0.00259 | 0.2484 (clean LT) |
| pos015 | trajectory | 0.294 / 0.226 / 0.253 | 100.0% | 0.01126 | 0.00197 | 0.2381 |
| pos030 | trajectory | 0.575 / 0.446 / 0.497 | 100.0% | 0.03774 | 0.00313 | 0.2282 |
