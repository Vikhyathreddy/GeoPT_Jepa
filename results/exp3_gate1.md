Linear probes (100 test bodies x 36 actions): within-body Spearman rho / normalised regret / top-1 within 2%

| task | backbone | decoder features | encoder features |
|---|---|---|---|
| A | S | 0.800 / 0.200 / 0.46 | 0.808 / 0.253 / 0.43 |
| A | N | 0.850 / 0.153 / 0.52 | 0.839 / 0.147 / 0.48 |
| A | J | 0.819 / 0.232 / 0.44 | 0.845 / 0.182 / 0.51 |
| A | L | 0.847 / 0.138 / 0.57 | 0.859 / 0.190 / 0.46 |
| A | LJ | 0.850 / 0.142 / 0.57 | 0.850 / 0.145 / 0.49 |
| A | LT | 0.850 / 0.181 / 0.55 | 0.825 / 0.155 / 0.45 |
| A | LT0 | 0.804 / 0.277 / 0.35 | 0.808 / 0.270 / 0.39 |
| B | S | 0.866 / 0.142 / 0.60 | 0.403 / 0.586 / 0.28 |
| B | N | 0.950 / 0.035 / 0.88 | 0.479 / 0.537 / 0.24 |
| B | J | 0.908 / 0.066 / 0.73 | 0.489 / 0.627 / 0.26 |
| B | L | 0.962 / 0.030 / 0.90 | 0.490 / 0.588 / 0.25 |
| B | LJ | 0.961 / 0.045 / 0.86 | 0.492 / 0.597 / 0.24 |
| B | LT | 0.947 / 0.043 / 0.86 | 0.485 / 0.534 / 0.27 |
| B | LT0 | 0.904 / 0.088 / 0.74 | 0.440 / 0.626 / 0.22 |

Probe paired differences in rho (95% bootstrap CI over the 100 test bodies)

| task | features | LT - L | LT - S | L - S |
|---|---|---|---|---|
| A | dec | +0.003 [-0.014, +0.020]  better on 61% of bodies | +0.050 [+0.024, +0.081]* better on 82% of bodies | +0.047 [+0.021, +0.073]* better on 74% of bodies |
| A | enc | -0.034 [-0.050, -0.019]* better on 24% of bodies | +0.017 [+0.002, +0.031]* better on 66% of bodies | +0.051 [+0.036, +0.068]* better on 89% of bodies |
| B | dec | -0.014 [-0.018, -0.011]* better on 19% of bodies | +0.081 [+0.073, +0.090]* better on 100% of bodies | +0.095 [+0.086, +0.105]* better on 100% of bodies |
| B | enc | -0.005 [-0.021, +0.011]  better on 47% of bodies | +0.081 [+0.057, +0.106]* better on 76% of bodies | +0.086 [+0.064, +0.110]* better on 76% of bodies |

R1: mean over fine-tune seeds 0-2 (rho / normalised regret / top-1 within 2% / field rel-L2)

| task | backbone | seeds | rho | regret | top-1 | field rel-L2 |
|---|---|---|---|---|---|---|
| B | S | 3 | 0.938 | 0.056 | 0.84 | 0.1161 |
| B | L | 3 | 0.944 | 0.049 | 0.85 | 0.1240 |
| B | LT | 3 | 0.942 | 0.052 | 0.83 | 0.1079 |

R1 paired differences (per-body values averaged over seeds; 95% CI over bodies)

| task | metric | LT - L | LT - S | L - S |
|---|---|---|---|---|
| A | rho | - | - | - |
| A | regret | - | - | - |
| B | rho | -0.002 [-0.006, +0.001]  better on 42% of bodies | +0.003 [-0.000, +0.007]  better on 55% of bodies | +0.006 [+0.002, +0.009]* better on 62% of bodies |
| B | regret | +0.003 [-0.007, +0.015]  better on 25% of bodies | -0.004 [-0.020, +0.012]  better on 37% of bodies | -0.007 [-0.022, +0.007]  better on 35% of bodies |

R3: mean over fine-tune seeds 0-2 (rho / normalised regret / top-1 within 2%)

| task | backbone | seeds | rho | regret | top-1 |
|---|---|---|---|---|---|
| B | LT | 3 | 0.935 | 0.042 | 0.86 |

R3 paired differences (per-body values averaged over seeds; 95% CI over bodies)

| task | metric | LT - L | LT - S | L - S |
|---|---|---|---|---|
| A | rho | - | - | - |
| A | regret | - | - | - |
| B | rho | - | - | - |
| B | regret | - | - | - |

Original plan-demo surrogates (single-action training, seed 0, first 40 test bodies), re-scored with scipy.stats.spearmanr

| task | backbone | rho (scipy) | rho (argsort, as reported) | normalised regret | top-1 |
|---|---|---|---|---|---|
| A | S | 0.701 | 0.701 | 0.228 | 0.50 |
| A | L | 0.841 | 0.841 | 0.171 | 0.47 |
| A | LT | 0.768 | 0.768 | 0.186 | 0.47 |
| B | S | 0.816 | 0.816 | 0.208 | 0.50 |
| B | L | 0.847 | 0.847 | 0.175 | 0.55 |
| B | LT | 0.857 | 0.857 | 0.117 | 0.72 |
