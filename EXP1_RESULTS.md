# EXP1_RESULTS: Experiment 1: noisy pretext targets

Spec: `experiments/01_noisy_targets.md`.

## Pre-registered predictions
Copied verbatim from the spec before any run of this experiment. Not edited afterwards.

- **P1.** Label noise barely hurts L: less than 5% relative degradation even at sigma = 2. MSE
  regression learns the conditional mean; each batch averages about 18k target values; and the
  corpus holds 12M noisy numbers against 0.15M parameters, seen only about 3 times, so there is
  nothing to memorise. L's pretext error against clean targets should stay near its clean value.
  For the same reason, frozen and fresh noise should match at full corpus size (option d), while in
  the small-corpus regime (option e) frozen noise should hurt more than fresh.
- **P2.** If L does degrade under heavy label noise, LT overtaking it is guaranteed by construction
  and does not on its own confirm the rule of thumb.
- **P3.** Trajectory noise (the fair test) is genuinely uncertain. The rule of thumb predicts that
  LT degrades more slowly than L and crosses it at some sigma_pos. The alternative is that both
  degrade alike because the noise enters both target types the same way.
- **P4.** Stochastic bounce: if JEPA's multimodality argument applies here, L should be worse at
  p = 0.5 than at both deterministic endpoints, relative to LT. Caveat: MSE on a stochastic target
  simply learns its mean, which may be harmless, and only about 18% of particles are affected.
- **P5.** LJ is the most noise-robust explicit variant, because its J term carries a noise-free
  signal.

## Outcomes
(Added below as stages complete.)

### Stage 1 (Gate 1): label noise sigma = 0.5, 2; trajectory noise sigma_pos = 0.15, 0.30; pretraining seed 0

**Method.**
- **Corpora.** Each corpus is `data/pretrain.npz` with one change: `data/exp1/pretrain_<tag>.npz`.
  - Label noise: `vd[:,1:] += sigma * 0.483 * N(0,1)`, using one standard-normal draw (seed 123) shared by both levels.
  - Trajectory noise: `X[:,1:] += sigma_pos * N(0,1)`, using one draw (seed 124) shared by both levels, then `vd[:,1:]` recomputed at the noisy points.
  - t = 0 is untouched.
  - The noise covers every sample, validation bodies included. The clean targets are stored (`vd_clean`, `X_clean`), and L's validation loss is also logged against them.
- **Pretraining.** L on all four corpora, and LT on the two trajectory corpora: 3000 steps, batch 16, seed 0, original hyper-parameters. Checkpoints are in `ckpt/exp1/`.
- **Label noise never reaches LT, LT0 or J.** After 50 steps, all three train bit-identically on the clean and label-noise corpora (`results/exp1_identity.json`: all True). The clean `ckpt/LT.npz` is therefore reused under label noise, and its fine-tuning rows are taken from `results/finetune.jsonl`.
- **Fine-tuning.** Exactly as in the original study: flow-aligned frame, n = 64, tasks A and B, fine-tuning seeds 0 and 1, the 100 TEST bodies. Rows are in `results/exp1_finetune.jsonl`.
  - The clean seed-0 references come from `results/finetune.jsonl`. The reproduction check re-ran S and L (task A, n = 64, seed 0) on this machine and matched them exactly.
- **Statistics.** Paired relative change on the same 100 test bodies, with per-body errors averaged over the 2 fine-tuning seeds, and a 95% bootstrap CI over bodies (`*` = CI excludes 0).
  - The CIs cover test-body sampling only. They do not cover pretraining-seed variance (one pretraining seed per condition) or fine-tuning-seed variance beyond the 2-seed average.
- **Choices the spec left open:** the noise seeds (123 for labels, 124 for positions); applying the noise to validation samples as well; and the x-axis, which is pooled `std(vd - vd_clean)` over t = 1,2 divided by the clean std.

**Gate 1 tables** (`results/exp1_gate1.json`, `results/exp1_gate1.md`). Test relative L2 error at n = 64, as means over fine-tuning seeds. "LT vs L" is LT's error relative to L (positive means L is better). The "vs clean" columns compare each variant with its own clean model. LT's label-noise rows are the clean LT model, identical by construction.

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


**Figure:** `figs/exp1_noise_curves.png`. It plots test error against measured target-space noise, with one panel per task × family and scratch / native-geometry reference lines.

**Reading.**
- **No crossover in any condition tested.** L beats LT by 19-31% on task A and by 3-8% on task B. Every CI excludes 0.
  - The gap narrows only under heavy label noise (lblfrz2: A +19.0%, B +2.8% [+0.2, +5.4]). That happens because L degrades while LT cannot, which is the by-construction case of P2.
- **Trajectory noise (the fair test).** Both variants degrade only slightly, and by similar amounts.
  - At pos030, L is +3.8% on A and +2.4% on B versus clean L; LT is +2.2% on A and +1.0% on B versus clean LT.
  - So LT does degrade a little more slowly, but the LT-L gap barely moves (A: +31.5% clean → +30.2%; B: +7.3% → +6.5%). At this rate a crossover would need far more noise than pos030.
- **Label noise does hurt L downstream more than P1 expected.** At sigma = 2, task A is +12.3% [+9.2, +15.4] and task B +4.9% [+2.8, +7.1]. Yet L's pretext error against clean targets rises only from 0.00177 to 0.00259.
  - So a small loss of pretext accuracy still costs some downstream accuracy. L keeps 85% (A) and 90% (B) of its gain over scratch at sigma = 2.
- **L's gain over scratch has not halved at any level tested** (lowest: 85% kept). So the spec's "refutes" verdict cannot be applied yet: it requires testing up to the noise level where that gain halves.

**Predictions so far.**
- **P1 (label noise barely hurts L, < 5% even at sigma = 2): refuted on task A (+12.3%) and borderline on B (+4.9%).** L's pretext error against clean targets stays near its clean value (0.0019 / 0.0026 vs 0.0018), which matches P1's mechanism. The fresh-vs-frozen and small-corpus parts await options d and e.
- **P2: confirmed in the weak form.** L degrades under heavy label noise, and LT, which cannot degrade, narrows the gap. It does not overtake.
- **P3 (trajectory noise): no crossover up to sigma_pos = 0.30** (target noise 0.50 × std). LT degrades slightly more slowly than L, but the gap is essentially unchanged. So far this sides with the alternative: the noise enters both target types much alike.
- **P4, P5:** not yet tested (bounce, LJ).
- **Verdict so far:** no support for the rule of thumb in this setting, not even weak support. The "refutes" verdict needs noise levels up to the point where L's gain over scratch halves (Stage 2 d and/or a larger sigma_pos).

**Stage 2 recommendation, with compute measured here (4 cores).** One L pretraining takes about 4-5 core-min, one LT about 10-17 core-min, and a fine-tuning run about 0.5-1 core-min.
1. **d. Label extremes and fresh vs frozen** (lblfrz4, lblfrz8, lblfrs4 with L and LJ): about 35 core-min. This is needed to reach the "gain halved" level for the label family.
2. **Larger trajectory noise** (pos060, and pos100 if needed; not in the menu) with L and LT: about 20 core-min per level. Without it the fair-test family can never reach a verdict.
3. **a. A second pretraining seed for clean, pos030 and lblfrz2, with L and LT:** about 60 core-min. The gaps here are large compared with the noise effects, but pretraining-seed variance is uncovered.
4. **c. Stochastic bounce** (bounce05, bounce10 with L, LT, LJ, LT0): about 90 core-min. This is the multimodality test (P4) and the most interesting remaining question.
5. **b. LJ / LT0** on the kept conditions (P5): about 18 core-min per condition.
6. **e. Small corpus:** about 40 core-min. **f. n = 16:** +50%. **g. pos005:** low value, since the curve is flat at the low end.
