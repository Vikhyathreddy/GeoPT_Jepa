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

**Reading** (revised after an independent audit; see the end of this section).
- **No crossover in any condition tested.** L beats LT by 19-32% on task A and by 3-8% on task B. Every CI excludes 0.
  - The gap narrows only under heavy label noise (lblfrz2: A +19.0%, B +2.8% [+0.2, +5.4]). There, L degrades while LT, which never sees those targets, cannot.
- **Trajectory noise (the fair test) is uninformative at these levels.**
  - At pos030, L is +3.8% (A) and +2.4% (B) against clean L; LT is +2.2% (A) and +1.0% (B) against clean LT.
  - Per body, the difference in degradation (L minus LT) is A +1.6 points [-1.4, +4.6] and B +1.4 points [-1.0, +3.8]. The change in the LT-L gap from clean to pos030 is A -1.3 points [-5.2, +2.5] and B -0.8 points [-3.3, +1.8].
  - All of these CIs include 0. At target noise up to 0.5 × std, the data cannot tell "LT degrades more slowly" from "both degrade alike". L still keeps 96% of its gain over scratch.
- **Label noise hurts L downstream more than P1 expected.** At sigma = 2: task A +12.3% [+9.2, +15.4], task B +4.9% [+2.8, +7.1].
  - The size is poorly pinned down: on task A, fine-tuning seed 0 alone gives +7.5% and seed 1 +19.5%. Both are above 5%.
  - L's pretext error against clean targets rises 46% (0.00177 to 0.00259; 44% on all 800 validation samples). It rises 47% on the t = 0 output too, whose targets are clean in every corpus.
  - So the noise degraded the shared representation; it was not simply averaged out. Clean-target pretext error does not track the downstream cost across families: pos030 raises it more (+74%) but costs less downstream.
- **The label family can never produce the spec's "refutes" verdict.**
  - LT is fixed by construction under label noise. L would tie LT once it keeps only 58% (A) or 85% (B) of its gain over scratch.
  - So any label-noise sweep that reaches the halving level must first pass a by-construction crossover. That is "weak support", which P2 discounts.
  - A refutation (or real support) can only come from the trajectory and bounce families, where L currently keeps 96% of its gain.

**Predictions so far.**
- **P1 (label noise barely hurts L, < 5% even at sigma = 2).**
  - Refuted on task A: +12.3%, and over 5% in each fine-tuning seed.
  - Borderline on task B: +3.7% as a ratio of mean errors, +4.9% as the mean per-body change, +6.7% with seed 0 alone.
  - P1's mechanism only partly holds: L's error against clean targets rises 46%, including on the always-clean t = 0 output.
  - The fresh-vs-frozen and small-corpus parts await options d and e.
- **P2: premise met, consequence not observed.** L does degrade under heavy label noise, and LT is invariant to it, verified bit for bit. But the overtaking P2 calls guaranteed has not happened at sigma = 2; only the by-construction narrowing is seen.
- **P3 (trajectory noise): undetermined.** There is no crossover up to sigma_pos = 0.30 (target noise 0.50 × std), and the relative degradation of L and LT is indistinguishable (CIs above).
- **P4, P5:** not yet tested (bounce, LJ).
- **Verdict so far:** no support for the rule of thumb, not even weak support. No refutation is possible yet either: the family that could refute it (trajectory) has not reached the gain-halving level.

**Caveats on the statistics.**
- The CIs resample the 100 test bodies only. They cover neither fine-tuning-seed variance (2 seeds, averaged) nor pretraining-seed variance (1 seed per condition).
- Seed effects are large here (task A at lblfrz2: +7.5% vs +19.5%), so effects of a few percent should be read with care.

**Stage 2 recommendation** (revised). Compute is measured on this machine (4 cores, shared). One L pretraining takes 4-11 core-min (median 5.3), one LT about 17, and a fine-tuning run 0.5-1.2 core-min (mean 0.84).
1. **Larger trajectory noise** (pos060, pos100; not in the menu) with L and LT: about 30 core-min per level, including 8 fine-tuning runs. This is the only family that can yield a verdict, and it needs to reach the level where L's gain over scratch halves.
2. **c. Stochastic bounce** (bounce05, bounce10 with L, LT, LJ, LT0): about 100 core-min. This is the multimodality test (P4).
3. **a. A second pretraining seed for clean and every kept condition (L, LT), plus fine-tuning seed 2:** about 25 core-min per condition. Seed variance is large compared with the effects measured here.
4. **d. Label extremes and fresh vs frozen** (lblfrz4, lblfrz8, lblfrs4 with L and LJ): about 55 core-min. This settles P1's fresh-vs-frozen part, but it can only end in the by-construction crossover, so it cannot decide the rule of thumb.
5. **b. LJ / LT0 on the kept conditions** (P5): about 20 core-min per condition. **e. Small corpus:** about 45 core-min. **f. n = 16:** +50%. **g. pos005:** low value, since the curve is flat at the low end.

**Independent audit.** Four agents recomputed every table entry and prose number from the raw results; all table entries reproduce exactly. They also challenged the interpretation. This section was revised because of it:
- The task-A range is now 19-32% (it said 19-31%).
- P2 is now "premise met, consequence not observed" (it said "confirmed in the weak form").
- P3 is now "undetermined" (it said the data sided with the alternative).
- The claim that P1's mechanism was confirmed has been removed.
- The label family's inability to refute is now stated.
- Seed caveats were added and the compute estimates corrected (they were low).

## Amendment (2026-10-14, after Gate 1): yardstick and Stage 2 scope
User decision, recorded before any Stage 2 run. The text above is unchanged.

**New yardstick.** L is judged by its advantage over native geometry N: the part of L's gain that comes from the dynamics.
- The t = 0 targets stay clean in every condition, so L can never sink much below N's level.
- That means the original rule ("up to the level where L's gain over scratch has halved") could never trigger.
- Amended verdict rule: "refutes for this setting" requires L >= LT at every level tested, up to the level where L's advantage over N has halved.

**Where the dynamics information sits** (the user's measurements on the clean data and model, to be re-measured by `python exp1.py diag`):
- The part of the t = 1,2 target that comes from particles sticking to the wall is only about 1.4% of the target variance.
- Label noise at sigma = 0.5 adds noise variance about 17 times that signal, and sigma = 2 about 277 times.
- In the clean L model, wall-hitting particles are about 18% of points but carry about 36% of the remaining pretext error.
- This is why L's overall pretext fit barely moved while task A degraded 12%: the noise drowns exactly the slice that matters. P1 failed for the same reason, since it treated the target as uniform.

**Why LT cannot test the rule of thumb here.** Clean LT scores about the same as N (within about 3%), so its latent part adds little over its explicit anchor. "LT overtakes L" can therefore only mean "L fell to geometry-only level".

**Stage 2 scope (user decision).**
- **Run:**
  - option d (lblfrz4, lblfrz8 frozen; lblfrs4 fresh) for L, reusing the clean LT;
  - LJ at lblfrz2 and lblfrz4;
  - a second pretraining seed (s1) for L clean and L lblfrz2, plus N clean and LT clean.
- **Diagnostics, no training:**
  - an L-vs-N table for every condition, with paired CIs and the share of L's clean advantage retained;
  - L's pretext error against clean targets, split into wall-hitting vs free particles and near-body (|vd0| < 0.15) vs far, for every L checkpoint.
- **Skip, with reasons:**
  - **Stronger trajectory noise.** pos030 already equals label sigma = 0.5 in target space, and that barely hurt either model. At sigma_pos >= 0.6 the landing points are effectively scrambled, which erases the dynamics signal for both models instead of testing which copes better.
  - **Bounce (c).** It changes only about 18% of particles. Clean LT is about N-level, so its latent part has nothing to show there.
  - **Small corpus (e), n = 16 (f), and LT0.** Out of scope for the question as reframed.

**Prediction recorded with this amendment, before the diagnostic is run:** label noise raises the pretext error of wall-hitting particles far more than that of free particles.

### Stage 2 results (2026-10-14; scope as in the amendment above)

**Method.**
- **New corpora:** lblfrz4 and lblfrz8 scale the same seed-123 standard-normal draw as lblfrz05/2.
  - lblfrs4 adds fresh noise to every training batch through `pretrain(batch_fn=...)`, with its own generator (seed + 10_000), so initialisation and batch order match the clean run. Its validation targets are clean.
- **Identity check:** LT, LT0 and J train bit-identically on clean data and under all three new label conditions (`results/exp1_identity2.json`: all True). So the clean `ckpt/LT.npz` is again reused for every label level.
- **Pretraining:** L on lblfrz4, lblfrz8 and lblfrs4; LJ on lblfrz2 and lblfrz4; and seed 1 for L (clean, lblfrz2), N (clean) and LT (clean). The seed changes both initialisation and batch order. Each is 3000 steps; checkpoints are in `ckpt/exp1/`.
- **Fine-tuning:** as in Stage 1 (n = 64, tasks A and B, fine-tuning seeds 0 and 1). 36 new rows in `results/exp1_finetune.jsonl`.
- **Diagnostics (`python exp1.py diag`):** the explicit head's error against clean targets on the 800 validation samples.
  - Split into particles that hit the wall within two steps vs free ones, and near-body (|vd0| < 0.15) vs far.
  - Each checkpoint is compared with the clean model of the same variant and pretraining seed.
  - The sticking signal is computed as the variance of vd minus the vector distance at the straight-line (free-flight) positions.
- **Statistics:** as before. The CIs cover body sampling only. Every condition has one pretraining seed, except clean and lblfrz2 for L (and clean N and LT), which have two.

**Tables** (`results/exp1_stage2_diag.md`, `results/exp1_stage2.md`):

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

**Figure:** `figs/exp1_noise_curves.png`. It shows pretraining seed 0 only, now with sigma = 4 and 8, the fresh-noise point (open marker) and LJ. The seed-1 replicate is in the tables above.

**Reading.**
- **Where the dynamics information sits: the user's measurements reproduce.**
  - The sticking signal is 1.44% of the t = 1,2 target variance.
  - Wall-hitting particles are 17.8% of validation points and carry 33% of clean L's pretext error (the user measured 36%; 41% for the seed-1 model).
- **The prediction recorded with the amendment is refuted.** Label noise does not raise the wall-hitting particles' error more than the free particles'. The opposite holds, at every level and in both seeds:
  - sigma 2: hit × 1.33 vs free × 1.48 (seed 1: 1.26 vs 1.46);
  - sigma 4: 1.89 vs 2.10;
  - sigma 8: 3.02 vs 3.47;
  - fresh sigma 4: 1.78 vs 1.88.
  - The hit share of the error falls from 33% to 30%. Label noise degrades the explicit head broadly, not specifically on the dynamic slice.
  - The trajectory-noise checkpoints show the opposite pattern (pos030: hit × 2.35 vs free × 1.44; hit share 44%), because there the noise enters through the landing positions.
- **On the amended yardstick (advantage over N), label noise erodes L's dynamics advantage fast.**
  - Task A: L keeps 93% of its clean advantage over N at sigma 0.5, 60% at sigma 2, 30% at sigma 4, and -37% at sigma 8 (worse than N).
  - Task B: L keeps 111%, 37%, -25% and -329%. These shares are noisy, because L's clean advantage over N on task B is only 5%.
  - The halving point is between sigma 2 and 4 on A, and at about sigma 2 on B. Seed 1 replicates it at sigma 2 (A 41%, B 44% retained).
  - Trajectory noise up to pos030 keeps 90% (A) and 75% (B), so it does not reach the halving point.
- **Fresh vs frozen noise at sigma 4: no difference** (A +0.9% [-1.7, +3.5], B +0.0% [-2.0, +2.1]). At full corpus size the frozen noise is not memorised.
- **LT vs L under label noise.**
  - L stays better than (or ties) the reused clean LT up to and including the halving level: A sigma 4, L better by 9.6% [+5.9, +13.4]; B sigma 2, L better by 2.8%; B sigma 4, a tie at -0.3% [-2.8, +2.3].
  - LT is better only at sigma 8 (A by 4.3%, B by 13.8%). There, L has already fallen below native geometry (A +9.9%, B +19.3% vs N). This is the by-construction crossover P2 describes.
- **LJ is not more noise-robust than L.**
  - It degrades at least as much (A +14.8% / +29.4% at sigma 2 / 4, vs L +12.3% / +21.6%).
  - It keeps 59% / 14% of its advantage over N on task A (L 60% / 30%), and at sigma 4 on A it is 2.9% worse than L.
  - Clean LJ is 3.3% better than clean L on task A and ties on B.
- **The second pretraining seed shows large pretraining-seed variance.**
  - Clean L: seed 1 vs seed 0 is A +5.8%, B -11.5%. Clean LT: A -12.6%, B -5.1%. Clean N: A -2.7%, B +0.8% (both CIs include 0).
  - At seed 1, clean LT beats N (A -7.7%, B -5.2%, both CIs exclude 0), whereas at seed 0 it does not (A +3.2%, B +0.7%). So the amendment's premise that clean LT is about N-level holds for seed 0 only. Seed 1's latent part does add something over geometry.
  - L still beats LT within seed 1: LT vs L is A +9.1% [+5.0, +13.4], B +15.0% [+11.9, +18.2].
  - The sigma = 2 degradation of L replicates in seed 1: A +11.1% (seed 0: +12.3%), B +13.2% (seed 0: +4.9%).

**Predictions: outcomes after Stage 2.**
- **P1: refuted.**
  - Label noise at sigma = 2 costs L +12.3% (A) and +4.9% (B) with seed 0, and +11.1% and +13.2% with seed 1. All are over 5%, except seed 0 on task B, which is borderline.
  - The fresh-vs-frozen part holds: they match at full corpus size.
  - The small-corpus part was not tested (option e skipped).
- **P2: confirmed as an interpretation.** L degrades under heavy label noise, and LT overtakes it only at sigma 8, once L has fallen below geometry-only level. That crossover is by construction and does not support the rule of thumb.
- **P3: undetermined.** Stage 1 found no crossover up to sigma_pos = 0.30 and indistinguishable degradation; the family was not extended, by decision.
- **P4: not tested.** Bounce was skipped by decision.
- **P5: refuted.** LJ is no more noise-robust than L.
- **Amendment prediction (wall-hitting error rises far more than free error under label noise): refuted.**

**Verdict.**
- **Under the amended rule: refutes for the label family.** L >= LT at every label level up to the level where L's advantage over N has halved (A: sigma 4; B: sigma 2, with a tie at sigma 4). LT overtakes only at sigma 8, after L has lost its whole advantage over geometry.
- **The trajectory family stopped (by decision) before the halving level,** so no verdict can be given for it.
- **Under the spec's original rules,** a crossover that appears only under label noise reads as "weak support". Here it appears only past the point where the label family is mechanically guaranteed to cross.
- **Overall:** no evidence in this setting that latent targets beat explicit ones because the explicit targets are noisy.

**Caveats.**
- The CIs cover test-body sampling only.
- Most conditions have one pretraining seed, and the seed-1 checkpoints differ from seed 0 by 5-13%, comparable to many of the effects measured. The LT ≈ N premise itself does not survive the second seed.
- Task-B retained shares are unstable because L's clean advantage over N there is small (5%).
