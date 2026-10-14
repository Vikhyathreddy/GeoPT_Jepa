# EXP3_RESULTS: Experiment 3: do latent-transport features plan better?

Spec: `experiments/03_action_ranking.md`.

## Pre-registered predictions
Copied verbatim from the spec before any run of this experiment. Not edited afterwards.

- **P1 (null; the analyst leans this way).** Ranking quality tracks field accuracy, so L >= LT on
  rho in every regime, and the original LT edge was noise.
- **P2 (alternative).** LT's features encode action-relevant structure better than its field error
  suggests: LT > L on rho in the probe and in R3 (CI excluding 0), at least on task B, while L still
  beats LT on field error.
- **P3.** With 8 actions per body, R1 planners clearly beat the original single-action surrogates
  (plan-demo rho: B 0.82-0.86, A 0.70-0.84), because multi-action data shows the effect of the
  action directly.
- **P4.** Dense field supervision beats KPI-only supervision at this budget (R1 > R3), and R2 gives
  at most a small ranking gain over R1.

## Outcomes
(Added below as stages complete.)

### Stage 1 (Gate 1): data, probes for 7 backbones, R1 and R3 for S, L, LT

**Data.**
- **`data/exp3/rank_train.npz`:** TRAIN bodies 0-63, each with 8 distinct actions out of the 36 candidates, drawn with `np.random.default_rng(0)`. That is 512 solver runs per task. It stores alpha, the q field, the plume field and both costs.
- **`data/exp3/rank_test_costs.npz`:** all 100 TEST bodies × 36 actions. The 40 bodies shared with `results/plan_true_costs.npz` reproduce it exactly (max abs diff A 0.0, B 0.0).
- **The sub-grid shortcut failed its check.** The within-body Spearman between true sub-grid and full-grid task-B costs has a minimum of 0.958 on the first 10 test bodies; the requirement is >= 0.99. Over all 100 bodies the minimum is 0.958 and the mean 0.985.
  - As the spec requires, task B therefore uses all fluid cells: for probe features, for evaluation costs, and for R3 pooling at test time.

**Method.**
- **Flow-aligned frame and costs** exactly as in `plan_demo.py`:
  - Task A: max of the predicted q over the 128 surface points.
  - Task B: sum of the predicted c, clipped to [0, 1], over the fluid cells.
  - R3 uses its scalar cost head instead.
- **R1 and R3:** 1000 steps, batch 4 bodies × 4 of their actions, 256 random fluid cells per sample for task B, fine-tuning seeds 0, 1, 2, all 64 bodies. Rows are in `results/exp3_runs.jsonl`; predicted test costs are in `results/exp3_preds/`.
- **Probes:** frozen backbone; [mean, max] features of the decoder output (and separately of the encoder tokens); ridge regression with grouped 8-fold cross-validation over training bodies. Rows are in `results/exp3_probe.jsonl`.
- **Metrics:** within-body Spearman rho (`scipy.stats.spearmanr`), normalised regret, and top-1 within 2%, over 100 test bodies × 36 actions.
- **Statistics:** paired differences of per-body values, averaged over fine-tuning seeds, with a 95% bootstrap CI over the 100 bodies.
  - The CIs cover body sampling only, not fine-tuning or pretraining seeds (one checkpoint per backbone).
- **Choices the spec left open:**
  - field normalisation from the 512 training fields; costs standardised by the 512 training costs;
  - task-B probe points = all fluid cells;
  - S and J probes use the fresh decoder that fine-tuning seed 0 starts from;
  - rho = 0 for a body whose predicted costs are constant (did not occur).

**Gate 1 tables** (`results/exp3_gate1.md`, `.json`):

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
| A | S | 3 | 0.889 | 0.074 | 0.73 | 0.0961 |
| A | L | 3 | 0.941 | 0.032 | 0.87 | 0.0631 |
| A | LT | 3 | 0.908 | 0.064 | 0.74 | 0.0828 |
| B | S | 3 | 0.938 | 0.056 | 0.84 | 0.1161 |
| B | L | 3 | 0.944 | 0.049 | 0.85 | 0.1240 |
| B | LT | 3 | 0.942 | 0.052 | 0.83 | 0.1079 |

R1 paired differences (per-body values averaged over seeds; 95% CI over bodies)

| task | metric | LT - L | LT - S | L - S |
|---|---|---|---|---|
| A | rho | -0.033 [-0.041, -0.027]* better on 6% of bodies | +0.019 [+0.011, +0.027]* better on 74% of bodies | +0.052 [+0.041, +0.065]* better on 98% of bodies |
| A | regret | +0.032 [+0.014, +0.051]* better on 19% of bodies | -0.010 [-0.035, +0.015]  better on 35% of bodies | -0.043 [-0.067, -0.019]* better on 48% of bodies |
| B | rho | -0.002 [-0.006, +0.001]  better on 42% of bodies | +0.003 [-0.000, +0.007]  better on 55% of bodies | +0.006 [+0.002, +0.009]* better on 62% of bodies |
| B | regret | +0.003 [-0.007, +0.015]  better on 25% of bodies | -0.004 [-0.020, +0.012]  better on 37% of bodies | -0.007 [-0.022, +0.007]  better on 35% of bodies |

R3: mean over fine-tune seeds 0-2 (rho / normalised regret / top-1 within 2%)

| task | backbone | seeds | rho | regret | top-1 |
|---|---|---|---|---|---|
| A | S | 3 | 0.843 | 0.163 | 0.55 |
| A | L | 3 | 0.878 | 0.096 | 0.66 |
| A | LT | 3 | 0.869 | 0.096 | 0.71 |
| B | S | 3 | 0.900 | 0.061 | 0.81 |
| B | L | 3 | 0.949 | 0.037 | 0.89 |
| B | LT | 3 | 0.935 | 0.042 | 0.86 |

R3 paired differences (per-body values averaged over seeds; 95% CI over bodies)

| task | metric | LT - L | LT - S | L - S |
|---|---|---|---|---|
| A | rho | -0.008 [-0.019, +0.000]  better on 50% of bodies | +0.026 [+0.016, +0.036]* better on 84% of bodies | +0.034 [+0.022, +0.049]* better on 84% of bodies |
| A | regret | -0.001 [-0.023, +0.023]  better on 40% of bodies | -0.068 [-0.098, -0.039]* better on 61% of bodies | -0.067 [-0.098, -0.039]* better on 57% of bodies |
| B | rho | -0.015 [-0.017, -0.012]* better on 11% of bodies | +0.035 [+0.031, +0.039]* better on 96% of bodies | +0.049 [+0.045, +0.054]* better on 100% of bodies |
| B | regret | +0.006 [-0.004, +0.015]  better on 25% of bodies | -0.018 [-0.031, -0.005]* better on 50% of bodies | -0.024 [-0.036, -0.013]* better on 51% of bodies |

Original plan-demo surrogates (single-action training, seed 0, first 40 test bodies), re-scored with scipy.stats.spearmanr

| task | backbone | rho (scipy) | rho (argsort, as reported) | normalised regret | top-1 |
|---|---|---|---|---|---|
| A | S | 0.701 | 0.701 | 0.228 | 0.50 |
| A | L | 0.841 | 0.841 | 0.171 | 0.47 |
| A | LT | 0.768 | 0.768 | 0.186 | 0.47 |
| B | S | 0.816 | 0.816 | 0.208 | 0.50 |
| B | L | 0.847 | 0.847 | 0.175 | 0.55 |
| B | LT | 0.857 | 0.857 | 0.117 | 0.72 |

**Figures:**
- `figs/exp3_ranking.png`: rho and normalised regret by backbone for the probe, R1 and R3.
- `figs/exp3_accuracy_vs_ranking.png`: R1 field error vs rho.
  - **Caveat:** the task-B field error is measured on the original single-action test set, where U ranges over 0.5-1.5, while R1 is trained at U = 1 only. It therefore mixes accuracy with extrapolation in U. Task A's q does not depend on U.

**Reading.**
- **There is no LT-over-L ranking signal in any regime.**
  - **R1:** LT - L = -0.033 [-0.041, -0.027] on A (L better on 94% of bodies); -0.002 [-0.006, +0.001] on B, a tie.
  - **R3:** -0.008 [-0.019, +0.000] on A, a tie; -0.015 [-0.017, -0.012] on B, L better.
  - **Probe:** decoder features +0.003 [-0.014, +0.020] on A, a tie, and -0.014 [-0.018, -0.011] on B, L better. Encoder features -0.034 on A (L better) and -0.005 on B, a tie.
  - Regret tells the same story.
- **The original plan demo's LT edge on task B does not reappear.** That was regret 8.8 vs 12.2 on 40 bodies with one seed. With multi-action training, 100 bodies and 3 seeds, LT ties L under R1 and trails it under R3 and the probe.
  - The re-score with tie-aware `spearmanr` moves the original numbers by < 0.001.
- **Pretraining helps ranking clearly.** L - S is positive in every regime and task (R1 A +0.052, B +0.006; R3 A +0.034, B +0.049; decoder probe A +0.047, B +0.095).
- **Among the frozen backbones (decoder probe), on task B:** L and LJ rank best, then N and LT. On task A, N, LJ, LT and L tie (0.847-0.850).
  - J's decoder probe uses an untrained decoder, so it says nothing about J's features. J's encoder probe ties the best on task A and ties L on task B. (Corrected after the audit.)
  - On task B the encoder-token features rank poorly for every backbone (rho 0.40-0.49). The ranking information sits in the decoder, which reads the query positions.
- **Ranking tracks accuracy on task A; on task B the accuracy metric is confounded** (corrected after the audit). R1's task-B field error on the original test set is dominated by extrapolation in U: per-body error correlates with |U - 1| (Spearman 0.75-0.85).
  - Near U = 1 (|U - 1| < 0.2, 36 bodies), LT and L tie on field error (-0.002 [-0.005, +0.0005]) and both beat S. At U = 1, the mean relative cost error over the 36 actions gives L ≈ LT < S.
  - That matches the rho order, so the data do not show ranking departing from accuracy on task B.
  - On A, R1's field error orders the backbones exactly as rho does (L < LT < S in error; L > LT > S in rho).
  - On B, R1's rho values are within 0.006 of each other while field errors differ by 15%, with the U caveat above.

**Predictions so far.**
- **P1 (null: L >= LT on rho in every regime, so the original LT edge was noise): supported.** LT never beats L with a CI excluding 0. L is strictly better in 4 of the 8 regime × task cells (R1 A, R3 B, decoder probe B, encoder probe A). The other 4 are ties; in one of them, the task-A decoder probe, LT's point estimate is ahead by +0.003 [-0.014, +0.020].
- **P2 (alternative: LT > L in the probe and in R3, at least on task B): not supported.** LT < L on task B in both the probe (-0.014) and R3 (-0.015), with CIs excluding 0.
  - The second half of P2 (L beats LT on field error) holds on task A (LT +36% [+32, +40] vs L, L better on 99% of bodies) and is a tie on task B near U = 1. The original sentence here was wrong for task A (corrected after the audit).
- **P3 (R1 beats the original single-action surrogates): the outcome is supported; the cause is not identified.** On the first 40 test bodies, every paired R1 - plan-demo rho difference excludes 0: A S +0.163 [+0.122, +0.210], L +0.085 [+0.057, +0.121], LT +0.119 [+0.079, +0.166]; B S +0.127, L +0.102, LT +0.090. R1 is averaged over fine-tuning seeds 0-2; the plan demo uses seed 0.
  - **Confounds:** 8 times the simulations; U = 1 training on task B; 16 vs 8 samples per step; different training bodies.
  - KPI-only R3 beats the plan demo by similar margins on task B, which points to the dataset rather than to field supervision. (Intervals and confounds added after the audit.)
  - The comparison mixes "multi-action data" with "8 times as many simulations" (512 vs 64).
- **P4, first half (dense field supervision beats KPI-only, R1 > R3): supported on task A; on task B only for scratch** (corrected after the audit).
  - Task A, rho: S +0.046, L +0.064, LT +0.039 (all CIs exclude 0; seed-consistent).
  - Task B: S +0.038 [+0.032, +0.044]; L -0.005 [-0.012, +0.000], a tie; LT +0.007 [+0.001, +0.013] on body CIs only, reversed with fine-tuning seed 0.
  - On normalised regret, R1 beats R3 for no backbone on task B.
  - R1 > R3 for every backbone on task A (by 0.04-0.06) and for S and LT on task B.
  - For L on task B, R3 ties R1 (0.949 vs 0.944).
  - The R2 half of P4 awaits Stage 2.

**Is there any LT-vs-L ranking signal, and what would Stage 2 cost?**
- **No LT advantage.** Where the two differ with a CI excluding 0, L is better.
- **Stage 2 as specified** needs about 62 runs, measured here at about 2 core-min per task-A run and about 6 per task-B run on a loaded machine:
  - choosing lambda for R2 on validation bodies 56-63 with S and L (about 8 runs);
  - R1, R2 and R3 for N and LJ (24 runs);
  - R2 for S, L and LT (18 runs) plus R2 for N and LJ (12 runs).
  - That is about 4-5 core-hours, roughly 70-80 minutes on 4 idle cores. R2's ranking term (smooth max of q, mean of c, RankNet) still needs to be implemented.
- **Expected value: low for the LT question.** Every Stage 1 regime already ranks LT <= L. Stage 2's main remaining value is P4's R2 half and the N / LJ comparison.


## Independent audit (step 5, 2026-10-14)
Three agents recomputed the probe, R1 and R3 tables (including the per-body metrics from the saved predicted costs), checked `exp3.py` against the spec and challenged the claims.
- **Every number reproduces, and no bug was found.**
- **The P1 and P2 verdicts are unchanged.** No pretraining or fine-tuning seed shows LT > L on rho in any cell. Corrections are applied in place above, marked "corrected after the audit".

**Further points.**
- **Robustness of the L-better cells across fine-tuning seeds:**
  - R1 A (LT - L: -0.034, -0.033, -0.033) and R3 A are stable.
  - R3 B (-0.002, -0.028, -0.014; across-seed t interval [-0.047, +0.018]) is fine-tuning-seed fragile.
  - So 3 cells are robust (R1 A, decoder probe B, encoder probe A), plus R3 B on body CIs only.
- **Pretraining seed 1, checked by the audit (probes only, no training):**
  - task B decoder: LT - L = -0.022 [-0.027, -0.017] (seed 0: -0.014);
  - task A decoder: -0.002, a tie;
  - task A encoder: -0.020 [-0.036, -0.007];
  - task B encoder: -0.015 [-0.031, +0.001], a tie.
  - The probe conclusions replicate at pretraining seed 1. The R1/R3 LT-vs-L claims stay single-pretraining-seed until the step-4 runs.
- **P1's first clause (ranking tracks field accuracy):** supported on task A; on task B the original field-error metric is confounded by U (see Reading).
- **R3 choices not recorded before (now recorded):**
  - The RankNet term averages, rather than sums, over the 24 within-body pairs per batch.
  - Each task-B action is pooled over its own random 256 fluid cells during training.
  - Neither treats L and LT differently.
- **Ties in task A:** only 56% of the alpha / alpha + 180 true-cost pairs are bit-equal (the rest differ by up to 2.7e-15). Rounding changes rho by at most 0.0009, and no verdict changes.
- **Statistics:** intervals cover the 100 test bodies only; one pretraining checkpoint per backbone, except the probes noted above.
