# Experiment 3: Do latent-transport features plan better?

## Question
In the original planning demo (`plan_demo.py`, `plan_stats.py`) a surrogate scored 36 candidate
flow directions per body, and the one with the lowest predicted cost was picked. On the plume task
the latent-transport model (LT) had the lowest regret and the best action-ranking correlation
(Spearman 0.857 vs 0.847 for L and 0.816 for scratch), despite worse field accuracy than L. But:

- LT's edge over L was not significant (regret difference -3.4, 95% CI [-7.7, +0.7]); LT vs scratch
  was (-7.0, [-12.2, -2.2]).
- Only 40 test bodies and one seed were used.
- The surrogates were trained only for field accuracy, on one action per training body.
- On task A, L ranked best (0.84 vs 0.77 for LT).

This experiment trains surrogates for the decision itself, with more bodies and seeds, to test
whether LT's features really rank actions better.

## Setup
- **Action:** the flow direction relative to the body; 36 candidates (0, 10, ..., 350 degrees).
  Work in the flow-aligned frame exactly as `plan_demo.py` does: rotate the body and the query
  points by -alpha, and use the constant prompt 0.25 along +x (U = 1).
- **Costs:** task A = max surface speed max(q) on `ptsA` (as in `plan_demo.py`); task B = plume
  footprint, the sum of c over fluid cells at U = 1.
- **Training data** (`data/exp3/rank_train.npz`):
  - TRAIN bodies 0-63, each with 8 actions drawn uniformly at random (seed 0): 512 solver runs per
    task.
  - Store alpha, the q field (128 values), the plume field (64x64) and both costs.
  - Bodies 56-63 are the validation split for any selection (lambda, ridge strength). After
    selecting, train the final models on all 64 bodies. Never use TEST bodies for selection.
- **Test data** (`data/exp3/rank_test_costs.npz`): all 100 TEST bodies x 36 actions, true costs for
  A and B. This extends `results/plan_true_costs.npz`, which has the first 40 bodies; recompute
  those 40 and check that they match.
- **Evaluation shortcut for task B:** compute predicted costs on the 32x32 sub-grid (every other
  cell, fluid cells only). These are the same cells for every action of a body, so the scale factor
  cancels within a body. Validate this first on 10 test bodies x 36 actions using true solver
  fields: the within-body Spearman between sub-grid and full-grid true costs must be >= 0.99.
  Otherwise use the full grid.

## Backbones
S, N, J, L, LJ, LT, LT0: the existing pretrained checkpoints, unchanged.

## Training regimes (all use the same 512 solver runs)
- **R1, field.** Field regression as in the original study, on the 512 (body, action) samples.
  Batch = 4 bodies x 4 of their actions; 1000 steps. Actions are ranked by the cost computed from
  the predicted field.
- **R2, field + rank.** The R1 loss plus lambda times a RankNet loss on differentiable costs.
  - Task A: soft max of the predicted q, tau*logsumexp(q/tau) with tau = 0.05.
  - Task B: mean predicted c over the sampled fluid cells.
  - RankNet: for every pair of actions of the same body in the batch, add
    `softplus(-sign(c_k - c_l) * (chat_k - chat_l) / T)`, with costs standardised by the training
    std and T = 1.
  - Choose lambda from {0.1, 1} on the validation bodies using S and L only, then fix it for all
    backbones.
- **R3, KPI only.** No field labels at all.
  - A scalar cost head reads the decoder outputs pooled over the probe points: the 128 surface
    points for task A, the sampled fluid cells for task B.
  - Pooling: mean plus log-mean-exp. Log-mean-exp (logsumexp - log N) keeps the pooling independent
    of the number of points.
  - The pooled vector goes through a 2-layer MLP to give chat.
  - Loss: RankNet plus 0.1 x MSE on the standardised cost (the MSE fixes the scale).
- **P, linear probe.** No fine-tuning; plain NumPy, no autograd.
  - Freeze the backbone. Features are [mean, max] over the probe points of the decoder output, and
    separately of the encoder tokens. The encoder version is needed for J, whose decoder is
    untrained.
  - Fit ridge regression to the standardised cost on the 512 training pairs. Choose the ridge
    strength by grouped cross-validation over training bodies: never split one body's actions
    across folds.
  - Rank the test actions by the prediction.

New autograd ops (logsumexp along an axis, softplus, and anything else needed) go into `ag.py`
with a finite-difference test in `test_ag.py`, which must pass before they are used.

## Metrics (100 test bodies x 36 actions)
- **Primary:** within-body Spearman rho between predicted and true costs. Use
  `scipy.stats.spearmanr`, which handles ties; task A costs come in exact pairs because
  q(alpha) = q(alpha + 180).
- **Secondary:**
  - Normalised regret = (true cost of the chosen action - best) / (mean true cost - best).
  - Top-1 within 2% of the best cost.
- The original `plan_stats.py` computed Spearman from argsort ranks, without tie handling.
  Recompute the original plan-demo numbers with `scipy.stats.spearmanr` so the comparison is
  like-for-like.
- For R1 and R2, also report the field rel-L2 on the original single-action test set
  (`finetune.evaluate`), to relate accuracy to ranking.
- Fine-tune seeds {0, 1, 2}. Average per-body metrics over seeds, then run paired bootstraps across
  bodies for LT - L, LT - S and L - S.

## Stage 1
- Build the data and run the sub-grid check.
- Add the new ops with their tests.
- Run linear probes for all 7 backbones.
- Run R1 and R3 for {S, L, LT} x {A, B} x 3 seeds (36 runs). Time one run before launching the rest.

### Gate 1: stop and report
- The probe table: rho per backbone, decoder vs encoder features.
- R1 and R3 tables with CIs.
- Whether there is any LT-vs-L ranking signal, and your estimate for Stage 2.

## Stage 2
- Select lambda for R2 on the validation bodies.
- Run R1, R2 and R3 for {S, N, L, LJ, LT} x {A, B} x 3 seeds, skipping runs that are already done.
- Optional low-budget repeat with 16 bodies x 8 actions. Differences between backbones may be
  larger when data is scarce.

## Figures
- `figs/exp3_ranking.png`: per task, rho (and normalised regret) by backbone, grouped by regime,
  with CIs.
- `figs/exp3_accuracy_vs_ranking.png`: field rel-L2 (x) against rho (y), one point per
  backbone x regime. Does ranking track accuracy?

## Predictions (pre-register by copying into EXP3_RESULTS.md before running)
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

## Deliverables
`EXP3_RESULTS.md` (predictions, method, tables, figures, outcome per prediction, caveats),
`results/exp3_*`, `figs/exp3_*.png`, and a commit per stage.
