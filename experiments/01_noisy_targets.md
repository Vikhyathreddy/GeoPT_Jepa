# Experiment 1: Noisy pretext targets. When do latent targets beat explicit ones?

## Question
In the original study, explicit GeoPT targets (L) beat every latent-target variant (LT, J), and
LJ = L + JEPA tied L. The rule of thumb under test: *predict explicit targets when they are exact and
cheap; latent (JEPA) targets earn their keep when the targets contain unpredictable detail.* Does LT
(or LJ) overtake L once the pretext signal is noisy?

## Design principle: where the noise enters decides what the experiment can show
- **Label noise** (added to the explicit target values `vd[:,1:]`) reaches L and LJ, but never
  LT/LT0/J. Those variants never read `vd[:,1:]`, so with the same seed their training is
  bit-identical to the clean run. Do not re-pretrain them; instead verify the identity with a
  50-step run (identical parameters) and reuse the clean checkpoints. A latent win under label noise
  is therefore guaranteed by construction whenever L degrades, which makes it weak evidence.
- **Trajectory noise** (noise on the landing positions `X[:,1:]`, as if the trajectories had been
  measured with a noisy sensor) changes both target types: L's target becomes vd(x + e) and LT's
  becomes latent(x + e). This is the fair test.
- **Stochastic dynamics** (a random stick-or-bounce outcome at the wall) makes the target
  multimodal for both. This tests JEPA's multimodality argument.

In every condition: t = 0 is clean (the static geometry is known exactly), the downstream
fine-tuning labels and test sets are clean and unchanged, and only the pretraining corpus differs.

## Conditions
| tag | family | change to the corpus | level |
|---|---|---|---|
| clean | none | none | reference |
| lblfrz05, lblfrz2 | label, frozen | `vd[:,1:] += sigma*0.483*N(0,1)`, drawn once (seed 123) and stored | sigma = 0.5, 2 |
| lblfrz4, lblfrz8 | label, frozen | as above | sigma = 4, 8 (Stage 2) |
| lblfrs4 | label, fresh | as above but redrawn every time a sample is drawn (`batch_fn` with its own RNG) | sigma = 4 (Stage 2) |
| pos015, pos030 | trajectory | `X[:,1:] += sigma_pos*N(0,1)` (stored), then `vd[:,1:]` recomputed at the noisy points with `world.vector_distance` | sigma_pos = 0.15, 0.30 |
| pos005 | trajectory | as above | sigma_pos = 0.05 (Stage 2) |
| bounce05, bounce10 | stochastic dynamics | corpus regenerated with bounce probability p at wall contact | p = 0.5, 1.0 (Stage 2) |
| small_* | memorisation regime | `--train-shapes 100` (800 samples, about 60 epochs): small_clean, small_lblfrz4, small_lblfrs4 | Stage 2 |

Measured on the original corpus: sigma_pos = 0.05 / 0.15 / 0.30 produces target-space noise of
0.09 / 0.25 / 0.50 times the clean target std, so pos030 and lblfrz05 sit at about the same
target-space noise level. For every condition, record the measured `std(vd_noisy - vd_clean)` at
t = 1,2 divided by the clean std, and use it as the common x-axis.

### Implementation notes
- Store corrupted corpora as `data/exp1/pretrain_<tag>.npz`, with the same keys as
  `data/pretrain.npz` plus `vd_clean` (and `X_clean` for pos*) so that L's pretext loss against
  clean targets can be logged.
- pos*: noisy landing points may fall inside the body. Keep them (a noisy sensor can do that).
- bounce: add `p_bounce=0.0, rng_bounce=None` to `world.transport_step` and `pretext_samples`.
  - With `p_bounce=0` the output must stay bit-identical: regenerate one 100-shape chunk and compare
    it with `np.array_equal`.
  - On wall contact, find the hit point and the outward normal n of the hit edge. With probability
    p, set v <- v - 2(v.n)n and move the remaining fraction (1 - s) of the step from the hit point.
    If that reflected segment hits the wall again, stick at the second hit. After a bounce the
    particle is free.
  - Draw the coins from a separate RNG, so that bodies, particles and velocities are identical to
    the clean corpus and only the outcomes at the wall differ.
  - Unit test on a circle of radius 0.4 (256-gon): a particle at (1, 0) with v = (-0.8, 0) ends
    within 0.01 of (0.6, 0) with bounce and of (0.4, 0) without.
  - p = 1.0 is deterministic (different physics but unimodal); p = 0.5 is multimodal. Comparing the
    two separates "different dynamics" from "unpredictable outcome". Only about 18% of particles
    touch the wall, so effects may be small. Report the fraction of t = 1,2 targets that changed.

## Variants and runs
- Pretraining: 3000 steps, batch 16, original hyper-parameters, saved to
  `ckpt/exp1/<V>_<tag>_s<seed>.npz`. The original `ckpt/L.npz` and `ckpt/LT.npz` are the clean
  seed-0 models.
- Fine-tuning: exactly as the original (flow-aligned frame, n = 64 labelled bodies, tasks A and B,
  fine-tune seeds 0 and 1, evaluated on the 100 TEST bodies). Results go to
  `results/exp1_finetune.jsonl` keyed by (variant, tag, pretrain_seed, task, n, ft_seed). Clean
  seed-0 references can be taken from `results/finetune.jsonl` once one of them has been re-run here
  and matches.
- Per pretraining run, log to `results/exp1_pretrain.jsonl`: final train/val pretext losses, L's
  (and LJ's) val loss against clean targets, and for LT/LT0 the collapse probe `val_tgt_std`.

## Stage 1 (pilot, about 60 core-minutes)
- Conditions: lblfrz05, lblfrz2, pos015, pos030; pretraining seed 0.
- Variants: L for all four. LT for the pos* conditions; for label noise, reuse the clean LT after
  the identity check.

### Gate 1: stop and report
- Table of test error (task x condition x variant), with paired CIs for LT - L and for each variant
  against its own clean version.
- L's pretext val error against clean targets, per condition.
- Measured target-space noise per condition.
- Which Stage 2 options are worth it, with your estimated compute.

## Stage 2 (menu; the user picks)
- a. Second pretraining seed (s1) for clean and every kept condition, L and LT (about 20 core-min
  per condition).
- b. LJ, plus LT0 for trajectory/bounce conditions, on all kept conditions (about 18 core-min per
  condition; under label noise LT0 is again identical to clean).
- c. Stochastic dynamics bounce05 and bounce10 with L, LT, LJ, LT0 (about 90 core-min).
- d. Label-noise extremes and fresh-vs-frozen: lblfrz4, lblfrz8, lblfrs4 with L and LJ (about
  30 core-min).
- e. Memorisation regime: small_clean, small_lblfrz4, small_lblfrs4 with L, plus LT for small_clean
  (about 40 core-min).
- f. Also fine-tune at n = 16 for every condition run so far (about +50%).
- g. pos005 to fill in the low end of the trajectory-noise curve (about 20 core-min).

## Analysis and figures
- `figs/exp1_noise_curves.png`: x = measured target-space noise / clean std; y = test rel-L2 at
  n = 64. One panel per task x family (label, trajectory); bounce in its own panel with x = p.
  Lines per variant. Horizontal reference lines for scratch and native geometry, which are
  noise-independent.
- Crossover: the smallest noise level at which LT (or LJ) beats L with a CI excluding 0, per family
  and task.
- Robustness: the noise level at which each variant's gain over scratch halves.

## Predictions (pre-register by copying into EXP1_RESULTS.md before running)
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

## Verdict rules
- **Supports the rule of thumb:** LT or LJ beats L with a CI excluding 0 on both tasks under
  trajectory noise or stochastic bounce.
- **Weak support:** a crossover appears only under label noise.
- **Refutes it for this setting:** L >= LT at every level of every family tested, up to the level
  where L's gain over scratch has halved.

## Deliverables
`EXP1_RESULTS.md` (predictions, method, tables, figure, outcome per prediction, verdict, caveats),
`results/exp1_*.jsonl`, `figs/exp1_*.png`, and a commit per stage.
