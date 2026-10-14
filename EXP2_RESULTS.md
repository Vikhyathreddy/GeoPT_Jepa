# EXP2_RESULTS: Experiment 2: preferred orientation and prompt alignment

Spec: `experiments/02_orientation_prompts.md`.

## Pre-registered predictions
Copied verbatim from the spec before any run of this experiment. Not edited afterwards.

- **P1.** With per-sample prompts in a canonical frame, the pretrained variants (L, Lc) show a
  penalty that grows from 0 to 90 degrees, larger at n = 16 than at n = 64 (less data with which to
  re-learn the mapping). Scratch shows none: its random initialisation is isotropic, so any constant
  rotation of the prompts is equivalent in distribution. This would reconcile the toy world with
  GeoPT.
- **P2. Sharp test for task A: a 180 degree shift should cost nothing for every variant.**
  - Potential flow is reversal-symmetric, q(alpha) = q(alpha + 180), so a model given the prompt at
    alpha + 180 is solving exactly the aligned problem for flows from the opposite side.
  - The canonical body distribution is invariant under a 180 degree rotation (random harmonic phases,
    symmetric stretch and offset), and pretraining is isotropic in velocity.
  - So task A's penalty profile should be about 0 at 0 degrees, positive at 20-90 degrees, and back
    to about 0 at 180 degrees.
  - A clear 180 degree penalty on task A would point to a bug or a non-physical shortcut;
    investigate it before reporting.
- **P3.** Task B has no such symmetry, and the sign of its 180 degree penalty is an open question.
  The original constant-prompt run hinted that a reversed prompt was slightly better for the plume
  (0.108 vs 0.114), plausibly because reversed transport highlights the wake. Report which prompt
  direction is best.
- **P4.** Lc >= L at shift 0, because its pretraining orientations match the downstream ones.
- **P5.** If no shift penalty appears even here, GeoPT's alignment effect probably depends on
  something this toy lacks (real canonical-geometry semantics, Transolver's slice attention, 3D, far
  larger models). Say so plainly.

## Outcomes
(Added below as stages complete.)

### Stage 1: sanity gates G1-G3

**Data and checks** (`results/exp2_checks.json`).
- **`world.make_shape(rng, rotate=False)`** keeps the long axis along x and still draws the rotation angle.
  - The default path is bit-identical: one rebuilt chunk of each original dataset matches under `np.array_equal`.
  - Canonical bodies have the same area and perimeter as the originals, to 6.0e-09 and 4.8e-07. That is float32 storage of the coordinates; the requirement is 1e-6.
- **`data/exp2/downstream_canon.npz`** uses the same seeds and bodies as `data/downstream.npz`, with U_B reused (identical). The new flow angles are alpha_A and alpha_B ~ U(-30, +30) degrees.
  - They come from `np.random.default_rng(chunk seed + 50_000)`, drawing alpha_A then alpha_B for each body. This choice is not in the spec.
  - Labels are in the canonical absolute frame.
- **`data/exp2/pretrain_canon.npz`** is the pretraining corpus with `rotate=False`. `ckpt/exp2/Lc_s0.npz` is L pretrained on it (3000 steps, seed 0; final validation loss 0.00173, against 0.00177 for the original L).
- **Caveat: "canonical" means the stretch axis.** For nearly round bodies the radial harmonics can tilt the principal axis. Across the canonical downstream bodies, the principal-axis angle to +x has median 1.9 degrees and 90th percentile 9.5 degrees, but reaches 86 degrees for the roundest bodies.

**Runs.** `frame="absolute"` (per-sample prompt 0.25*U*(cos(alpha+shift), sin(alpha+shift)), U = 1 for A); n = 64; fine-tuning seeds 0 and 1; S, L, Lc; shift 0 vs `prompt_mode="zero"`; 24 runs in `results/exp2_finetune.jsonl`.
- **Statistics:** paired relative change over the 100 test bodies, with per-body errors averaged over the 2 fine-tuning seeds, and a 95% bootstrap CI (`*` = excludes 0).
- The CIs do not cover pretraining-seed variance (one L and one Lc checkpoint).
- **Mean-predictor baseline (choice not in the spec):** the mean label over all 256 TRAIN bodies, the same mean used for normalisation.

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


**Gate verdicts.**
- **G1 (learnability): passes, with a caveat on task A.**
  - Scratch beats the mean predictor on both tasks with CIs excluding 0: B 0.418 vs 0.899 (-53%), A 0.334 vs 0.415 (-19%).
  - The task-A margin is modest. Scratch is also much worse here than in the flow frame (0.334 vs 0.193 at n = 64), so the absolute frame with per-sample prompts is clearly harder for a randomly initialised model.
  - The pretrained models do not have this problem (L 0.159 on A, 0.194 on B).
- **G2 (the prompt must carry information): passes clearly.** Zeroing the prompt costs L +133% (A) and +159% (B), and Lc +120% and +177%; every CI excludes 0.
  - Scratch barely uses the prompt (+4.6% on A, +9.2% on B). It cannot learn to read a per-sample flow angle from 64 examples.
  - So the gain from pretraining here comes largely from the pretrained prompt pathway. That is the GeoPT mechanism, and it was absent in the constant-prompt flow frame.
- **G3 (L vs Lc at shift 0): mixed.** Lc is worse on A (+6.8% [+3.5, +10.2]) and better on B (-5.2% [-9.4, -0.8]). P4 (Lc >= L) holds on task B only.
- **Is a shift sweep worth running? Yes.**
  - The flow angle now strongly affects the labels, and the pretrained models depend heavily on the prompt.
  - Scratch's flat response to the prompt makes it a clean noise floor for the difference-in-differences analysis.

**Stage 2 estimate.** One run here takes about 30 s (A) to 62 s (B) of one core. Stage 2's 180 runs (S, L, Lc × 5 shifts × 2 tasks × n 16/64 × 3 seeds) need about 2.3 core-hours, which is about 40 minutes on 4 cores. The 24 gate runs cover 12 of them (shift 0, normal prompt, seeds 0 and 1, n = 64), so these can be reused.
