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
  - Canonical bodies have the same area and perimeter as the originals, to 6.0e-09 and 4.8e-07. The requirement is 1e-6. The area check runs in float64; the perimeter check ran on float32 vertices, and recomputed in float64 the perimeter difference is at most 2.9e-08.
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


**Gate verdicts** (revised after an independent audit; see the end of this section).
- **G1 (learnability): fails on task A, so per the spec Stage 1 stops here for diagnosis.**
  - Against the scalar training mean (the table above), scratch passes: A 0.334 vs 0.415, B 0.418 vs 0.899.
  - But in this canonical absolute frame the surface points and grid cells line up across bodies. So the per-position training mean is also a constant, input-ignoring prediction, and it is much stronger.
  - Task A: the per-surface-point mean profile scores 0.322, and scratch is worse than it: +5.7% [+1.8, +9.6], better on only 45% of bodies.
  - Task B: the per-cell mean field scores 0.504. Scratch beats it, but by -14.0% [-20.2, -7.4], not -53%.
  - Under the guard "scratch at n = 64 beats the mean predictor by a wide margin", task A fails and task B passes narrowly.
- **Diagnosis so far.**
  - Scratch is also 26% worse ([+20, +33]) than a prompt-blind geometry oracle: the average of the exact q over 25 angles spanning ±30°, which scores 0.280. So it fails to learn even the geometry-only map from 64 bodies.
  - Its final training loss is tiny (normalised MSE [0.0041, 0.0036] on A), which points to memorising the 64 training bodies. That is the failure mode of the abandoned absolute-frame run.
  - The pretrained models do not have this problem (L 0.159 on A, 0.194 on B, about half of scratch).
  - Not yet tested, all cheap: scratch at n = 256; more steps or weight decay; scratch at n = 16, which Stage 2 needs and which is likely at the mean predictor.
- **G2 (the prompt carries information): passes for L and Lc.**
  - Zeroing the prompt costs L +133% (A) and +159% (B), and Lc +120% and +177%.
  - For scratch the effect (+4.6% on A, +9.2% on B) has the same sign in both seeds, but it is no larger than scratch's own seed 0 vs seed 1 difference.
  - At the label level the premise holds: the prompt-blind oracle scores 0.280 on A, against 0 for the exact solution.
- **G3 (L vs Lc at shift 0): undetermined.** The pooled numbers (A +6.8%, B -5.2%) flip sign between fine-tuning seeds.
  - Task A: +19.4% with seed 0, -2.3% with seed 1.
  - Task B: -9.7% with seed 0, +3.7% with seed 1.
  - L's own seed-to-seed spread on B (0.209 vs 0.179) is twice the Lc-L gap. P4 is undetermined on both tasks.
- **Implications for Stage 2's design, if it goes ahead.**
  - Scratch barely reads the prompt. Its shift curve will be flat by construction, so the difference-in-differences against scratch reduces to each variant's own shift penalty.
  - Fine-tuning-seed noise is large: L on B differs by 26% between two seeds under this bootstrap.
  - So Stage 2 should judge shift penalties against seed-level variability, for example with a bootstrap over seeds as well as bodies, or each variant's seed-to-seed difference as the noise floor. Otherwise a 10-25% "penalty", including a spurious 180° penalty on task A, could be seed noise.
- **Statistics caveat.** The CIs resample the 100 test bodies only. They cover neither fine-tuning-seed variance (2 seeds, averaged) nor pretraining-seed variance (one L and one Lc checkpoint).
- **A correction to an earlier sentence.** The flow-frame scratch number (0.193) comes from a different dataset (randomly oriented bodies, full-circle flows), so the comparison with 0.334 does not isolate the frame or the prompt parameterisation.

**Stage 2 estimate.** One run here takes about 30 s (A) to 62 s (B) of one core. Stage 2's 180 runs (S, L, Lc × 5 shifts × 2 tasks × n 16/64 × 3 seeds) need about 2.3 core-hours, about 35 minutes on 4 idle cores. The 24 gate runs cover 12 of them (shift 0, normal prompt, seeds 0 and 1, n = 64), so these can be reused.

**Independent audit.** Four agents recomputed every table entry and prose number from the raw results; all table entries reproduce exactly. They also challenged the interpretation. This section was revised because of it:
- G1 on task A changed from "passes with a caveat" to "fails against the per-position mean baseline: stop and diagnose".
- G3 changed from "mixed" to "undetermined", since it flips with the fine-tuning seed.
- The claim that scratch is a clean noise floor was withdrawn.
- The seed caveats and the corrected perimeter explanation were added.
