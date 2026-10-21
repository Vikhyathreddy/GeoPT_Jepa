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

## Amendment (2026-10-14, before any n = 64 shift run): scratch control corrected; new pre-registered predictions
User decision (Goals and Validation, steps 1-2). Written and committed before any n = 64 shift run exists. At this point the only n = 64 runs are shift 0 and zero prompt; task B n = 16 (S, L, Lc, all shifts) and the scratch n = 256 control have been run.

### Step 1: the scratch reading, corrected
**Earlier claims replaced:**
- An earlier sentence above said scratch's shift curve "will be flat by construction" because scratch barely reads the prompt.
- An informal summary called scratch's task-A n = 256 pattern (a penalty at 90°, none at 180°) "the predicted physics pattern".

**The correct reason scratch must be flat at 90° and 180°: symmetry.**
- Rotating every prompt by 90° permutes and sign-flips the two prompt inputs of `q.1`; rotating by 180° flips their signs.
- The Gaussian initialisation is invariant under that change, and AdamW (with weight decay and global-norm clipping) is equivariant under permutations and sign flips of parameter coordinates.
- So over initialisation seeds, scratch's error has exactly the same distribution at 0°, 90° and 180°, whether or not it reads the prompt.
- At 20° and 45° the symmetry is only approximate, because Adam's per-coordinate normalisation is not rotation-equivariant.

**The check: differences between 3-seed means, against the pooled seed-to-seed SD** (SE of a 3-seed difference = SD × sqrt(2/3)):

| task, n | pooled seed SD | 0° (seeds) | 90° | 180° | z(90) | z(180) |
|---|---|---|---|---|---|---|
| A, 256 | 0.0223 | 0.196, 0.163, 0.201 | 0.217, 0.234, 0.199 | 0.179, 0.176, 0.225 | +1.64 | +0.36 |
| B, 256 | 0.0165 | 0.166, 0.166, 0.197 | 0.193, 0.161, 0.163 | 0.166, 0.179, 0.153 | -0.29 | -0.75 |
| B, 16 | 0.0276 | 0.458, 0.517, 0.495 | 0.462, 0.515, 0.489 | 0.463, 0.512, 0.488 | -0.08 | -0.12 |

**Reading.**
- All differences fall within scratch's own seed spread (|z| <= 1.64), so the control passes, with no sign of a bug.
- The task-A n = 256 "90° penalty" (+0.030) is seed noise and must not be read as physics.
- The n = 256 control does read the prompt (zeroing it costs A 0.187 → 0.360, B 0.176 → 0.470) and is still flat at 90° and 180°. Flatness therefore says nothing about prompt use.
- At n = 16, each scratch seed barely moves with the shift at all.

### Step 2: predictions and verdict rules for the n = 64 sweep, pre-registered now
**Scope.**
- L = `ckpt/L.npz` (pretraining seed 0) and L seed 1 = `ckpt/exp1/L_clean_s1.npz`, both pretrained on randomly oriented bodies.
- Lc = `ckpt/exp2/Lc_s0.npz`.
- Shifts 0, 20, 45, 90 and 180, plus zero prompt; n = 64; fine-tuning seeds 0-2; the canonical data and absolute frame of Stage 1.

**Statistics.**
- Delta_V(shift) = err(shift) - err(0), per test body, averaged over fine-tuning seeds.
- "Beats" / "excludes zero" refers to the paired 95% bootstrap interval over the 100 test bodies.
- SE_seed is the standard error of a 3-seed difference, from the variant's pooled seed SD at that n and task.

**P6, task A at 180° costs about nothing.**
- For L seed 0, L seed 1 and Lc at n = 64: |Delta_V(180)| <= 2 × SE_seed.
- A task-A 180° penalty with its paired CI excluding zero and larger than 2 × SE_seed counts as a failure, and triggers an investigation (bug or non-physical shortcut) before anything is reported.

**P7, new prediction: on task B the reversed prompt beats the aligned one.**
- At n = 64, Delta_L(180) < 0 with its paired CI excluding zero, separately for L seed 0 and for L seed 1. Both are required.
- **Provenance:** this effect was first seen at n = 16 (L 0.448 → 0.346, Lc 0.439 → 0.325, scratch flat), so task-B n = 16 is exploratory. Only n = 64 is a confirmatory test.

**Verdict rule for the question (from the Goals and Validation note).**
- **Support:** pretrained models lose more than scratch at 20-90° shifts. Operationally, for L at n = 64, the difference-in-differences Delta_L(shift) - Delta_S(shift) is positive with its CI excluding zero at one or more of 20, 45 and 90 degrees, on a task, in both L seeds.
- **Refutation:** no loss beyond scratch's at any shift at n = 16 or n = 64. Operationally, no positive difference-in-differences with a CI excluding zero, for L or Lc, at any shift, n or task.
- **Anything else is unresolved.**

### Stage 2 and Stage 3a results (2026-10-14): the shift sweep, L pretraining seed 1, q.1 frozen
**Decisions recorded (user, 2026-10-14).**
- **Task A passes G1, with a caveat.** The G1 diagnostic (`results/exp2_diag.json`) gives scratch at n = 64 a training rel-L2 of 0.031-0.032 against a test rel-L2 of 0.33-0.34 on task A.
  - That near-zero training error marks memorisation of the 64 training bodies (L reaches a similar training error, 0.038, but tests at 0.15-0.17).
  - Scratch still beats the scalar mean predictor (0.415), and the sweep measures the pretrained models, which behave well.
- **P4 is recorded as failed on task A and passed on task B** (shift 0, n = 64, fine-tuning seeds 0-2: L 0.1654 vs Lc 0.1670 on A; L 0.1856 vs Lc 0.1805 on B). Fine-tuning-seed spread is comparable to these gaps (see the noise floor table).

**Runs.**
- 330 rows in `results/exp2_finetune.jsonl` (all keys unique): S, L and Lc × shifts 0/20/45/90/180 × tasks A and B × n = 16 and 64 × fine-tuning seeds 0-2, plus zero-prompt references.
- Scratch at n = 256 as a control that uses the prompt.
- L pretraining seed 1 (`Ls1`) at n = 64.
- Stage 3a (`q.1` frozen; S, L, Lc; shifts 0, 45, 180; n = 64).
- The n = 64 shift runs were all made after the pre-registration commit (d10cbf6).

**Tables** (`results/exp2_stage2.md`, `.json`):

Shift penalty Delta_V(shift) = err(shift) - err(0), per body, averaged over fine-tune seeds 0-2; 95% bootstrap CI over the 100 test bodies (covers bodies only, not seeds).

| task | n | variant | err(0) | 20 | 45 | 90 | 180 | zero prompt |
|---|---|---|---|---|---|---|---|---|
| A | 16 | S | 0.3905 | -0.0003 [-0.0014, +0.0008] | +0.0006 [-0.0012, +0.0025] | -0.0001 [-0.0025, +0.0023] | +0.0020 [-0.0011, +0.0049] | +0.0021 [+0.0001, +0.0043]* |
| A | 16 | L | 0.2707 | +0.0187 [+0.0125, +0.0247]* | +0.0228 [+0.0162, +0.0296]* | +0.0163 [+0.0088, +0.0236]* | +0.0099 [+0.0016, +0.0181]* | +0.1235 [+0.1088, +0.1390]* |
| A | 16 | Lc | 0.2693 | +0.0168 [+0.0127, +0.0208]* | +0.0284 [+0.0223, +0.0342]* | +0.0312 [+0.0233, +0.0388]* | +0.0207 [+0.0124, +0.0288]* | +0.1218 [+0.1065, +0.1375]* |
| A | 64 | S | 0.3392 | +0.0052 [+0.0017, +0.0088]* | +0.0086 [+0.0039, +0.0131]* | +0.0027 [-0.0012, +0.0067] | +0.0014 [-0.0032, +0.0062] | +0.0198 [+0.0151, +0.0246]* |
| A | 64 | L | 0.1654 | +0.0015 [-0.0021, +0.0052] | -0.0154 [-0.0211, -0.0099]* | -0.0293 [-0.0350, -0.0241]* | -0.0024 [-0.0096, +0.0048] | +0.1926 [+0.1719, +0.2132]* |
| A | 64 | Ls1 | 0.1748 | +0.0000 [-0.0038, +0.0038] | -0.0006 [-0.0047, +0.0036] | -0.0188 [-0.0236, -0.0143]* | -0.0068 [-0.0135, +0.0001] | +0.1886 [+0.1689, +0.2080]* |
| A | 64 | Lc | 0.1670 | -0.0061 [-0.0100, -0.0024]* | -0.0179 [-0.0238, -0.0123]* | -0.0164 [-0.0228, -0.0102]* | +0.0029 [-0.0050, +0.0102] | +0.1872 [+0.1668, +0.2076]* |
| A | 256 | S | 0.1867 | - | - | +0.0298 [+0.0244, +0.0352]* | +0.0066 [+0.0023, +0.0109]* | +0.1732 [+0.1567, +0.1902]* |
| B | 16 | S | 0.4901 | +0.0009 [-0.0014, +0.0032] | +0.0007 [-0.0020, +0.0034] | -0.0017 [-0.0058, +0.0023] | -0.0027 [-0.0072, +0.0017] | +0.0055 [+0.0010, +0.0099]* |
| B | 16 | L | 0.4481 | +0.0016 [-0.0018, +0.0048] | -0.0067 [-0.0115, -0.0022]* | -0.0538 [-0.0613, -0.0464]* | -0.1020 [-0.1150, -0.0889]* | +0.0126 [+0.0069, +0.0183]* |
| B | 16 | Lc | 0.4391 | -0.0051 [-0.0083, -0.0019]* | -0.0142 [-0.0194, -0.0089]* | -0.0516 [-0.0599, -0.0431]* | -0.1139 [-0.1281, -0.0993]* | +0.0210 [+0.0142, +0.0275]* |
| B | 64 | S | 0.4296 | -0.0007 [-0.0060, +0.0050] | +0.0112 [+0.0054, +0.0170]* | -0.0002 [-0.0062, +0.0058] | -0.0028 [-0.0115, +0.0054] | +0.0224 [+0.0139, +0.0305]* |
| B | 64 | L | 0.1856 | +0.0012 [-0.0028, +0.0052] | -0.0075 [-0.0129, -0.0024]* | -0.0329 [-0.0407, -0.0251]* | -0.0202 [-0.0264, -0.0140]* | +0.2816 [+0.2556, +0.3082]* |
| B | 64 | Ls1 | 0.1842 | +0.0104 [+0.0079, +0.0128]* | +0.0081 [+0.0036, +0.0128]* | -0.0283 [-0.0347, -0.0219]* | -0.0349 [-0.0423, -0.0272]* | +0.2728 [+0.2487, +0.2979]* |
| B | 64 | Lc | 0.1805 | -0.0001 [-0.0028, +0.0025] | -0.0045 [-0.0087, -0.0006]* | -0.0209 [-0.0267, -0.0154]* | -0.0261 [-0.0321, -0.0198]* | +0.2825 [+0.2558, +0.3106]* |
| B | 256 | S | 0.1760 | - | - | -0.0039 [-0.0069, -0.0006]* | -0.0102 [-0.0138, -0.0064]* | +0.2944 [+0.2657, +0.3222]* |

Difference-in-differences Delta_V(shift) - Delta_S(shift) at the same n (95% CI over bodies)

| task | n | variant | 20 | 45 | 90 | 180 |
|---|---|---|---|---|---|---|
| A | 16 | L | +0.0190 [+0.0129, +0.0250]* | +0.0222 [+0.0156, +0.0289]* | +0.0164 [+0.0089, +0.0237]* | +0.0080 [-0.0008, +0.0164] |
| A | 16 | Ls1 | - | - | - | - |
| A | 16 | Lc | +0.0171 [+0.0128, +0.0213]* | +0.0278 [+0.0214, +0.0340]* | +0.0312 [+0.0230, +0.0392]* | +0.0188 [+0.0101, +0.0269]* |
| A | 64 | L | -0.0036 [-0.0083, +0.0010] | -0.0239 [-0.0314, -0.0167]* | -0.0320 [-0.0388, -0.0255]* | -0.0037 [-0.0120, +0.0041] |
| A | 64 | Ls1 | -0.0052 [-0.0105, -0.0001]* | -0.0091 [-0.0151, -0.0033]* | -0.0214 [-0.0277, -0.0155]* | -0.0081 [-0.0160, -0.0001]* |
| A | 64 | Lc | -0.0113 [-0.0161, -0.0065]* | -0.0265 [-0.0334, -0.0200]* | -0.0190 [-0.0267, -0.0116]* | +0.0015 [-0.0073, +0.0099] |
| B | 16 | L | +0.0007 [-0.0035, +0.0048] | -0.0073 [-0.0127, -0.0021]* | -0.0521 [-0.0598, -0.0443]* | -0.0993 [-0.1122, -0.0859]* |
| B | 16 | Ls1 | - | - | - | - |
| B | 16 | Lc | -0.0060 [-0.0098, -0.0025]* | -0.0149 [-0.0205, -0.0094]* | -0.0499 [-0.0585, -0.0413]* | -0.1111 [-0.1257, -0.0964]* |
| B | 64 | L | +0.0019 [-0.0052, +0.0085] | -0.0187 [-0.0266, -0.0112]* | -0.0327 [-0.0421, -0.0232]* | -0.0174 [-0.0270, -0.0075]* |
| B | 64 | Ls1 | +0.0110 [+0.0046, +0.0172]* | -0.0032 [-0.0108, +0.0043] | -0.0281 [-0.0362, -0.0203]* | -0.0320 [-0.0419, -0.0225]* |
| B | 64 | Lc | +0.0006 [-0.0059, +0.0065] | -0.0157 [-0.0231, -0.0084]* | -0.0208 [-0.0292, -0.0126]* | -0.0232 [-0.0328, -0.0128]* |

Fraction of the prompt's benefit lost, (err(shift) - err(0)) / (err(zero) - err(0)), from seed-averaged mean errors; 95% CI from a bootstrap over bodies of the ratio of means

| task | n | variant | 20 | 45 | 90 | 180 |
|---|---|---|---|---|---|---|
| A | 16 | L | +15.1% [+10.5, +19.5] | +18.5% [+13.4, +23.6] | +13.2% [+7.4, +19.4] | +8.0% [+0.9, +15.0] |
| A | 16 | Lc | +13.8% [+11.0, +16.6] | +23.3% [+19.6, +26.9] | +25.6% [+20.5, +30.5] | +17.0% [+10.6, +23.7] |
| A | 64 | L | +0.8% [-1.1, +2.6] | -8.0% [-11.5, -4.9] | -15.2% [-19.3, -11.8] | -1.2% [-5.0, +2.4] |
| A | 64 | Ls1 | +0.0% [-2.0, +2.0] | -0.3% [-2.5, +1.9] | -10.0% [-12.9, -7.3] | -3.6% [-7.4, -0.0] |
| A | 64 | Lc | -3.3% [-5.6, -1.1] | -9.6% [-13.4, -6.3] | -8.7% [-13.1, -5.1] | +1.5% [-2.7, +5.4] |
| B | 16 | L | +12.7% [-15.0, +45.5] | -52.8% [-116.4, -15.5] | -425.9% [-771.7, -273.0] | -807.0% [-1467.5, -540.1] |
| B | 16 | Lc | -24.2% [-49.4, -8.2] | -67.8% [-115.3, -37.5] | -245.9% [-370.1, -175.0] | -542.3% [-788.7, -408.5] |
| B | 64 | L | +0.4% [-1.0, +1.8] | -2.7% [-4.6, -0.8] | -11.7% [-14.5, -9.1] | -7.2% [-9.3, -5.2] |
| B | 64 | Ls1 | +3.8% [+2.9, +4.7] | +3.0% [+1.3, +4.6] | -10.4% [-12.6, -8.2] | -12.8% [-15.4, -10.2] |
| B | 64 | Lc | -0.0% [-1.0, +0.9] | -1.6% [-3.0, -0.2] | -7.4% [-9.4, -5.4] | -9.2% [-11.5, -7.0] |

Seed-level noise floor at shift 0: largest relative difference in test error between two fine-tune seeds of the same variant (mean errors)

| task | n | variant | errors by seed | max pairwise relative difference |
|---|---|---|---|---|
| A | 16 | S | 0.3703, 0.4049, 0.3964 | 9.4% |
| A | 16 | L | 0.2625, 0.2959, 0.2536 | 16.7% |
| A | 16 | Lc | 0.2771, 0.2674, 0.2633 | 5.2% |
| A | 64 | S | 0.3417, 0.3265, 0.3495 | 7.1% |
| A | 64 | L | 0.1525, 0.1658, 0.1778 | 16.6% |
| A | 64 | Ls1 | 0.1603, 0.1831, 0.1810 | 14.2% |
| A | 64 | Lc | 0.1759, 0.1564, 0.1686 | 12.5% |
| A | 256 | S | 0.1964, 0.1625, 0.2012 | 23.8% |
| B | 16 | S | 0.4581, 0.5169, 0.4953 | 12.8% |
| B | 16 | L | 0.4314, 0.4741, 0.4389 | 9.9% |
| B | 16 | Lc | 0.4199, 0.4793, 0.4182 | 14.6% |
| B | 64 | S | 0.4160, 0.4205, 0.4523 | 8.7% |
| B | 64 | L | 0.2090, 0.1791, 0.1687 | 23.9% |
| B | 64 | Ls1 | 0.2348, 0.1656, 0.1524 | 54.1% |
| B | 64 | Lc | 0.1766, 0.1798, 0.1851 | 4.8% |
| B | 256 | S | 0.1657, 0.1656, 0.1966 | 18.7% |

Stage 3a: q.1 frozen (n = 64); shift penalty err(shift) - err(0) within the frozen runs, and frozen vs unfrozen at the same shift

| task | variant | err(0), q.1 frozen | penalty 45 | penalty 180 | frozen vs free at 0 | at 180 |
|---|---|---|---|---|---|---|
| A | S | 0.3571 | +0.0034 [-0.0005, +0.0073] | -0.0017 [-0.0065, +0.0030] | +5.6% [+4.2, +7.1]* better on 18% of bodies | +4.7% [+3.5, +5.9]* better on 28% of bodies |
| A | L | 0.1772 | -0.0148 [-0.0203, -0.0093]* | -0.0072 [-0.0144, -0.0001]* | +8.3% [+6.4, +10.2]* better on 20% of bodies | +4.9% [+3.3, +6.5]* better on 29% of bodies |
| A | Lc | 0.1770 | -0.0128 [-0.0183, -0.0074]* | -0.0027 [-0.0102, +0.0044] | +7.0% [+5.4, +8.7]* better on 24% of bodies | +3.6% [+1.9, +5.3]* better on 37% of bodies |
| B | S | 0.4589 | -0.0030 [-0.0091, +0.0030] | -0.0018 [-0.0084, +0.0048] | +7.7% [+5.8, +9.7]* better on 21% of bodies | +7.5% [+5.6, +9.3]* better on 23% of bodies |
| B | L | 0.2749 | -0.0434 [-0.0527, -0.0349]* | -0.0841 [-0.0956, -0.0731]* | +50.8% [+45.6, +56.6]* better on 2% of bodies | +16.0% [+13.8, +18.2]* better on 7% of bodies |
| B | Lc | 0.2735 | -0.0495 [-0.0587, -0.0408]* | -0.0858 [-0.0983, -0.0748]* | +54.7% [+49.0, +61.0]* better on 2% of bodies | +22.5% [+20.2, +24.8]* better on 2% of bodies |

Pre-registered checks (amendment of 2026-10-14)

| check | variant | Delta(180) = err(180) - err(0) [95% CI over bodies] | 2 x SE_seed | outcome |
|---|---|---|---|---|
| P6 task A 180 ~ 0 | L | -0.0024 [-0.0096, +0.0048] | 0.0199 | holds |
| P6 task A 180 ~ 0 | Ls1 | -0.0068 [-0.0135, +0.0001] | 0.0227 | holds |
| P6 task A 180 ~ 0 | Lc | +0.0029 [-0.0050, +0.0102] | 0.0158 | holds |
| P7 task B reversed beats aligned | L | -0.0202 [-0.0264, -0.0140] | 0.0238 | holds |
| P7 task B reversed beats aligned | Ls1 | -0.0349 [-0.0423, -0.0272] | 0.0602 | holds |

Difference-in-differences Delta_V(shift) - Delta_S(shift) used by the verdict rule (95% CI over bodies)

| task | n | variant | 20 | 45 | 90 | 180 |
|---|---|---|---|---|---|---|
| A | 16 | L | +0.0190 [+0.0129, +0.0250]* | +0.0222 [+0.0156, +0.0289]* | +0.0164 [+0.0089, +0.0237]* | +0.0080 [-0.0008, +0.0164] |
| A | 16 | Lc | +0.0171 [+0.0128, +0.0213]* | +0.0278 [+0.0214, +0.0340]* | +0.0312 [+0.0230, +0.0392]* | +0.0188 [+0.0101, +0.0269]* |
| A | 64 | L | -0.0036 [-0.0083, +0.0010] | -0.0239 [-0.0314, -0.0167]* | -0.0320 [-0.0388, -0.0255]* | -0.0037 [-0.0120, +0.0041] |
| A | 64 | Ls1 | -0.0052 [-0.0105, -0.0001]* | -0.0091 [-0.0151, -0.0033]* | -0.0214 [-0.0277, -0.0155]* | -0.0081 [-0.0160, -0.0001]* |
| A | 64 | Lc | -0.0113 [-0.0161, -0.0065]* | -0.0265 [-0.0334, -0.0200]* | -0.0190 [-0.0267, -0.0116]* | +0.0015 [-0.0073, +0.0099] |
| B | 16 | L | +0.0007 [-0.0035, +0.0048] | -0.0073 [-0.0127, -0.0021]* | -0.0521 [-0.0598, -0.0443]* | -0.0993 [-0.1122, -0.0859]* |
| B | 16 | Lc | -0.0060 [-0.0098, -0.0025]* | -0.0149 [-0.0205, -0.0094]* | -0.0499 [-0.0585, -0.0413]* | -0.1111 [-0.1257, -0.0964]* |
| B | 64 | L | +0.0019 [-0.0052, +0.0085] | -0.0187 [-0.0266, -0.0112]* | -0.0327 [-0.0421, -0.0232]* | -0.0174 [-0.0270, -0.0075]* |
| B | 64 | Ls1 | +0.0110 [+0.0046, +0.0172]* | -0.0032 [-0.0108, +0.0043] | -0.0281 [-0.0362, -0.0203]* | -0.0320 [-0.0419, -0.0225]* |
| B | 64 | Lc | +0.0006 [-0.0059, +0.0065] | -0.0157 [-0.0231, -0.0084]* | -0.0208 [-0.0292, -0.0126]* | -0.0232 [-0.0328, -0.0128]* |

P6 (task A 180 costs about nothing, L s0, L s1, Lc): holds. P7 (task B reversed beats aligned at n = 64 in both L seeds): holds. Verdict by the pre-registered rule: unresolved.

**Figure:** `figs/exp2_shift_curves.png`.

**Pre-registered checks.**
- **P6 (task A, 180° costs about nothing at n = 64): holds** for L seed 0 (-0.0024 [-0.0096, +0.0048]), L seed 1 (-0.0068) and Lc (+0.0029). All are well within 2 × SE_seed.
- **P7 (task B, reversed beats aligned at n = 64 in both L seeds): holds by the pre-registered rule.** L seed 0: -0.0202 [-0.0264, -0.0140]; L seed 1: -0.0349 [-0.0423, -0.0272].
  - **Caveat (corrected after the audit; it applies to both L seeds):** those intervals cover test bodies, not fine-tuning seeds. Neither seed's effect clears 2 × SE_seed (L seed 0: |-0.0202| < 0.0238; L seed 1: |-0.0349| < 0.060).
  - Pairing fine-tuning seeds, L seed 0 is negative in all three (-0.027, -0.015, -0.019; t95 [-0.036, -0.005]), while L seed 1 (-0.083, -0.018, -0.003; t95 [-0.141, +0.071]) and Lc (t95 [-0.058, +0.006]) are not robust.
  - L seed 1's magnitude comes mostly from one poor shift-0 run (its errors span 0.235, 0.166 and 0.152).
  - Lc shows the same direction (-0.0232 difference-in-differences), and it holds in every task-B cell at n = 16.
- **Verdict on the question (pre-registered rule): unresolved.**
  - **Support fails.** At n = 64, no shift makes L lose more than scratch in both L seeds. The difference-in-differences is negative (shifting helps) at 45° and 90° on both tasks, in both seeds. The only positive cell is L seed 1 on task B at 20° (+0.011), and it does not replicate in seed 0.
  - **Refutation fails.** At n = 16 on task A, L and Lc do lose more than scratch at 20-90° (+0.016 to +0.031, CIs excluding 0). That is 13-26% of the prompt's benefit lost.

**Reading.**
- **An alignment penalty appears only in one corner: small data (n = 16) on task A** (13-26% of the prompt's benefit lost). Corrected after the audit, three qualifications apply:
  - This is outside the pre-registered confirmatory test, which covers n = 64.
  - For L it is not robust across fine-tuning seeds (every 3-seed paired t interval includes 0).
  - Lc's 180° difference-in-differences there (+0.019) is as large and as seed-consistent as its 20° one (+0.017). The same checkpoint-level asymmetry may produce both, which weakens reading it as GeoPT's physical alignment effect.
- **With 64 labels, only the 90° shift helps consistently** (corrected after the audit; the original line said "misalignment never hurts, and 45-90° shifts help"). At 90°: task A L -15%, L seed 1 -10%, Lc -9% of the prompt's benefit; task B -12%, -10%, -7%.
  - L seed 1 on task B is hurt at 20° (+0.0104 [+0.0079, +0.0128]) and 45° (+0.0081 [+0.0036, +0.0128]), on body-only CIs and inconsistently across fine-tuning seeds.
  - At 45° L seed 1 gains nothing on task A either.
  - Scratch has small penalties at 20° and 45° (A +0.005 and +0.009, B 45° +0.011). These push the pretrained models' difference-in-differences down, so some "helps beyond scratch" cells reflect scratch getting worse.
  - **Task A, exploratory (post hoc) analysis:** the 90° gain grows with body elongation for all three pretrained models (Spearman of aspect ratio vs Delta(90): -0.32, -0.26 and -0.32; p <= 0.009). The rounder half of the bodies gains -0.007 to -0.021 and the more elongated half -0.025 to -0.038; scratch shows no such link.
  - This fits a non-physical mechanism. The canonical bodies have their long axis along x, so a 90° prompt points at the broad side, where pretraining's transport features are most geometry-rich.
  - On task B there is no elongation link, so the B gain has another source.
- **On the plume task the pretrained models prefer the reversed prompt** (P3: the best prompt is reversed). That holds at n = 16 (L 0.448 → 0.346, Lc 0.439 → 0.325) and at n = 64 (both L seeds, and Lc). The best prompt direction at n = 64 is 90° for L seed 0 and 180° for L seed 1 and Lc.
- **Stage 3a, q.1 frozen.** No shift penalty appears even when the prompt-reading layer cannot re-map.
  - On task B, shifting helps a lot (L: 45° -0.043, 180° -0.084). Freezing q.1 at the aligned prompt costs +51% (L) and +55% (Lc).
  - So the pretrained prompt pathway's own semantics prefer a rotated or reversed prompt on the plume task. The spec's "penalty only when frozen" pattern does not occur.
- **Investigation of the n = 16 task-A 180° penalty for Lc** (+0.019 [+0.010, +0.027] beyond scratch). The spec's P2 asks for an investigation before such a penalty is reported.
  - It is consistent across fine-tuning seeds (+0.019, +0.024, +0.019), so it is not seed noise.
  - It is not a code bug. Scratch is flat at 180° at n = 16, 64 and 256 (per-seed differences within ±0.024). Lc is flat at 180° at n = 64 (P6), and L's n = 16 value is inconsistent across seeds (+0.003, -0.022, +0.049).
  - Mechanism (corrected after the audit; the original suggested that the pretext is not symmetric under v → -v).
    - The 180°-shifted problem is physically identical, since q(alpha) = q(alpha + pi) to 4e-15, so no invariance has to be learned.
    - The pretext is symmetric under a joint 180° rotation of body and velocity, and the canonical body distribution is 180°-symmetric.
    - A penalty therefore needs a network that is not 180°-equivariant: absolute Fourier(x) features and a single finite pretraining run.
    - That points to a checkpoint-level (pretraining-seed) effect, which neither the body CI nor the fine-tuning seeds cover.
  - That is a non-physical asymmetry inherited from pretraining. It is not a test of the physics, and it vanishes at n = 64.
  - The right test (not run) is a second Lc pretraining seed. Random prompt sign flips during fine-tuning would not separate the two explanations.

**Predictions: outcomes.**
- **P1 (penalty growing 0 → 90° for L and Lc, larger at n = 16; scratch none): mostly failed.** A penalty appears only on task A at n = 16. It grows toward 90° only for Lc (L: +0.019, +0.023, +0.016) and is not fine-tuning-seed robust for L.
  - At n = 64 there is no consistent penalty, and 90° helps.
  - Scratch has small 20°/45° penalties at n = 64, so its clause holds only at the exactly symmetric 90° and 180° (corrected after the audit).
- **P2 (task A 180° costs nothing for every variant): held at n = 64,** pre-registered as P6, with the pre-registered criterion (|Delta| <= 2 × SE_seed).
  - At n = 16, Lc's +0.021 penalty is seed-consistent but inside 2 × SE_seed (0.026), so under the P6 criterion it would not count as a failure.
  - L at n = 16 has a body-CI-significant +0.010 that is seed-inconsistent; scratch at n = 256 has +0.007, within seed noise. (Criteria made explicit after the audit.)
- **P3 (task B sign of the 180° penalty, open): the reversed prompt is better** for L and Lc at n = 16 (one L pretraining seed) and at n = 64 (both L seeds, on body CIs). Across fine-tuning seeds this is robust only for L seed 0 (corrected after the audit; L seed 1 was run at n = 64 only).
- **P4 (Lc >= L at shift 0): failed on task A, passed on task B** (recorded per the user's decision; see above).
- **P5 (no penalty → GeoPT's effect depends on something the toy lacks): partly applies.** The toy shows a penalty only with 16 labels on task A. With 64 labels the pretrained prompt semantics are re-mapped (or even exploited), and the q.1-frozen runs show that the remaining prompt semantics prefer non-aligned prompts.
- **P6: held. P7: held by the pre-registered rule.** Neither L seed's effect clears 2 × SE_seed; across fine-tuning seeds it is robust for L seed 0 only.

**Caveats.**
- The CIs cover the 100 test bodies only.
- Fine-tuning-seed spread at n = 64 reaches 17-24% for L and 54% for L seed 1 on task B (noise floor table), comparable to many of the shift effects.
- There is one Lc checkpoint, and two L pretraining seeds (seed 1 at n = 64 only).
- Scratch at n = 64 memorises (the G1 caveat).


## Independent audit (step 5, 2026-10-14)
Three agents recomputed every number, checked the code and challenged the claims for the Stage 2/3a sweep and the pre-registered checks. Every number reproduces.
- **No bug:** shift units, the per-sample prompt, the `Ls1` checkpoint, `q.1` freezing, zero-prompt mode and the verdict code all check out.
- **Pre-registration holds:** every n = 64 shift row postdates the pre-registration commit (d10cbf6).
- **The verdict stays "unresolved".** The corrections are interpretive and are applied in place above, marked "corrected after the audit".

**Further points from the audit.**
- **P4 (recorded per the user as failed on task A and passed on task B): neither gap is significant.** With 3 fine-tuning seeds, the paired Lc vs L difference is A +2.3% [-0.3, +4.9] and B -1.3% [-4.6, +2.0], and the per-seed sign flips on both tasks. The record therefore reflects the user's decision, not a significant result.
- **Stage 3a freezes only `q.1`.** `q.2` and the decoder blocks still train, so the frozen runs show that a model with a fixed first prompt layer still prefers rotated or reversed prompts on task B. They do not isolate the pretrained semantics.
- **G1 caveat, with all 3 seeds:** scratch on task A is still worse than the per-surface-point mean profile (+7.3% [+3.6, +11.1]). The task-A scratch control is valid through the symmetry argument, not through its fit.
- **Statistics:** intervals cover the 100 test bodies only. Pretraining seeds: L 2 (seed 1 at n = 64 only), Lc 1.

## Stage 3a in one place: what freezing the prompt layer (`q.1`) showed
**Setup:** n = 64; S, L and Lc; shifts 0°, 45° and 180°; fine-tuning seeds 0-2; only `q.1` frozen at its pretrained (or, for S, random) values. Table in `results/exp2_stage2.md`.

**Findings.**
1. **Freezing `q.1` costs accuracy everywhere, at the aligned prompt.**
   - Task A: S +5.6%, L +8.3%, Lc +7.0%.
   - Task B: L +51% (0.275 vs 0.186), Lc +55%.
   - So fine-tuning normally re-maps the pretrained prompt layer substantially, especially for the plume task.
2. **Even with `q.1` frozen, misalignment never hurts. It helps.**
   - Task B: L -0.043 at 45° and -0.084 at 180°; Lc -0.050 and -0.086.
   - Task A: L -0.015 at 45° and -0.007 at 180°; Lc -0.013 at 45° and ~0 at 180°.
   - Scratch is flat (all CIs include 0).
3. **The spec's diagnostic pattern ("a penalty only when `q.1` is frozen") does not occur.** There is no alignment penalty for fine-tuning to absorb.
   - Instead, with the first prompt layer fixed, a reversed prompt on task B recovers much of what freezing lost: L goes from 0.275 to 0.191, close to the unfrozen aligned 0.186.

**Caveat (from the audit):** `q.2` and the decoder blocks still train. So these runs show only that a model with a fixed first prompt layer prefers a rotated or reversed prompt; they do not isolate the pretrained prompt semantics. CIs cover the 100 test bodies only; one checkpoint each for L and Lc.
