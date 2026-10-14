# Experiment 2: Preferred orientation. When does prompt alignment matter?

## Question
GeoPT (arXiv 2602.20399, Fig. 13) reports that rotating the dynamics prompt away from the true
flow direction hurts fine-tuning. The original toy study found no such effect: 0, 90 and 180 degrees
all scored the same, also with the prompt-reading layer `q.1` frozen, and even with no prompt at
all (`results/followup_*.jsonl`).

Hypothesis: in the toy study **the prompt was a constant**. The flow-aligned frame puts the flow
along +x for every sample, so the body-to-flow angle is carried by the rotated geometry, not by the
prompt. A constant input carries no information, and fine-tuning can re-map it freely. In GeoPT's
setting, bodies have a canonical orientation (cars face -x) and **the prompt varies per sample and
carries the flow angle**. This experiment switches to that parameterisation and repeats the shift
sweep.

## New data
- `world.make_shape(rng, rotate=True)`: with `rotate=False` the body keeps its long axis along x.
  Still draw the rotation angle, so the RNG stream (and therefore every body) matches the original;
  just don't apply it.
  - The default must stay bit-identical: rebuild one chunk with defaults and compare it with
    `np.array_equal`.
  - Test: canonical body i and original body i have the same area and perimeter (to 1e-6).
- `data/exp2/downstream_canon.npz`: the same seeds and bodies as `data/downstream.npz`, but
  canonical.
  - Keep consuming the original RNG draws in the same order, so the bodies match, and reuse the
    original U_B draw.
  - Draw the new flow angles from a separate RNG: alpha_A, alpha_B ~ U(-30 deg, +30 deg).
  - Labels are in the canonical (absolute) frame: q from `panel_solve(V, alpha_A)` and the plume from
    `plume_solve(V, alpha_B, U_B)`. Same keys as `downstream.npz`.
- `data/exp2/pretrain_canon.npz`: identical to `data/pretrain.npz` (same seeds) but with
  `rotate=False`.

## Fine-tuning setup
Use `frame="absolute"` from the setup step: no rotation of geometry or queries, and a per-sample
prompt `0.25*U*(cos(alpha + shift), sin(alpha + shift))` with U = 1 for task A. The prompt now
carries the flow angle, as in GeoPT. `shift` is the deliberate misalignment.

## Variants
- **S**: scratch.
- **L**: the original `ckpt/L.npz`, pretrained on randomly oriented bodies.
- **Lc**: L pretrained on `pretrain_canon.npz` (3000 steps, seed 0), saved to
  `ckpt/exp2/Lc_s0.npz`.
- Optional: LT and LTc.

## Stage 1: sanity gates (stop and report after these)
- **G1. Learnability.** S at n = 64 must have test rel-L2 well below the mean-predictor baseline
  (a constant prediction equal to the training mean) on both tasks. Report both numbers. The
  absolute frame failed before with random body orientations; canonical bodies with +-30 degree
  flows should be fine. If not, stop and diagnose.
- **G2. The prompt must carry information.** At n = 64 with 2 seeds, compare shift = 0 against
  `prompt_mode="zero"` for S, L and Lc. The zero-prompt error must be clearly higher (CI excluding
  0). Otherwise the flow angle barely affects the labels and shift effects can't be measured; report
  that and stop.
- **G3.** L vs Lc at shift 0: does pretraining on canonical orientations help in a canonical
  downstream task?

## Stage 2: shift sweep (about 2 core-hours)
Variants {S, L, Lc} x shift {0, 20, 45, 90, 180} degrees x task {A, B} x n {16, 64} x fine-tune
seeds {0, 1, 2} = 180 runs, written to `results/exp2_finetune.jsonl`.

## Stage 3 (optional)
- a. Freeze `q.1` (the layer that reads the prompt; `freeze=("q.1",)`) for shifts {0, 45, 180}. If
  a penalty appears only when `q.1` is frozen, fine-tuning normally absorbs the misalignment in that
  first layer.
- b. Widen the flow range to +-90 degrees.
- c. LT and LTc.

## Analysis
- Shift penalty of a variant: Delta_V(shift) = err(shift) - err(0), per test body, averaged over
  seeds. Use the difference-in-differences against scratch, Delta_V(shift) - Delta_S(shift), with
  paired bootstrap CIs over the 100 test bodies. Scratch has no pretrained prompt semantics, so its
  shift curve is the noise floor.
- `figs/exp2_shift_curves.png`: in the style of `figs/fig2_prompt_shift.png`. x = shift, y = test
  error, panels per task x n, lines per variant.
- Physics signature: compare penalties at 90 vs 180 degrees for each task.

## Predictions (pre-register by copying into EXP2_RESULTS.md before running)
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

## Deliverables
`EXP2_RESULTS.md` (predictions, method, gate results, tables, figure, outcome per prediction,
caveats), `results/exp2_*.jsonl`, `figs/exp2_*.png`, and a commit per stage.
