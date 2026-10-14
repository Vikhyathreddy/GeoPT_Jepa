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
