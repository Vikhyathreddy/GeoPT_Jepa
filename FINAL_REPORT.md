# GeoPT × JEPA toy world: final report on the three follow-up questions
2026-10-21 · branch `claude/followup-experiments`. Details: `EXP1_RESULTS.md`, `EXP2_RESULTS.md`, `EXP3_RESULTS.md`, `SEED_REPLICATION.md`, `results/seed_replication.md`.

**How to read the labels.**
- **Robust:** the claim holds in at least 4 of 5 pretraining seeds, and the across-seed 95% t interval excludes zero (rule committed before the seed runs).
- **Single-seed:** the rule is not met.
- **Contradicted:** the interval excludes zero the other way.
- **Coverage:** across-seed intervals cover pretraining seeds, each averaged over its fine-tuning seeds, on the fixed 100 test bodies. Single-seed body intervals cover test bodies only.
- **Scope:** this is a 2D toy with 147k-parameter models. The verdicts are design hypotheses, not evidence at scale.

## Verdicts

| Question | Verdict | Basis |
|---|---|---|
| 1. Do latent (JEPA) targets beat explicit (GeoPT) ones under noise? | **Closed: not testable with this latent design.** Its latent part doesn't reliably add dynamics information over its own explicit anchor. | Pre-registered branch (2026-10-21). Clean LT beating N is single-seed: task A 3/5 seeds, -1.0% [-10.3, +8.3]; task B 4/5, -2.6% [-5.8, +0.7]. So no stronger noise was run. |
| 2. Does a misaligned prompt hurt once the prompt carries the flow angle? | **Pre-registered rule: unresolved. Across seeds, at 90° the answer is no: it helps.** | Step 3 rule: unresolved. Across five L seeds, "L loses more than scratch at 90°" is contradicted at n = 64 (A, B) and at n = 16 (B). At A n = 16 it is single-seed (3/5). The n = 16 task-A penalty at 20°/45° exists for pretraining seed 0 only. |
| 3. Do latent-transport features rank design options better? | **Refuted.** L ranks as well as or better than LT everywhere tested. | No seed shows LT robustly ahead. Decoder probe, task B: L > LT robust but tiny (+0.010 [+0.0002, +0.020]); task A: tie (single-seed). R1/R3: pretraining seed 0 only. |

## Claims by robustness

**Robust (5 pretraining seeds).**
- **Clean L beats clean LT:** A +17.5% [+3.3, +31.8], B +10.5% [+6.1, +14.9]. Clean L beats N: A +19.9%, B +14.3%.
- **Label noise at sigma = 2 degrades L:** A +10.6%, B +8.6%.
- **At sigma = 2, L still beats the clean LT on task B:** +2.5% [+1.1, +3.8].
- **Removing the prompt hurts L,** at n = 16 and 64 on both tasks.
- **The reversed prompt beats the aligned one on task B,** at n = 64 (-0.023, pre-registered as P7, 5/5) and at n = 16 (-0.105, 5/5). The n = 16 effect was first seen on seed 0, so seeds 1-4 are its replication.
- **Task A at 180° costs about nothing** (null check holds at both n).
- **A 90° shift helps L** (its own effect): n = 64 on A (-0.030) and B (-0.023), and n = 16 on B (-0.045). *This claim was added after the seed-0/1 results and was not pre-registered; treat it as exploratory, though robust.*
- **Decoder probe, task B, L ranks better than LT:** +0.010 (lower bound +0.0002, fragile).

**Contradicted.**
- **"L loses more than scratch at 90°"** (difference-in-differences): n = 64 A and B, n = 16 B. Scratch's 90° effect is about 0 and shared across L seeds, so this mirrors the 90° gain above.

**Single-seed (not replicated, or not replicable by design).**
- **At sigma = 2, L beats LT on task A** (4/5 seeds, interval includes 0).
- **At sigma = 4, L beats LT on task A:** 1/5 seeds. LT is nominally better in 4 of 5. Task B was not pre-registered; it goes the other way in all 5 seeds.
- **Clean LT beats N** (A 3/5, B 4/5): the deciding claim for question 1.
- **The task-A n = 16 penalty at 90°** (3/5). The 20°/45° penalties at n = 16, and the Lc 180° penalty at n = 16, exist for seed 0 only.
- **Decoder probe, task A** (tie). **R1 and R3 ranking:** seed 0 only, by design.
- **Everything else from Experiments 1-3 that was not in the seed table:**
  - the noise curve beyond sigma 2, trajectory noise, LJ, fresh vs frozen noise;
  - Lc, Stage 3a, the n = 256 scratch control;
  - R3, and P3/P4 of Experiment 3.

## Predictions: held or failed
**Experiment 1.**
- P1 failed: label noise hurt L by more than 5%.
- P2 held as an interpretation.
- P3 undetermined.
- P4 not run.
- P5 failed on task A.
- Amendment prediction (hit vs free particles): not supported proportionally.

**Experiment 2.**
- P1 mostly failed: a penalty appears only at n = 16 on task A, for seed 0.
- P2 held at n = 64 and robustly across seeds at both n. Lc at n = 16 is the seed-0 exception.
- P3 (task B prefers the reversed prompt): held, robust.
- P4: recorded as failed on A and passed on B per the user's decision; neither gap is significant.
- P5 partly applies.
- P6 held, robust.
- P7 held, robust.

**Experiment 3.**
- P1 (null) held.
- P2 failed.
- P3 held (outcome only; cause not identified).
- P4 held on task A; on task B only for scratch.

**Seed rule.** E1-c' failed to reach robustness, so branch 2 was taken.

## What was learned
1. **GeoPT's explicit lifted targets beat latent transport and native geometry robustly.** Combining the two buys nothing.
2. **This LT's latent part does not reliably add dynamics information beyond its explicit anchor.** So the "latent targets win under noise" rule of thumb cannot be tested with it. A latent design that robustly beats N would be needed first.
3. **When the prompt carries the flow angle, it matters a lot** (removing it more than doubles the error at n = 64). Misalignment doesn't behave like GeoPT's report:
   - 90° helps, robustly, though this was exploratory;
   - on the plume task the reversed prompt helps, robustly;
   - a penalty shows up only at small n for one pretraining seed.
4. **Ranking tracks accuracy:** L ≥ LT for planning. Training on multiple actions per body improves planning a lot, though that is confounded with having more simulations.

## Integrity notes
- **Pre-registration:** every prediction and rule was committed before its data; amendments are dated, and failed predictions stay in the record.
- **Audits:** independent audits after every stage reproduced all numbers; no code bugs were found. Interpretive corrections are marked in place.
- **The step-4 audit flagged** the E2-g pre-registration gap, the narrow E1-c' miss on task B, E1-f on task B, and the fragile E3-a on task B. All are reflected above.
- **Not done (optional):**
  - the equal-budget test (step 6);
  - a second Lc pretraining seed;
  - Experiment 3 Stage 2 (R2);
  - bounce and stronger trajectory noise. These are moot under branch 2.
