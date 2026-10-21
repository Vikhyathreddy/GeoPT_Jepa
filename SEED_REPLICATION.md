# Pretraining-seed replication (step 4): pre-registered rule
Committed on 2026-10-14, before any pretraining run at seeds 2-4 starts. Source: the "Goals and Validation" note, step 4.

## Runs
**Pretraining** (3000 steps, original hyper-parameters, on the randomly oriented corpus unless noted). Checkpoints go to `ckpt/exp1/`.
- **N, L and LT, clean, at seeds 2, 3 and 4.** Seed 0 is the original `ckpt/{N,L,LT}.npz`; seed 1 is `ckpt/exp1/{N,L,LT}_clean_s1.npz`.
- **L on label noise:** lblfrz2 at seeds 2-4 (seeds 0-1 exist), and lblfrz4 at seeds 1-4 (seed 0 exists). The Experiment 1 verdict is about the noise curve, so it can only hold across seeds if the noisy L is replicated too. LT under label noise is the same seed's clean LT, by construction.

**Downstream evaluations per pretraining seed.**
- **Experiment 1:** n = 64, tasks A and B, fine-tuning seeds 0 and 1, flow-aligned frame. Rows go to `results/exp1_finetune.jsonl`.
- **Experiment 2:** L at n = 64, shifts 0, 90 and 180 plus zero prompt, tasks A and B, fine-tuning seeds 0-2, canonical data and absolute frame. Rows go to `results/exp2_finetune.jsonl`. L seed 1 also gets 20° and 45° (step 3).
- **Experiment 3:**
  - linear probe (decoder features) for N, L and LT at every seed;
  - R1 for L and LT, tasks A and B, fine-tuning seed 0 only, at every seed (seed 0 uses its existing fine-tuning-seed-0 run).
  - Rows go to `results/exp3_probe.jsonl` and `results/exp3_runs.jsonl`. R3 stays single-seed.

## Statistic per seed
For each claim and pretraining seed k:
- d_k = the mean over the 100 test bodies of the paired per-body difference (or relative change, as the claim states).
- Per-body values are first averaged over that seed's fine-tuning seeds.

## Rule
- **Robust:** a claim is robust if its direction holds in at least 4 of the 5 pretraining seeds (k = 0-4), AND the across-seed 95% interval excludes zero on the claimed side.
  - The across-seed interval is mean(d) ± t(0.975, df = 4) × sd(d) / sqrt(5), with t = 2.776.
- **Single-seed:** otherwise the claim is reported as single-seed (true for the seed it was first measured on).
- **Contradicted:** if the across-seed interval excludes zero on the opposite side.
- **Null-type checks** (Experiment 2's task-A 180° check) are reported with the same across-seed interval. They hold across seeds if that interval contains zero, or lies within ±2 × the pooled SE_seed.

## Claims
| id | claim (direction) | statistic d_k |
|---|---|---|
| E1-a A, B | clean L beats clean LT | relative change err(LT) / err(L) - 1 > 0 |
| E1-b A, B | clean L beats N | err(N) / err(L) - 1 > 0 |
| E1-c A, B | LT vs N (the "LT ≈ N" premise) | err(LT) / err(N) - 1; reported, not directional |
| E1-d A, B | label sigma = 2 degrades L | err(L, sigma 2) / err(L clean) - 1 > 0 |
| E1-e A, B | at sigma = 2, L still beats the clean LT | err(LT) / err(L, sigma 2) - 1 > 0 |
| E1-f A | at sigma = 4, L still beats the clean LT | err(LT) / err(L, sigma 4) - 1 > 0 |
| E2-a A, B | zeroing the prompt hurts L (n = 64) | err(zero) - err(0) > 0 |
| E2-b B | reversed prompt beats aligned for L (n = 64) | err(180) - err(0) < 0 |
| E2-c A | 180° costs about nothing for L (n = 64) | err(180) - err(0); null-type |
| E2-d A, B | L loses more than scratch at 90° (n = 64) | (err_L(90) - err_L(0)) - (err_S(90) - err_S(0)) > 0 |
| E3-a A, B | L ranks better than LT, decoder probe | rho(L) - rho(LT) > 0 |
| E3-b A, B | L ranks better than LT, R1 | rho(L) - rho(LT) > 0 |
| E3-c A, B | LT ranks better than L anywhere (support for Q3) | the opposite sign of E3-a/b |

Scratch (Experiment 2 difference-in-differences) has no pretraining seed: the same scratch runs (fine-tuning seeds 0-2) serve every L seed.

## Amendment (2026-10-21, before any step-4 run)
User decision. The runs and rule above are otherwise unchanged.

### Scope changes
- **Experiment 2:** L at 0°, 90°, 180° and zero prompt, on tasks A and B, at **both n = 16 and n = 64**, fine-tuning seeds 0-2. This applies to pretraining seeds 1-4; seed 1 previously had n = 64 only, so its n = 16 runs are added. The reason: the task-A penalty and the large task-B reversed-prompt gain appeared only at n = 16. Claims E2-a to E2-d are evaluated at each n separately.
- **Experiment 3:** the R1 runs for new seeds are dropped. The decoder probes for N, L and LT at seeds 1-4 stay. E3-b (R1) is reported as **single-seed** (pretraining seed 0 only).

### Conditional plan for question 1, fixed now
**The deciding claim, reported first in `results/seed_replication.md`:**
- **E1-c': clean LT beats clean N.**
- d_k = err(LT_k) / err(N_k) - 1, claimed direction < 0, on each task. Same fine-tuning protocol as Experiment 1 (n = 64, fine-tuning seeds 0-1).
- "LT beats N robustly" means **robust under the seed rule on both tasks A and B**. Both tasks are required because the Goals-and-Validation support rule for question 1 requires both tasks.

**Branch 1: LT beats N robustly. Then the latent part does add dynamics information, and the fair test is run:**
- Trajectory-noise corpora pos060 and pos100 (sigma_pos = 0.6 and 1.0; built as pos015/pos030, with seed 124's draw scaled).
- L and LT pretrained on each, at all five pretraining seeds. Fine-tuning as in Experiment 1.
- Judged by the amended rule:
  - **Support:** LT beats L on both tasks, CI excluding 0, robust under the seed rule.
  - **Refutation:** L >= LT at every trajectory level up to the level where L's advantage over N (same seed) has halved, robust across seeds.
  - **Otherwise unresolved.**

**Branch 2: otherwise (not robust on both tasks, single-seed or contradicted). Question 1 is closed:**
- Verdict: "not testable with this latent design: its latent part doesn't reliably add dynamics information over its own explicit anchor".
- No stronger noise is run.

## Outcome (2026-10-21)
- **Results:** `results/seed_replication.md`.
- **E1-c' (clean LT beats N) is single-seed on both tasks** (A 3/5, B 4/5; both intervals include 0). So **branch 2** applies: question 1 is closed as not testable with this latent design, and no stronger noise is run.
- **Audit (subagent, read-only):** all values and verdicts reproduce, the pre-registration order is confirmed, and the checkpoints are correct. Notes:
  - E2-g ("90° helps L") was added after the seed-0/1 results and is not in the pre-registered claims table, so it is exploratory.
  - E2-d's scratch control is shared across L seeds.
  - E1-f on task B (not pre-registered) goes the other way in all 5 seeds.
  - E3-a on task B is robust only by a +0.0002 lower bound.
  - E2-c's SE_seed measures fine-tuning-seed noise.
