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
