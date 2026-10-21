# GeoPT × JEPA follow-ups: status and plan

Repository: `Vikhyathreddy/GeoPT_Jepa`, branch `claude/followup-experiments`. Detailed results: `EXP1_RESULTS.md`, `EXP2_RESULTS.md`, `EXP3_RESULTS.md`. Seed rule: `SEED_REPLICATION.md`.

## Done so far

### 1. Repository and setup
- **Original study pushed** to `main`. One bug fixed: `make_data.py` crashed on a fresh clone because `data/` didn't exist.
- **Reproduction:** the original numbers reproduce exactly on this machine (scratch 0.190967, L 0.114163; task A, n = 64, seed 0).
- **Shared infrastructure added** per `experiments/00_setup.md`, with every default behaviour checked bit-identical against `main` (727 arrays compared):
  - `N_PROCS` for parallel workers
  - the `pretrain.py` command line and hooks
  - the `finetune.py` options (frame, prompt mode, checkpoint, data)
  - a resumable job runner and a paired-bootstrap helper
- **Earlier ad-hoc experiments reverted.** I had built quick versions of all three experiments before your specs arrived; they didn't follow the specs.
- **Predictions pre-registered** in `EXPn_RESULTS.md` before any run. They have never been edited.

### 2. Experiment 1: latent vs explicit targets under noise
- **Stage 1:** label noise (σ = 0.5, 2) and trajectory noise (0.15, 0.30), with L and LT.
- **Stage 2:** label noise at σ = 4 and 8 plus fresh noise; LJ; a second pretraining seed; diagnostics. It ran under your dated amendment, which judges L by its advantage over N.
- **Results:**
  - **L stays ahead of LT up to σ = 4.** LT wins only at σ = 8, after L has fallen below native geometry.
  - **Trajectory noise up to 0.30 barely affects either model.**
  - **Fresh noise = frozen noise.** LJ is no more robust than L.
  - **Your 1.4% sticking-signal measurement reproduces.** Label noise adds error roughly in proportion to each kind of particle's clean error, so it doesn't single out wall-hitting particles.
  - **Pretraining-seed variance is large.** With seed 1, clean LT beats N.
- **After the audit: unresolved.**
  - The label-noise "refutation" is fixed by construction at seed 0.
  - At seed 1, L only ties LT at the halving level.
  - The trajectory family stopped before L's advantage halved.

### 3. Experiment 2: does a misaligned prompt hurt?
- **Gates.** Canonical bodies, absolute frame, per-sample prompt.
  - G1 passes with a written caveat: scratch memorises (train error 0.03 vs test 0.33).
  - G2: the prompt matters a lot (removing it more than doubles L's error).
- **Step 1:** scratch's 0°/90°/180° differences are within its own seed noise. Scratch is flat at 90° and 180° by symmetry, not because it ignores the prompt.
- **Step 2:** P6 and P7 were pre-registered before any n = 64 run existed.
- **Step 3:** 330 runs, including L pretraining seed 1 and the frozen-`q.1` runs (Stage 3a).
  - **P6 holds:** on task A, 180° costs nothing.
  - **P7 holds by its rule:** on task B, reversed beats aligned at n = 64 in both L seeds. The effect is within fine-tuning-seed noise, and robust across fine-tuning seeds only for L seed 0.
  - **Verdict: unresolved.** A penalty appears only at n = 16 on task A. At n = 64, only a 90° shift consistently *helps*. On task A that gain grows with body elongation (exploratory).
  - **Lc's 180° penalty at n = 16 was investigated:** no bug. The likely cause is checkpoint-level.
  - **P4 is recorded as you decided** (failed on A, passed on B). Neither gap is significant.
- **Stage 3a, with the prompt-reading layer `q.1` frozen (n = 64):**
  - Freezing costs accuracy at the aligned prompt: task A +6-8%; task B +51% (L) and +55% (Lc).
  - Even frozen, misalignment never hurts. On task B the reversed prompt helps a lot (L 0.275 → 0.191, close to the unfrozen aligned 0.186).
  - The spec's "penalty only when frozen" pattern doesn't occur.
  - Caveat: `q.2` and the decoder still train, so this doesn't isolate the pretrained prompt semantics.

### 4. Experiment 3: do latent-transport features rank designs better?
- **Gate 1:** linear probes for all 7 backbones, plus R1 and R3 for S, L and LT (3 fine-tuning seeds).
- **Results:**
  - **No regime shows LT ranking better than L.** L is clearly better in 3 cells and ties elsewhere.
  - **The original plan demo's LT edge doesn't reappear.**
  - **Training on 8 actions per body improves planning a lot,** though with confounds.
  - **The probe result replicates at pretraining seed 1.**
- **Verdict:** refuted at seed 0. R1/R3 remain single-seed.

### 5. Audits (step 5)
- **Nine independent agents** recomputed every number, checked the code and challenged the claims.
  - Everything reproduced, and no bugs were found.
  - Interpretive corrections are marked "corrected after the audit" in all three results files.

### Current reading

| Question | Verdict now |
|---|---|
| 1. Latent targets win when targets are noisy? | Unresolved (no support; no informative refutation) |
| 2. Misaligned prompt hurts? | Unresolved (penalty only at n = 16, task A; 90° helps at n = 64) |
| 3. LT ranks designs better? | Refuted at pretraining seed 0; probe replicates at seed 1 |

## Step 4 and the write-up (done, 2026-10-21)
- **Five pretraining seeds:**
  - 16 new pretraining runs;
  - Experiment 1: 64 fine-tuning runs;
  - Experiment 2: 168 runs at n = 16 and 64;
  - Experiment 3: 24 probes.
- **Audited by a subagent:** all numbers reproduce, and the pre-registration order is confirmed.
- **Question 1 took branch 2.** Clean LT beating N is single-seed (A 3/5, B 4/5), so question 1 is closed as not testable with this latent design. No stronger noise was run.
- **Final verdicts and robust / single-seed labels:** see `FINAL_REPORT.md`.

## Remaining (optional, only if you ask)
- Equal-budget test for Experiment 3 (step 6).
- A second Lc pretraining seed.
- Experiment 3 Stage 2 (R2).
