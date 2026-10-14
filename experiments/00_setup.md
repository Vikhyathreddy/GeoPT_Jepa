# 00: Setup, reproduction check, shared infrastructure

Goal: confirm that the code reproduces the original numbers on this machine, then add the small
amount of shared infrastructure the three experiments need, without changing any default behaviour.

## Steps

1. **Environment.** Python >= 3.10 with numpy, scipy and matplotlib. No GPU or PyTorch needed.
   Create a virtualenv if needed. Write the Python/numpy/scipy versions and the CPU core count to
   `results/env.txt`.

2. **Version control.** If this is not already a git repository: `git init`, keep the provided
   `.gitignore` (it excludes `data/`, `.venv/`, `__pycache__/`), and commit everything as
   "baseline: original study".

3. **Autograd check.** `python test_ag.py` must end with `ALL OK`.

4. **Data.** `python make_data.py` should print `pretrain: 2100 shapes, 16800 samples` and
   `downstream: 356 shapes`.

5. **Reproduction check.** With the shipped checkpoints, run `finetune.finetune(D, V, "A", 64, 0)`
   for V = "S" and V = "L" (with `D = finetune.load()`). Expected test relative L2: S about 0.1910,
   L about 0.1142 (see `results/finetune.jsonl`). Within 10% relative is acceptable, since
   numpy/scipy/BLAS versions differ across machines. If a number is outside that range, stop and
   report before changing anything.

6. **Shared infrastructure.** Each item must leave default behaviour bit-identical. After the
   refactor, re-run step 5 and confirm the same numbers you got in step 5 on this machine.
   - a. `N_PROCS` environment variable for every multiprocessing pool (default 2). Workers set
     OPENBLAS/OMP/MKL thread counts to 1 before importing numpy.
   - b. `pretrain.py` CLI:
     `python pretrain.py VARIANT --steps 3000 --seed 0 --data data/pretrain.npz --out ckpt/x.npz --train-shapes 2000`.
     Keep the old positional form `python pretrain.py L 3000` working, writing `ckpt/L.npz` as before.
   - c. Hooks in `pretrain()`:
     - an optional `batch_fn(batch, rng_aux)` applied to each training batch after `make_batch`,
       used for fresh label noise. Seed `rng_aux` separately (for example
       `np.random.default_rng(seed + 10_000)`), so that initialisation and batch order are identical
       to the clean run with the same seed.
     - optional clean validation targets: when the dataset has `vd_clean`, also log the explicit
       heads' validation loss against the clean targets.
     - `train_shapes`: train only on samples with `sid < train_shapes`; validation stays `sid >= 2000`.
   - d. In `finetune.fit()` / `finetune()` / `evaluate()`:
     - `ckpt=` path override (default `ckpt/<variant>.npz`) and `encoder_only=` (default
       `variant == "J"`).
     - `frame="flow"` (default, current behaviour) or `"absolute"`: no rotation of geometry or
       queries, and the prompt becomes per-sample, `0.25*U*(cos(alpha+shift), sin(alpha+shift))`
       with U = 1 for task A.
     - `prompt_mode="normal"` (default) or `"zero"` (prompt set to 0).
     - `data=` path for the downstream file (default `data/downstream.npz`).
   - e. A generic resumable runner in the style of `run_all.py`: a list of job dicts goes in, JSON
     lines are appended to a given results file, jobs whose key already exists are skipped, each job
     logs one line, and `N_PROCS` workers run in parallel. Store per-body errors in every row.
   - f. A generic paired-bootstrap helper that generalises `paired_stats.py`: inputs are two sets of
     rows; per-body errors are averaged over seeds; outputs are the mean relative change, its 95% CI
     and the % of bodies improved.

7. **Commit** with the message "infrastructure for follow-up experiments".

## Report back
The step-5 numbers next to the expected ones, the core count, and anything that surprised you.
