# GeoPT x JEPA toy world: notes for Claude Code

A small 2D physics world that tests whether GeoPT-style lifted pretraining (explicit targets from
synthetic dynamics) and JEPA-style latent prediction can be combined. Pure NumPy/SciPy/matplotlib,
CPU only, custom autograd in `ag.py`. `README.md` has the original study's results.
`experiments/` holds the specs for the follow-up experiments: follow them, and when a spec is
silent, make a sensible choice and write it down in the results file.

## File map
- `ag.py`: reverse-mode autograd (ops + AdamW). `test_ag.py`: finite-difference checks (float64).
- `world.py`: bodies (`make_shape`), GeoPT pretext (`vector_distance`, `transport_step`,
  `pretext_samples`), task A solver (`panel_solve`: potential-flow surface speed), task B solver
  (`plume_solve`: steady advection-diffusion, 64x64 grid), dataset builders.
- `make_data.py`: writes `data/pretrain.npz` (2,100 bodies x 8 dynamics; `sid >= 2000` are validation
  bodies) and `data/downstream.npz` (356 bodies: TRAIN = 0-255, TEST = 256-355). About 3 min; not in git.
- `model.py`: backbone (32 surface-patch tokens -> 2 self-attention blocks; queries
  [Fourier(x), prompt v, |v|] -> 2 cross-attention blocks; D = 64; ~147k params) and pretext losses.
- `pretrain.py`: pretraining loop (EMA targets for J/LJ/LT/LT0) -> `ckpt/<V>.npz` + `.json` log.
- `finetune.py`: downstream fine-tuning and evaluation in the flow-aligned frame (`fit`, `finetune`,
  `evaluate`, `prompts`).
- `run_all.py`: the original 124-run grid -> `results/finetune.jsonl`.
- Analyses: `analyze.py` (figures, tables), `paired_stats.py`, `probe_fields.py`, `shortcut_probe.py`,
  `plan_demo.py`, `plan_stats.py`, `followup_freeze.py`, `followup_zero_prompt.py`, `qual_fig.py`.

## Variant codes
S scratch | N native geometry (vector distance at x) | J Geo-JEPA (masked patches; encoder only) |
L GeoPT-lifted (vector distance along the trajectory) | LJ = L + 0.1*J |
LT latent transport (predict the EMA latent at the landing point, plus an explicit anchor at v = 0) |
LT0 = LT without the anchor.

## Facts and gotchas (read before changing code)
- Downstream key `Cp` actually stores surface speed q = |Vt|/U (legacy name). `ptsA` holds panel
  midpoints [::2] (128 points).
- Downstream training uses the flow-aligned frame: `finetune.load` rotates each body by -alpha, and
  the prompt is the same constant (+x) for every sample. The absolute frame with randomly oriented
  bodies failed: models memorised their training bodies and lost to the mean predictor
  (`results/finetune_absolute_frame_abandoned.jsonl`). Any new downstream setup must pass this guard:
  scratch at n = 64 beats the mean predictor by a wide margin.
- Potential flow is reversal-symmetric: q(alpha) = q(alpha + pi) exactly, so task A costs come in tied pairs.
- Pretext arrays: `X` (N,3,192,2) particle positions at t = 0,1,2 (first 160 volume particles, then
  32 surface particles); `v` (N,192,2) per-particle velocity; `vd` (N,3,192,2) vector distance to
  the wall at each t. LT/LT0/J never read `vd[:,1:]`; L and LJ do.
- Measured: only 17.6% of particles touch the wall within two steps (volume 11.5%, surface 48%).
  Std of `vd[:,1:]` = 0.483; of `vd[:,0]` = 0.356; mean |v| = 0.333.
- Pretraining = 3000 steps x batch 16, about 3 epochs over the 16,000 training samples. The CLI
  default in `pretrain.py` is 2500 steps; the original runs used 3000.
- Performance: never write `x**3` on float32 arrays in hot paths (about 70x slower than `x*x*x`).
  Worker processes must set OPENBLAS_NUM_THREADS / OMP_NUM_THREADS / MKL_NUM_THREADS = 1.
- Every new autograd op needs a finite-difference test in `test_ag.py` that passes before the op is used.
- Timings on a 2-core VM, per process: pretraining 3000 steps takes L 4.5 min, LJ 5.4, LT 9.6,
  LT0 7.4, N 4.6, J 1.8. Fine-tuning 1000 steps + eval: task A about 27 s, task B about 55 s.

## Rules for the follow-up experiments
- Never modify or overwrite the original outputs: `results/finetune.jsonl`, the other files in
  `results/`, and `ckpt/{S,N,J,L,LJ,LT,LT0}.npz`. New work goes to `results/expN_*`, `ckpt/expN/`,
  `data/expN/`, `figs/expN_*`.
- Default behaviour of existing functions must stay bit-identical. Add parameters with defaults,
  and verify with `np.array_equal` or by re-running a reproduction check.
- Pre-register: before running an experiment, copy its Predictions section into `EXPn_RESULTS.md`
  and commit. Never edit those predictions afterwards; add the outcomes below them.
- Never select anything (hyper-parameters, lambda, checkpoints) on TEST bodies 256-355. Use held-out
  TRAIN bodies.
- Long runs: launch in the background with nohup, log to `logs/`, and make jobs resumable
  (append-only JSONL keyed by job). Poll the logs. Time one job first, and before launching anything
  estimated at over 2 hours, report the estimate and wait.
- Statistics: paired comparisons on the same test bodies, with per-body errors averaged over seeds,
  95% bootstrap CIs, and the % of bodies improved. Say which seeds (pretraining vs fine-tuning) the
  CI does and does not cover.
- Stop at every Gate in a spec and report before continuing.
- Figures: follow `analyze.py` conventions (COL/MRK/NAME per variant, direct labels, legend below the
  plot, light background). Open every figure and check it before reporting.
- Commit after each stage with a descriptive message.
