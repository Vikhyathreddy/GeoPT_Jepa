# GeoPT x JEPA: a toy-world experiment

Can GeoPT-style *lifted* pretraining (geometry + synthetic dynamics, explicit targets) and
JEPA-style *latent* prediction (EMA targets, action-conditioned predictors) be combined, and does
the combination help a physics simulator learn from fewer labeled simulations?

Everything runs on CPU in pure NumPy/SciPy (the sandbox had no PyTorch/JAX), using a small
gradient-checked autograd engine (`ag.py`, `test_ag.py`).

## World (`world.py`)
* Random smooth 2D bodies (star-shaped radial curves, stretched and rotated).
* **Pretext data (no labels):** GeoPT's lifted target. Particles get i.i.d. random velocities,
  move in straight lines, stick to the wall on contact, and the target is the vector distance to
  the wall at t = 0, 1, 2. Per sample: 160 volume + 32 surface particles.
* **Task A (labels):** surface speed q = |Vt|/U from potential flow (source-panel method,
  validated against the analytic circle/ellipse solutions to 1e-4). Elliptic and global, and
  exactly reversible: q(alpha) = q(alpha + pi).
* **Task B (labels):** steady advection-diffusion plume emitted by the body into the potential-flow
  velocity field (64x64 upwind finite volumes). Transport-dominated and directional.

## Backbone (`model.py`, 147k parameters)
32 surface-patch tokens -> 2 self-attention blocks (geometry encoder); queries carry position plus
a **velocity prompt** -> 2 cross-attention blocks (decoder) -> linear head.

## Pretext variants
| code | objective |
|---|---|
| S | none (scratch) |
| N | native geometry: vector distance at x |
| L | GeoPT lifted: vector-distance trajectory under a random velocity prompt |
| J | Geo-JEPA: predict EMA-target latents of masked surface patches |
| LJ | hybrid 1: L + 0.1 J (shared encoder) |
| LT | hybrid 2, latent transport: from (x0, v), predict the EMA-target latent of the point x_t the particle is carried to (JEPA predictor whose *action* is GeoPT's synthetic velocity), plus an explicit vector-distance anchor at v = 0 |
| LT0 | LT without the anchor (collapse probe) |

## Reproduce
The repo ships the pretrained checkpoints (`ckpt/`), results (`results/`), logs and figures.
The generated datasets (`data/`, ~200 MB) are not committed; `make_data.py` rebuilds them.
```bash
pip install -r requirements.txt   # numpy, scipy, matplotlib
python test_ag.py                 # gradient checks
python make_data.py               # ~3 min: 2,100 unlabeled shapes x 8 dynamics, 356 labeled shapes
for v in N L J LJ LT LT0; do python pretrain.py $v 3000; done
python run_all.py all             # 124 fine-tuning runs -> results/finetune.jsonl
python analyze.py                 # figures + tables
python probe_fields.py; python shortcut_probe.py
python plan_demo.py S L LT        # planning with the learned simulator
python plan_stats.py              # paired bootstrap of planning regret, action-ranking correlation
python paired_stats.py            # paired per-body comparisons with 95% bootstrap intervals
python followup_freeze.py         # prompt misalignment with the prompt-reading layer frozen
python followup_zero_prompt.py    # lifted model fine-tuned with no prompt at all
python qual_fig.py
```
Downstream fine-tuning works in the flow-aligned frame (see `finetune.py`'s docstring for why).
`results/finetune_absolute_frame_abandoned.jsonl` keeps the first, flawed attempt, in which every
model memorized its training shapes.

## Results at a glance (test relative L2 error, mean of 2 fine-tuning seeds)
| pretext | A@16 | A@64 | A@256 | B@16 | B@64 | B@256 |
|---|---|---|---|---|---|---|
| Scratch | 0.265 | 0.193 | 0.122 | 0.288 | 0.158 | 0.109 |
| Native geometry | 0.230 | 0.145 | 0.093 | 0.198 | 0.120 | 0.097 |
| Geo-JEPA | 0.258 | 0.173 | 0.104 | 0.224 | 0.133 | 0.108 |
| GeoPT-lifted | 0.211 | 0.115 | 0.073 | 0.185 | 0.113 | 0.089 |
| Lifted + JEPA (joint) | 0.208 | 0.111 | 0.073 | 0.186 | 0.113 | 0.089 |
| Latent-transport JEPA | 0.238 | 0.148 | 0.088 | 0.202 | 0.120 | 0.096 |

Labels needed to match scratch@256: GeoPT-lifted 56 (A) / 82 (B); joint 52 / 79; native 110 / 125;
latent transport 108 / 118; Geo-JEPA 165 / 240.

Caveats: 2D toy physics, 147k-parameter models, one pretraining seed per variant; the bootstrap
intervals cover test-body sampling, not pretraining variance.
