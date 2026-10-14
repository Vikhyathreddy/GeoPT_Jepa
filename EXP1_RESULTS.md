# EXP1_RESULTS: Experiment 1: noisy pretext targets

Spec: `experiments/01_noisy_targets.md`.

## Pre-registered predictions
Copied verbatim from the spec before any run of this experiment. Not edited afterwards.

- **P1.** Label noise barely hurts L: less than 5% relative degradation even at sigma = 2. MSE
  regression learns the conditional mean; each batch averages about 18k target values; and the
  corpus holds 12M noisy numbers against 0.15M parameters, seen only about 3 times, so there is
  nothing to memorise. L's pretext error against clean targets should stay near its clean value.
  For the same reason, frozen and fresh noise should match at full corpus size (option d), while in
  the small-corpus regime (option e) frozen noise should hurt more than fresh.
- **P2.** If L does degrade under heavy label noise, LT overtaking it is guaranteed by construction
  and does not on its own confirm the rule of thumb.
- **P3.** Trajectory noise (the fair test) is genuinely uncertain. The rule of thumb predicts that
  LT degrades more slowly than L and crosses it at some sigma_pos. The alternative is that both
  degrade alike because the noise enters both target types the same way.
- **P4.** Stochastic bounce: if JEPA's multimodality argument applies here, L should be worse at
  p = 0.5 than at both deterministic endpoints, relative to LT. Caveat: MSE on a stochastic target
  simply learns its mean, which may be harmless, and only about 18% of particles are affected.
- **P5.** LJ is the most noise-robust explicit variant, because its J term carries a noise-free
  signal.

## Outcomes
(Added below as stages complete.)
