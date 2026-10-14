# EXP3_RESULTS: Experiment 3: do latent-transport features plan better?

Spec: `experiments/03_action_ranking.md`.

## Pre-registered predictions
Copied verbatim from the spec before any run of this experiment. Not edited afterwards.

- **P1 (null; the analyst leans this way).** Ranking quality tracks field accuracy, so L >= LT on
  rho in every regime, and the original LT edge was noise.
- **P2 (alternative).** LT's features encode action-relevant structure better than its field error
  suggests: LT > L on rho in the probe and in R3 (CI excluding 0), at least on task B, while L still
  beats LT on field error.
- **P3.** With 8 actions per body, R1 planners clearly beat the original single-action surrogates
  (plan-demo rho: B 0.82-0.86, A 0.70-0.84), because multi-action data shows the effect of the
  action directly.
- **P4.** Dense field supervision beats KPI-only supervision at this budget (R1 > R3), and R2 gives
  at most a small ranking gain over R1.

## Outcomes
(Added below as stages complete.)
