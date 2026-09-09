# Roadmap toward a publishable repository

## v0.2 — correctness oracle (current)

- [x] explicit 2-D cubical complex;
- [x] exact elementary-collapse test;
- [x] critical edge / vertex candidates;
- [x] forward event chain to empty complex on synthetic center/donut;
- [x] topology invariants in tests;
- [x] exact sequential macro batching;
- [ ] full Mixed-WM38K audit.

## v0.3 — scalable operator

- incremental coface counts and free-pair queue;
- maximal/sequential collapse sweeps;
- trajectory serialization;
- real 52x52 benchmark;
- statistics of unsupported cores;
- optional polar/wafer geometry weighting *only among valid Morse events*.

## v0.4 — reverse generator

- packed 0/1/2-cell tensor representation;
- event-type predictor;
- inverse-event candidate generator;
- event-support scorer;
- time/topology-level conditioning;
- path likelihood or discrete-flow alternative;
- exact structural projector.

## v0.5 — baselines and ablations

- Gaussian/DDPM wafer generation baseline;
- uniform categorical/absorbing baseline;
- DiGress-style discrete baseline;
- ConStruct-style constraint baseline;
- TopoDiffusionNet-style topology-loss baseline;
- node/deletion-order baseline approximating GraphARM.

Ablate:

1. binary state vs cubical-complex state;
2. random cell deletion vs elementary collapse;
3. no critical separation vs explicit critical events;
4. per-event vs macro-event reverse generation;
5. topology-only event weighting vs wafer geometry weighting.

## v1.0 — public reproduction package

- pinned environment;
- training/evaluation configs;
- checkpoints;
- dataset preparation scripts (without redistributing restricted data);
- paper tables/figures reproducible from commands;
- seed runs and statistical summaries;
- CI tests for topology invariants.
