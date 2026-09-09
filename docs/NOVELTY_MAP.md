# Novelty map: what we borrow, what we modify, what must be ours

## 1. Mature machinery we should reuse

These should not be claimed as contributions:

- discrete/categorical diffusion as a model class;
- absorbing/empty terminal states;
- reverse Markov parameterization;
- GNN/CNN/Transformer time-conditioned backbones;
- EMA, optimizer, checkpointing, distributed training;
- standard generation metrics and basic topology metrics.

## 2. Closest prior work and the boundary we must respect

### DiGress (ICLR 2023)
Discrete denoising diffusion for graph node/edge categories. Useful as training and evaluation infrastructure.

**Do not claim:** discrete graph diffusion itself.

### GraphARM (ICML 2023)
Learns a data-dependent graph node absorption order.

**Red line:** if our method reduces to "use current topology to choose the next die/node to delete", it is too close.

### EDGE (ICML 2023)
Uses discrete edge deletion toward an empty graph.

**Do not claim:** structural deletion to an empty state.

### ConStruct (NeurIPS 2024)
Absorbing graph diffusion with projections enforcing structural constraints throughout generation.

**Do not claim:** "first constrained diffusion trajectory" or topology/structure validity by itself.

### SEDD (ICML 2024), MD4 (NeurIPS 2024)
Mature discrete diffusion objectives, density-ratio/score formulations, state-dependent masking variants.

**Do not claim:** state-dependent schedules in general.

### TopoDiffusionNet (ICLR 2025), TopoCellGen (CVPR 2025)
Use persistent-homology information/losses to improve/control generated topology.

**Do not claim:** first topology-aware diffusion or first Betti/persistent-homology diffusion.

### Discrete Morse topology learning (ICLR 2021 / ICLR 2023)
Discrete Morse theory has already been used for topology-aware segmentation and probabilistic structural representation learning.

**Do not claim:** first use of discrete Morse theory in deep learning.

## 3. Research contribution we are targeting

### A. Mixed-dimensional cubical state space

Instead of storing every intermediate state as a binary mask, corruption evolves a finite cubical complex containing 0-, 1-, and 2-cells. A thick defect can collapse to a 1-D skeleton or 0-D critical cell while retaining its homotopy type.

Target claim (subject to final literature audit):

> A discrete generative corruption process for wafer maps whose intermediate states are mixed-dimensional cubical complexes.

### B. Morse event operator

A regular transition is an elementary collapse `(sigma,tau)` where `sigma` is a free face of the unique coface `tau`. It preserves homotopy by construction.

When no regular collapse is available, a critical-cell surgery changes the topology by one unit:

- critical 1-cell removal: `beta1 -> beta1 - 1`;
- critical 0-cell removal: `beta0 -> beta0 - 1`.

Thus topology is not a loss or condition; it changes which transitions are mathematically admissible.

### C. Criticality-separated probability path

The path alternates between:

- regular collapse phases within a homotopy stratum;
- sparse critical events between strata.

This provides a two-scale generative path: morphology/skeleton simplification versus topological surgery.

### D. Structural reverse target

The reverse network should predict inverse events rather than pixel noise:

- elementary expansion;
- critical 1-cell attachment;
- critical 0-cell birth.

For scalable generation, an event may be macro-batched as a sequentially valid collapse/expansion sweep.

## 4. What would make the paper too weak

The following final forms are not sufficient:

- DiGress + a wafer mask;
- ConStruct + Betti constraint;
- DDPM + persistent-homology loss;
- topology feature concatenation;
- GraphARM-style learned die deletion order;
- Mamba/attention inserted into an otherwise standard discrete diffusion.

## 5. Falsification criteria

We should abandon or substantially change this direction if the literature audit finds an existing generative model that already combines all of the following:

1. a cubical/simplicial complex as the actual diffusion state;
2. elementary homotopy collapses as forward corruption events;
3. explicit critical-cell topology-changing events;
4. a learned reverse generative process over inverse structural events.

A domain transfer to wafer maps alone would not be enough.
