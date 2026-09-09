# Related papers and code to audit before freezing the method

The purpose of this list is not to copy implementations. It is to identify mature infrastructure and to keep the novelty boundary honest.

## Core discrete diffusion / graph generation

1. **DiGress: Discrete Denoising Diffusion for Graph Generation (ICLR 2023)**
   - Paper: https://openreview.net/forum?id=UaAD-Nu86WX
   - Code: https://github.com/cvignac/DiGress
   - Inspect: transition matrices, discrete reverse model, datasets, sampling, evaluation.

2. **Autoregressive Diffusion Model for Graph Generation / GraphARM (ICML 2023)**
   - Paper: https://proceedings.mlr.press/v202/kong23b.html
   - Third-party PyTorch implementation: https://github.com/caio-freitas/GraphARM
   - Inspect: topology/data-dependent absorption ordering. This is the closest conceptual red line.

3. **EDGE: Efficient and Degree-Guided Graph Generation via Discrete Diffusion Modeling (ICML 2023)**
   - Paper: https://proceedings.mlr.press/v202/chen23k.html
   - Code: https://github.com/tufts-ml/graph-generation-EDGE
   - Inspect: sparse edge absorption and reverse sampling.

4. **Blackout Diffusion (ICML 2023)**
   - Paper: https://proceedings.mlr.press/v202/santos23a.html
   - Code: https://github.com/lanl/Blackout-Diffusion
   - Inspect: discrete Markov corruption to an empty state.

5. **SEDD (ICML 2024)**
   - Paper: https://proceedings.mlr.press/v235/lou24a.html
   - Code: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
   - Inspect: reverse rate/density ratio and score-entropy training.

6. **MD4 (NeurIPS 2024)**
   - Code: https://github.com/google-deepmind/md4
   - Inspect: masked diffusion and generalized/state-dependent schedules.

7. **ConStruct (NeurIPS 2024)**
   - Paper: https://proceedings.neurips.cc/paper_files/paper/2024/hash/f82385b8804009f9a81e1a30f1ff14e3-Abstract-Conference.html
   - Code: https://github.com/manuelmlmadeira/ConStruct
   - Inspect: absorbing transition, projector, structural constraints, DiGress integration.

8. **DeFoG: Discrete Flow Matching for Graph Generation (ICML 2025)**
   - Paper: https://proceedings.mlr.press/v267/qin25d.html
   - Code: https://github.com/manuelmlmadeira/DeFoG
   - Inspect: if event generation fits flow matching better than diffusion.

## Topology-aware generation

9. **TopoDiffusionNet (ICLR 2025)**
   - Paper: https://proceedings.iclr.cc/paper_files/paper/2025/hash/4e889e581d4a6f2be932c6f65e7792a8-Abstract-Conference.html
   - Code: https://github.com/Saumya-Gupta-26/TopoDiffusionNet
   - Inspect: PH loss and topology control; make sure our topology is in the operator, not merely loss/guidance.

10. **TopoCellGen (CVPR 2025)**
    - Paper: https://openaccess.thecvf.com/content/CVPR2025/html/Xu_TopoCellGen_Generating_Histopathology_Cell_Topology_with_a_Diffusion_Model_CVPR_2025_paper.html
    - Code: https://github.com/Melon-Xu/TopoCellGen
    - Inspect: topology metrics, cell-count control, 2-D biological layout analogy.

## Discrete Morse theory / structural representation

11. **Topology-Aware Segmentation Using Discrete Morse Theory (ICLR 2021, Spotlight)**
    - Code: https://github.com/HuXiaoling/DMT_loss
    - Inspect: practical discrete-Morse structures and implementation details.

12. **Learning Probabilistic Topological Representations Using Discrete Morse Theory (ICLR 2023, Spotlight)**
    - Paper: https://arxiv.org/abs/2206.01742
    - Code: https://github.com/HuXiaoling/Structural_Uncertainty
    - Inspect: probabilistic structural representation rather than pixel-wise output.

## Audit questions for every repository

- What is the actual state variable?
- What is one forward transition?
- Does the transition depend on the current global structure?
- Does topology enter the forward kernel, reverse kernel, loss, condition, or projector?
- What is the terminal prior/state?
- What does the neural network predict?
- Does the code have an analytic forward marginal or simulate trajectories?
- How many model evaluations are required per sample?
- What structural invariants are guaranteed versus merely encouraged?
