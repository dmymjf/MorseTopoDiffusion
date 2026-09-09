# MorseTopoDiffusion

Research code for **structural-event discrete diffusion on wafer-map cubical complexes**.

Instead of corrupting pixels/dies independently, a binary defect mask is lifted to a 2-D cubical complex and evolved with topology-defined events:

- `R-`: elementary collapse `(free face, unique coface)`, which preserves homotopy exactly;
- `H-`: exposed critical 1-cell removal, reducing `beta1` by one;
- `C-`: isolated critical 0-cell removal, reducing `beta0` by one.

The reverse model operates on the inverse event algebra `R+ / H+ / C+`. It **does not predict arbitrary pixels**. For each mixed-dimensional state, the operator enumerates topology-valid inverse structural events, and the neural network scores only those admissible candidates.

> Status: runnable research prototype. Forward operator, exact inverse replay, mixed-dimensional tensorization, reverse event network, NPZ trajectory cache, training scripts, sampling script, unit tests, and GitHub Actions are implemented. This is not yet a claim of final paper-level novelty or SOTA performance.

## 1. Core state representation

A wafer defect mask is represented as a finite cubical complex

```text
K = (K^0, K^1, K^2)
```

with vertices, horizontal/vertical edges, and defect squares. Intermediate corruption states may therefore be mixed-dimensional:

```text
2-D region -> 1-D skeleton -> critical topology event -> new stratum -> ... -> empty complex
```

For the neural model the complex is embedded into an interleaved lattice of size `(2H+1) x (2W+1)` with separate occupancy/domain channels for vertices, horizontal edges, vertical edges and squares.

## 2. Forward structural process

The reference forward chain is a continuous-time event process. At every state, admissibility is determined by the cubical topology:

```text
if elementary collapse exists: use R-
else if a critical exposed edge can kill one hole: use H-
else if an isolated critical vertex exists: use C-
else: fail loudly for audit
```

Regular events preserve `(beta0, beta1)` exactly. Each critical event reduces `beta0 + beta1` by exactly one.

## 3. Reverse structural-event model

Given a current complex `K`, time/progress `t`, and optional multi-label condition `y`:

1. enumerate all admissible inverse events `R+ / H+ / C+` inside the wafer domain;
2. encode the mixed-dimensional complex using a conventional FiLM residual CNN;
3. score each **structural candidate event** using local features at its cells plus global/context features;
4. optionally choose `STOP`.

This preserves the research emphasis on the transition algebra rather than a bespoke backbone.

## 4. Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -e .[dev]
```

A CUDA-enabled PyTorch installation can be used for training, but the operator tests run on CPU.

## 5. Verify the repository

```bash
pytest -q
python scripts/demo_end_to_end.py
```

Current expected test result:

```text
8 passed
```

The end-to-end demo verifies three things:

```text
forward structural corruption -> empty complex
exact inverse event replay -> original complex
neural reverse-event loss -> successful backward()
```

## 6. Tiny synthetic training smoke test

```bash
python scripts/train_synthetic.py \
  --steps 30 \
  --size 20 \
  --channels 24 \
  --out outputs/synthetic_smoke.pt
```

Then sample:

```bash
python scripts/sample_conditional.py \
  --ckpt outputs/synthetic_smoke.pt \
  --label 0,1,0,0,0,0,0,0 \
  --size 20 \
  --max-events 200 \
  --out outputs/sample.png
```

A smoke-trained model is only for checking code execution; meaningful generation requires real training.

## 7. Mixed-WM38K / NPZ interface

The current data path expects:

```text
images: [N,H,W]
labels: [N,K]
```

Common wafer encoding `{0=outside, 1=normal, 2=defect}` is detected automatically. For binary defect masks, pass an explicit wafer-domain mask if available; otherwise a circular domain is used.

Build the forward trajectory cache:

```bash
python scripts/build_trajectory_cache.py \
  --npz mixedwm38k_train.npz \
  --out outputs/mixedwm38k_train_morse.pkl
```

Train the reverse structural-event model:

```bash
python scripts/train_npz.py \
  --npz mixedwm38k_train.npz \
  --cache outputs/mixedwm38k_train_morse.pkl \
  --steps 10000 \
  --channels 48 \
  --out outputs/morsetopo.pt
```

For a separate wafer mask array, add for both commands:

```bash
--mask-key masks
```

## 8. Repository layout

```text
MorseTopoDiffusion/
├── src/morsetopo/
│   ├── complexes/
│   │   ├── cubical.py          # explicit planar cubical complex
│   │   └── tensor.py           # mixed-dimensional neural lattice encoding
│   ├── operators/
│   │   ├── events.py           # forward event records
│   │   ├── morse.py            # R-/H-/C- operator
│   │   └── reverse.py          # exact inverse R+/H+/C+ candidate algebra
│   ├── diffusion/
│   │   ├── forward.py
│   │   ├── macro_forward.py
│   │   └── reverse_chain.py
│   ├── models/
│   │   └── event_net.py        # candidate-event scorer
│   ├── training/
│   │   ├── reverse_dataset.py
│   │   └── losses.py
│   └── data/
│       ├── synthetic.py
│       └── wafer.py
├── scripts/
│   ├── validate_operator.py
│   ├── benchmark_macro.py
│   ├── demo_end_to_end.py
│   ├── build_trajectory_cache.py
│   ├── train_synthetic.py
│   ├── train_npz.py
│   └── sample_conditional.py
├── tests/
├── configs/
├── docs/
├── baselines/
└── .github/workflows/tests.yml
```

## 9. What is inherited vs research-specific

The project does **not** claim discrete diffusion, CTMCs, cubical complexes, discrete Morse theory, topology-aware generation, or neural candidate scoring as individually new.

The research target is the combination of:

1. a mixed-dimensional cubical state space for wafer defects;
2. elementary-collapse/critical-cell events as the corruption transition algebra;
3. reverse generation over topology-valid inverse structural events instead of arbitrary pixel/token state changes.

See:

- `docs/NOVELTY_MAP.md`
- `docs/RELATED_CODE.md`
- `docs/METHOD_SKETCH.md`
- `docs/ROADMAP.md`

## 10. Immediate research checkpoints

Before treating this as a final model:

1. audit every Mixed-WM38K training mask for unsupported non-collapsible states;
2. measure micro-event counts and macro compression ratios at 52x52;
3. compare against independent categorical masking, DiGress-style transitions, GraphARM-style ordering, and ConStruct-style projection;
4. evaluate generated morphology/topology distributions, not only classification accuracy;
5. only then freeze the final reverse objective and acceleration scheme.
