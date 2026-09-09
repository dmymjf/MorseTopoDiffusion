# Code status

## Implemented

- explicit 2-D cubical complex and Betti computation
- elementary collapse discovery/application
- critical edge / critical vertex forward surgery
- exact forward trajectory simulation
- sequential macro batching of regular collapses
- interleaved mixed-dimensional tensor representation
- exact inverse structural event definitions (`R+`, `H+`, `C+`)
- inverse candidate enumeration restricted to a wafer domain
- exact inverse replay audit
- neural structural-event candidate scorer
- STOP token for reverse-chain termination
- event-level reverse training loss
- NPZ wafer adapter for `{0,1,2}` and binary encodings
- trajectory caching
- synthetic smoke training
- NPZ training
- conditional reverse-chain sampling
- unit tests and GitHub Actions

## Verified locally

- `pytest -q`: 8 passed
- `python scripts/demo_end_to_end.py`: forward, exact inverse replay and backward pass all pass
- tiny synthetic training checkpoint can be saved and reloaded by the sampler
- tiny NPZ pipeline: cache build + training completes without failure

## Deliberately not frozen

- final CTMC/path-likelihood objective
- scalable macro-event reverse decoder
- incremental free-pair data structure for full Mixed-WM38K throughput
- final baseline ports
- paper-level topology/morphology metrics
