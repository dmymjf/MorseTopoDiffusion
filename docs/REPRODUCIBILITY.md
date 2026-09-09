# Reproducibility checklist

The public-repository target should satisfy:

- exact train/validation/test split script and seeds;
- exact preprocessing and wafer-mask convention;
- immutable forward-operator config saved in every checkpoint;
- topology-invariant CI tests;
- versioned baseline commits;
- 3-5 random seeds for neural experiments;
- per-class and mixed-type generation metrics;
- runtime, memory, number of reverse evaluations;
- ablations tied one-to-one to claimed contributions;
- scripts that regenerate every paper table and major figure.
