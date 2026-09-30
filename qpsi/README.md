# qpsi — the ECG tokenizer

Turns a raw 12-lead recording into the feature space the frozen classifier was
trained on:

```python
from qpsi.config import QPSIConfig
from qpsi.pipeline import run_pipeline

result = run_pipeline(
    signal,                      # (12, n_samples) float array, millivolts
    fs=500,
    recording_id="HR16370",
    plot_enabled=False,
    qpsi_config=QPSIConfig(compute_direct_measurements=True),
)
final = result["_final_json"]
features = {**final["direct_features"], **final["plane_direct_features"]}
# 26,490 keys: ml_direct::<lead>::<P|QRS|T>::<group>::<stat>
#              plane_direct::<limb|chest>::<region>::<group>::<stat>
```

Lead order is positional and fixed: `I, II, III, aVR, aVL, aVF, V1…V6`.

The classifier consumes only the direct-measurement track — the no-Gaussian
per-beat primitives aggregated to per-recording statistics. The Gaussian
decomposition still runs (the pipeline is one unit) and populates the clinical
JSON, but none of its output is in the model's feature space.

Backend callers should go through `backend/services/inference.py`
(`extract_features`) rather than calling `run_pipeline` directly — it applies
the exact extraction settings the universe was built with and coerces the
result into the package's `feat_order`.

## Dependencies

numpy, scipy, scikit-learn, wfdb. Nothing else is required.

`qpsi[nn]` (torch) is only needed for the AF neural probability, which is
opt-in via `QPSI_AF_NEURAL_ENABLE=1` and off by default; `qpsi[viz]`
(matplotlib/plotly) only for the plotting helpers.

## Optional Rust accelerator

`qpsi_native/` is a Rust crate covering the hot kernels. It is a speed option,
not a correctness one — the direct-measurement features come out bit-identical
either way (verified on shipped records: 26,490 of 26,490 values exactly equal,
max abs difference 0.0). Extraction takes ~1.3 s per record with it and ~21 s
without.

```bash
pip install maturin
maturin build --release -m qpsi/qpsi_native/Cargo.toml
pip install target/wheels/qpsi_native-*.whl
```

The build needs a Rust toolchain and invokes Python once to generate constants
from `constants.py` (set `PYTHON` if `python` is not the interpreter you want).
When the extension is absent, every call site falls back to its Python path and
logs that it did.

## What was left behind

This is the extraction path only. The upstream package also carried
ground-truth audit harnesses — paired-extraction divergence audits, a
single-patient deep dive, a Rust-vs-Python diagnostic, snapshot capture, a
parity gate — plus their pinned baselines and the WFDB-root resolver they
share. Each one drives a corpus that exists only on the machine the pipeline
was developed on, so none of them are here. Nothing in the extraction path
imports them.
