# Exact-query cache benchmark

Run from this PR checkout with Python 3.10+ and the normal Vex build prerequisites.
The checked-in `fixture/` contains two source files and seven symbols. The runner
uses only Python's standard library. On Windows use `python` and the `.exe`
binary suffix.

## Build and run

```sh
git worktree add --detach ../vex-before ca47682f0fd6691e490c6ee7d28f8d29829e7c15
cargo build --release --locked --bin vex --manifest-path ../vex-before/Cargo.toml
cargo build --release --locked --bin vex
python3 scripts/benchmark-query-cache.py --baseline-exe ../vex-before/target/release/vex --head-exe target/release/vex --model-cache-root /path/to/vex/embeddings --run-dir ../query-cache-benchmark --runs 5
```

Use release builds with the same optional features on both sides. The model
cache root must already contain `models--Qdrant--all-MiniLM-L6-v2-onnx`, including
its `refs/main` and active snapshot. A previous CPU semantic index can populate
that cache. Common roots are `$XDG_CACHE_HOME/vex/embeddings` or
`$HOME/.cache/vex/embeddings` on Linux, `$HOME/Library/Caches/vex/embeddings` on
macOS, and `%LOCALAPPDATA%\vex\embeddings` on Windows. The runner copies the
model repository into a new run directory and blocks model-network fallback.
It does not install models or alter the original cache.

The runner pins `minilm-l6-v2` in its isolated configuration and removes the
ambient `VEX_EMBEDDER` override. The run directory must not exist. Keep other CPU-heavy jobs stopped during the
measurement. The runner builds one shared CPU semantic index, warms the exact
head query cache, then alternates baseline/head order for at least five pairs.
It records each argv, exit, elapsed time, stdout, stderr, binary SHA, model SHA,
source SHA, index SHA and query-cache state in `benchmark-results.json`.

## Correctness and controls

- Parsed results and the JSON envelope must match. Only the dynamic
  `_meta.vex.dev/index_age_ms` field is excluded; raw stdout is retained.
- Every timed head query must leave its one-record, 384-dimension cache entry
  unchanged.
- Case-changed query bytes and a SHA-only model-config change must miss the old
  cache identity while preserving results.
- Corrupting a copied query-cache entry must use fresh inference; restoring it
  must hit before the model factory.
- Source, index, configuration and copied model bytes must be unchanged or
  restored when the run finishes.
- Process startup and an artifact-SHA pass are controls. The SHA control hashes
  the five guarded model files once; the Rust hit path validates identity twice.

Exit 0 means the run and all controls passed. It does not establish a general
speedup. Inspect `summary.validity_failures`, paired samples, medians, ranges
and the recorded machine before making a timing claim. This is a warm exact-query
CPU fixture measurement; cold queries, unrelated queries, large repositories and
Dart Analysis Server latency are separate measurements. Model initialization and
inference are not profiled separately.

Unit coverage can be run without this timing experiment:

```sh
cargo test --locked --lib embed::query_cache::tests
```
