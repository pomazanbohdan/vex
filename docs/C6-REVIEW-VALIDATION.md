# C6 review validation

## Behavior covered

- Fresh indexing, ordinary updates and revision refreshes use deterministic file ordering. Symbol-indexed vectors and cluster sidecars follow that order.
- A file with ordinary references and zero symbols retains its postings when another file changes. Duplicate occurrences retain their multiplicity.
- A full semantic fixture followed by an embeddings-disabled update retains the correct two-vector prefix of three symbols. A second seed retains one vector ending inside a two-symbol file. Both cases survive a second changed-file update and a no-op.
- Incomplete vector coverage removes both `index.hnsw` and `index.hashes`; valid stored vectors are retained without padding or extra inference.
- A valid revision marker is six bytes. Missing, truncated, wrong, trailing and oversized markers are stale; validation reads at most seven bytes. Invalidation and publication retain their existing sync/rename operations.

## Reproduce the automated checks

Run from a Rust/MSVC developer shell on Windows, using a target directory dedicated to this checkout:

```powershell
cargo +stable fmt --all --check
cargo +stable clippy --workspace --all-targets --locked --target-dir C:/temp/vex-c6-target -- -D warnings
cargo +stable test --workspace --locked --target-dir C:/temp/vex-c6-target
cargo +stable build --release --workspace --locked --target-dir C:/temp/vex-c6-target
```

Focused checks:

```powershell
cargo +stable test --locked --target-dir C:/temp/vex-c6-target --test core_reference_regression_test
cargo +stable test --locked --target-dir C:/temp/vex-c6-target --test partial_vector_update_test
cargo +stable test --locked --target-dir C:/temp/vex-c6-target --test async_parse_revision_test
cargo +stable test --locked --target-dir C:/temp/vex-c6-target --lib parse_revision::tests
```

`partial_vector_update_test` writes actual distinct 384-dimensional stored vectors and uses public `pipeline::run`/`pipeline::update`; it requires no model download or inference. It verifies vector bits by symbol identity, ordinary FST postings and paired HNSW cleanup.

## Default-options equality

The default-options regression enables clusters and the pattern index. Fresh cluster assignments are `FLAG_COMPUTED`; an incremental carry is `FLAG_COMPUTED | FLAG_STALE`. The test asserts both states and compares every other byte exactly after clearing only that defined stale bit. The no-clusters regression compares all bytes without normalization.

## Isolated release acceptance

The author separately runs the captured release CLI and MCP binaries in disposable temporary projects with private HOME/cache directories. Acceptance checks include exact CLI/MCP reference parity, revision-marker removal followed by refresh, no-op byte stability, and natural MCP EOF with exit zero. Two additional temporary projects exercise the 2/3 and 1/3 stored-vector cases through the release `vex update --no-semantic` command twice, followed by a no-op; retained vector bits, refs and HNSW sidecar absence are checked after each update.

The PR follow-up records the exact tested commit, release binary SHA256 and observed test counts. Local immutable receipts retain commands, stdout/stderr and source/artifact hashes.

## Timing boundary

Durability and the original migration cap are unchanged. The earlier one-time migration strict timing failure remains a documented limitation accepted by the author. These correctness checks do not establish a new performance pass, speedup, or cross-platform CI result. No new ETW capture is required.
