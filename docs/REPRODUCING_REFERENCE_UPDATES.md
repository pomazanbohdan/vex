# Reproduce reference loss on an unrelated update

Run from this PR checkout with Python 3.10+ and the normal Vex build
prerequisites. On Windows use `python` and the `.exe` binary suffix.

```sh
git worktree add --detach ../vex-before ca47682f0fd6691e490c6ee7d28f8d29829e7c15
cargo build --release --locked --bin vex --manifest-path ../vex-before/Cargo.toml
cargo build --release --locked --bin vex
python3 scripts/reproduce-update-refs.py --vex ../vex-before/target/release/vex --expect before --run-dir ../refs-before
python3 scripts/reproduce-update-refs.py --vex target/release/vex --expect after --run-dir ../refs-after
```

Each run directory must be new. The script creates three Rust source files and
an isolated index. It records a `target_name` reference in `caller.rs`, edits a
different file, and performs an explicit incremental update.

The upstream mode requires the reference to exist initially and disappear after
the unrelated update. The branch mode requires identical references before and
after the update. Both modes verify that `caller.rs` itself did not change.
`result.json` retains command outputs and the binary SHA-256. Neither mode needs
semantic models or writes application source.

## Other regression suites

```sh
cargo test --locked --test core_reference_regression_test --test cli_incremental_hnsw_test
cargo test --locked --test cli_tests_for_test --test cli_usages_strict_test --test cli_search_d4_test
cargo test --locked --lib search::
```

The suites also cover partial vectors, embedder changes, reference-only files,
duplicate occurrences, HNSW aliases and stale sidecars, exact strict names,
type-only hierarchy edges, scoped limits and same-named call-graph sites.
