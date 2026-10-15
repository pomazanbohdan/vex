# Reproduce Dart support

Run these commands from this PR checkout. Python 3.10+ and the normal Vex build
prerequisites are required. Replace `/path/to/dart-sdk` with an existing SDK;
the script does not install an SDK or packages. On Windows use `python` and the
`.exe` binary suffix.

```sh
git worktree add --detach ../vex-before ca47682f0fd6691e490c6ee7d28f8d29829e7c15
cargo build --release --locked --bin vex --manifest-path ../vex-before/Cargo.toml
cargo build --release --locked --bin vex
python3 scripts/reproduce-dart.py --vex ../vex-before/target/release/vex --expect before --run-dir ../dart-before
python3 scripts/reproduce-dart.py --vex target/release/vex --expect after --sdk /path/to/dart-sdk --run-dir ../dart-after
```

Each run directory must be new. The script writes its two Dart files, a native
control, configuration, isolated index and `result.json` under that directory.
It does not write application source or the user's normal Vex cache.

The upstream check passes when the native control is indexed but Dart
declarations, refs, hierarchy and test entrypoints are absent. The branch check
passes when those Dart facts are present and the real SDK resolves both helper
references with compiler provenance. The source file hashes must stay unchanged.
Both modes return zero only when their declared old/new expectation is observed;
`result.json` retains every command, exit code and output.

## Regression suites

```sh
cargo test --locked --test dart_query_test --test dart_core_regression_test --test cli_tests_for_test
VEX_DART_SDK=/path/to/dart-sdk cargo test --locked --test dart_strict_refs_test --test dart_accessors_test
```

The native-binder guide exception is proposed for maintainer review. This PR
retains the real SDK backend rather than introducing a name-only native result
under the strict contract.
