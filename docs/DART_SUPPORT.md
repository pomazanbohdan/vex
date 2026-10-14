# Dart support

This branch registers `.dart` as language ID20 without changing existing IDs.
The grammar is pinned to `tree-sitter-dart-orchard =0.7.0`.

## Extraction

- Classes, enums, mixins, named extensions and extension types are indexed.
- Top-level functions, methods, getters/setters and constructors use the existing
  Vex symbol schema. Callable body tokens include the declaration body rather
  than only the nested signature.
- Import sites retain their literal URI; they are not compiler-resolved imports.
- Call queries select invocation nodes. Bare identifier reads, comments and
  literal string text do not create call edges.
- Normal, member, conditional, cascade and generic constructor-like calls use
  the existing innermost named-function ownership and name-based resolver.
- Hierarchy extraction preserves `extends`, `implements` and mixin `uses` edges.
- AST refs skip comments and retain code inside string interpolation.
- Dart identifier nodes retain lowercase, short and legal keyword-like names.
- Canonical test paths include `*_test.dart` and `integration_test/`.
- `tests-for` admits Dart `main` after matching the test path; utilities still
  require `--include-fixtures`, and custom test patterns replace the defaults.
- Pattern skeletons register Dart declaration nodes. ast-grep remains the chosen
  structural-search route on hosts that apply the shared code-tooling policy.

## Boundaries

Dart has an optional compiler backend for `usages --strict` and the
`impact.channels.strict_refs` channel. Each query starts the SDK's Analysis
Server, maps indexed declarations to outline elements, and calls
`search.findElementReferences` with `includePotential=false`. The client waits
for the complete `search.results` stream and verifies the returned declaration
identity. Imports, prefixes, parts, private libraries, constructors, shadowing
and typed receivers therefore use compiler resolution. Dynamic/potential and
unresolved references are excluded. Multiple indexed declarations named alike
retain Vex's existing union semantics; each constituent search uses its exact
file and UTF-16 declaration offset. Searches run in batches of at most16; each
complete batch is converted before its SDK JSON is discarded. Required file
outlines gate readiness while unrelated global diagnostics may continue.

SDK JSON is validated in a header pass before payload allocation. Ignored
diagnostics do not build a `Value` tree; retained responses/events are parsed
from the original full line. Field order and duplicate-field handling match
the previous parser. Invalid escapes, numeric overflow, nesting limits and
trailing JSON remain errors. The validated discard visitor checks scalars
rather than using Serde's permissive `IgnoredAny` skip. The incoming queue
holds at most64 retained messages and applies backpressure to the reader.

Successive declaration batches keep the same SDK child and the original query
deadline. Search stream state is cleared after each call, including on errors.
A failed transport/request/search poisons the session; further requests fail
instead of reusing partial state.
Search completion requires a boolean `isLast`; malformed events fail immediately
before any search state is stored.
Each top-level strict query still owns a fresh child and source/context snapshot.
Cross-query SDK reuse requires a
separate core batch coordinator and is not enabled by this adapter change.

The SDK normalizes explicit getters/setters to their synthetic property variable.
Vex accepts this case only for GETTER/SETTER outlines and FIELD/TOP_LEVEL_VARIABLE
responses with the same file/name, offset0, UTF-16 name length, line1 and column1.
All ordinary declarations retain exact offset/kind checks. Requests remain keyed
by their original outline identities, preserving same-name properties in separate
owners even when their SDK response locations are identical.
[SDK accessor search handler](https://github.com/dart-lang/sdk/blob/3.13.5/pkg/analysis_server/lib/src/handler/legacy/search_find_element_references.dart#L25-L31)
and [SDK location converter](https://github.com/dart-lang/sdk/blob/3.13.5/pkg/analysis_server/lib/src/protocol_server.dart#L213-L234)
define this normalization and synthetic location.
Strict Dart target selection compares full indexed names with the query using
the FST's lowercase semantics; CamelCase subtoken postings do not add targets.

SDK discovery: explicit `VEX_DART_SDK` (SDK directory or Dart executable), then
ancestor `.fvm/flutter_sdk`, ancestor `.fvmrc` version in the user's
`fvm/versions`, then `dart` on PATH. An invalid explicit override returns
`Dart strict backend unavailable` without trying other SDKs. Ordinary search,
refs and call graphs remain operational without an SDK and retain name-based
resolution. Anonymous callbacks attach to the enclosing named callable.

Strict queries read existing package configs and pubspecs without writing
application files or running pub. Analysis caches live in the user's Vex cache
(`LOCALAPPDATA/vex/dart-analysis` on Windows, with the existing USERPROFILE
fallback). Empty environment values do not select a relative application path. Startup and project analysis are
paid per query; no semantic edges or semantic cache are persisted in the index.
The existing SDK AOT runtime and analysis snapshot are used when both are present;
otherwise the configured Dart executable starts the JIT snapshot. No runtime is
downloaded or installed.
The default process deadline is 180 seconds; `VEX_DART_TIMEOUT_MS` sets a deadline
between 1 and 600000 ms. Success, timeout and errors all kill and reap the
isolated server. No dependency or SDK installation is performed.

Before starting, indexed Dart content hashes must match live source discovery
and the manifest, including the index's 1 MiB file limit. Source and project
context hashes (pubspecs, locks, package configs, analysis options and FVM config)
are checked again before results are returned. Stale or concurrently edited
source produces an explicit backend error, including with `--no-stale-check`.
Protocol UTF-16 reference offsets are validated against exact UTF-8 source
bytes. References are grouped by file and sorted by UTF-16 offset; each source is
read once for conversion and traversed once with a character cursor. Surrogate
splits, offsets beyond EOF and incorrect SDK line numbers fail explicitly. The
query deadline is checked during declaration collection, conversion and final
source/context verification. Only references inside indexed project files survive; external files
and ignored/generated files outside that set are outside query coverage.
Dependency source files outside the indexed project are read live by the SDK;
their content is not covered by the project hash snapshot.

JSON usages reports `_meta.vex.dev/reference_backend`; impact reports
`channels.strict_refs.backend`. Both include backend/protocol identity, SDK
executable, freshness policy, resolved-only coverage and elapsed milliseconds.
`execution_runtime` reports the actual AOT/JIT mode, executable and snapshot.
Text usages prints the backend and timing on stderr. Missing SDK, incompatible
protocol, unresolvable declaration mapping and timeout return unavailable
instead of ordinary name matches. Native languages keep persisted binder edges.
In mixed projects, a name defined only in native languages uses those persisted
edges without requiring the Dart SDK.

Orchard0.7.0 can recover from unsupported syntax and produce partial trees. A
successful parse or index does not guarantee complete Dart3.13 syntax coverage.
The earlier vida_mobile parser comparison observed ERROR/MISSING recovery on
valid Dart source, including higher-order and null-asserted invocations. This
branch integrates the published grammar; it does not rewrite that grammar.

## Verification

`cargo test --test dart_query_test` covers loading, symbols/imports, ownership,
generic/conditional/cascade calls, non-call identifier filtering, body tokens,
type aliases, hierarchy kinds and string interpolation.

The language is also included in shared-tree, parser-count, query-compilation
and snapshot coverage. On Windows the release uses `--features gpu-directml`
to preserve the installed GPU capability.

`cargo test --test dart_strict_refs_test` covers real process lookup, impact,
constructor mapping, scope filtering, stale-source rejection, timeout recovery
and missing-SDK ordinary-search behavior. Set `VEX_DART_SDK` for compiler tests;
without it they emit a skip note. `cargo test --lib channel::dart::tests` also
checks exact library/member/private/constructor identity against a real server
using prefixes, package imports, parts, local shadows, dynamic receivers and
Unicode. No protocol mocks stand in for compiler proof.

`cargo test --test dart_accessors_test` covers paired/isolated/top-level accessors,
same-line properties in separate owners, unions spanning multiple batches,
Unicode reads/writes, compound assignment,
local/dynamic shadows, scope filtering and runtime provenance with the real SDK.
